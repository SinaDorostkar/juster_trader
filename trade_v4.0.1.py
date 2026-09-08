import os
import time
import json
import math
import random
import logging
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import turso_serverless


# ============================================================
# V4.0 HYBRID TCN + LSTM + ATTENTION CRYPTO SIGNAL BOT
# ============================================================
# Architecture:
#   1H OHLCV  -> TCN -> BiLSTM -> Attention
#   4H OHLCV  -> TCN -> BiLSTM -> Attention
#   1D OHLCV  -> TCN -> BiLSTM -> Attention
#   Microstructure history -> TCN -> BiLSTM -> Attention
#   BTC/regime + current microstructure -> context MLP
#                         |
#                      Fusion
#                    /       \
#             win probability  event time
#
# V4 deliberately keeps "prediction" separate from "signal".
# Every clean prediction can become a training observation, while only
# predictions passing the risk/EV/exposure filters become Telegram signals.
#
# The .pth checkpoint is written under models/ and should be committed by
# GitHub Actions after a successful retraining run.
# ============================================================

VERSION = "4.0.1"
ARCHITECTURE = "V4.0-HYBRID-TCN-LSTM-ATTN"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
YOUR_CHAT_ID = os.getenv("YOUR_CHAT_ID", "")
CHANNEL_ID = os.getenv("CHANNEL_ID", "")

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

BINANCE_FAPI = "https://fapi.binance.com"

STATIC_WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SHIBUSDT",
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "TRXUSDT", "BNBUSDT"
]

SCAN_TOP_N = 40
MIN_24H_VOLUME_USD = 5_000_000

STRATEGIES = {
    "day": {
        "primary": "1h",
        "target_atr": 1.8,
        "stop_atr": 1.0,
        "timeout_h": 240,
        "model_path": "models/brain_v401_day.pth",
        "threshold_default": 0.70,
        "seq": {"1h": 168, "4h": 90, "1d": 60},
        "micro_len": 24,
    },
    "swing": {
        "primary": "4h",
        "target_atr": 3.5,
        "stop_atr": 1.8,
        "timeout_h": 720,
        "model_path": "models/brain_v401_swing.pth",
        "threshold_default": 0.70,
        "seq": {"1h": 72, "4h": 90, "1d": 60},
        "micro_len": 24,
    },
}

PRICE_FEATURES = [
    "ret1", "ret3", "ret6", "ret12",
    "range_pct", "body_pct",
    "upper_wick_pct", "lower_wick_pct",
    "atr_pct", "vol_rel", "rsi", "macd_pct",
    "bb_width", "adx", "ema_spread", "obv_slope"
]

MICRO_FEATURES = [
    "obi5", "obi10", "obi20", "obi50", "obi100",
    "spread", "funding", "oi_change", "price_ret", "valid"
]

STATIC_FEATURES = [
    "obi20", "spread", "funding", "oi_change",
    "btc_ret24", "btc_vol24", "btc_ema_spread"
]

PRICE_DIM = len(PRICE_FEATURES)
MICRO_DIM = len(MICRO_FEATURES)
STATIC_DIM = len(STATIC_FEATURES)

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "0.60"))
MAX_MAJOR_EXPOSURE = float(os.getenv("MAX_MAJOR_EXPOSURE", "0.40"))
MAX_ALT_EXPOSURE = float(os.getenv("MAX_ALT_EXPOSURE", "0.25"))
MAX_SINGLE_POSITION = float(os.getenv("MAX_SINGLE_POSITION", "0.20"))

MAJORS = {"BTCUSDT", "ETHUSDT"}

MIN_EV = float(os.getenv("MIN_EV", "0.002"))
MIN_MICRO_HISTORY = int(os.getenv("MIN_MICRO_HISTORY", "24"))

RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "50"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "250"))
MIN_CLASS_COUNT = int(os.getenv("MIN_CLASS_COUNT", "30"))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "5000"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "120"))

HTTP_TIMEOUT = 20
MAX_RETRIES = 4

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("trade_v4")

session = requests.Session()
session.headers.update({"User-Agent": "crypto-v4-bot/4.0"})


# ============================================================
# DATABASE
# ============================================================

_db = None


def db_connect():
    global _db

    if _db is not None:
        return _db

    if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
        raise RuntimeError("Missing TURSO_DATABASE_URL or TURSO_AUTH_TOKEN")

    _db = turso_serverless.connect(
        TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN
    )
    return _db


def db_execute(sql, params=()):
    db = db_connect()
    cur = db.cursor()
    cur.execute(sql, params)

    try:
        db.commit()
    except Exception as exc:
        log.error("DB commit failed for statement %r: %s", sql[:120], exc)
        raise

    return cur


def db_query(sql, params=()):
    db = db_connect()
    cur = db.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


def init_db():
    # V4.0.1 uses isolated tables so contaminated V4.0/V4.0.0 observations
    # can never silently enter the clean causal training set.
    db_execute("""
    CREATE TABLE IF NOT EXISTS predictions_v401 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        predicted_at INTEGER NOT NULL,
        source_close_ms INTEGER NOT NULL,
        decision_entry_price REAL NOT NULL,
        target_price REAL NOT NULL,
        stop_price REAL NOT NULL,
        live_entry_price REAL,
        live_target_price REAL,
        live_stop_price REAL,
        atr REAL NOT NULL,
        predicted_prob REAL,
        calibrated_prob REAL,
        predicted_hours REAL,
        threshold REAL,
        signaled INTEGER NOT NULL DEFAULT 0,
        resolved INTEGER NOT NULL DEFAULT 0,
        outcome INTEGER,
        outcome_type TEXT,
        resolved_at INTEGER,
        hours_to_result REAL,
        train_candidate INTEGER NOT NULL DEFAULT 0,
        trained INTEGER NOT NULL DEFAULT 0,
        model_version TEXT NOT NULL,
        sequence_json TEXT NOT NULL,
        static_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)

    db_execute("""
    CREATE TABLE IF NOT EXISTS market_snapshots_v401 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        snapshot_ms INTEGER NOT NULL,
        price REAL NOT NULL,
        obi5 REAL, obi10 REAL, obi20 REAL, obi50 REAL, obi100 REAL,
        spread_bps REAL, funding REAL,
        open_interest REAL, oi_change REAL,
        btc_ret24 REAL, btc_vol24 REAL, btc_ema_spread REAL,
        UNIQUE(symbol, snapshot_ms)
    )
    """)

    db_execute("""
    CREATE TABLE IF NOT EXISTS model_state_v401 (
        strategy TEXT PRIMARY KEY,
        model_version TEXT,
        trained_samples INTEGER DEFAULT 0,
        last_train_ms INTEGER DEFAULT 0,
        new_resolved_since_train INTEGER DEFAULT 0,
        threshold REAL, temperature REAL,
        test_logloss REAL, test_brier REAL,
        test_precision REAL, test_ev REAL
    )
    """)

    db_execute("""CREATE INDEX IF NOT EXISTS idx_v401_predictions_resolution
    ON predictions_v401(resolved, strategy)""")
    db_execute("""CREATE INDEX IF NOT EXISTS idx_v401_predictions_source
    ON predictions_v401(symbol, strategy, source_close_ms)""")
    db_execute("""CREATE INDEX IF NOT EXISTS idx_v401_snapshots
    ON market_snapshots_v401(symbol, snapshot_ms)""")


# ============================================================
# BINANCE HTTP
# ============================================================

def api_get(base, path, params=None):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                base + path,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 429:
                time.sleep(min(15, 2 ** attempt))
                continue

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (2 ** attempt))

    raise RuntimeError(f"Binance API failed: {path}: {last_error}")


def get_exchange_info():
    return api_get(BINANCE_FAPI, "/fapi/v1/exchangeInfo")


def valid_futures_symbols():
    data = get_exchange_info()

    result = {}

    for item in data.get("symbols", []):
        symbol = item.get("symbol", "")

        if item.get("status") != "TRADING":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        contract_type = item.get("contractType", "")
        if contract_type != "PERPETUAL":
            continue

        base = item.get("baseAsset", "").upper()

        # Futures universe: USDT-margined perpetual contracts only.
        if any(x in base for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue

        result[symbol] = item

    return result


def get_top_symbols():
    valid = valid_futures_symbols()
    tickers = api_get(BINANCE_FAPI, "/fapi/v1/ticker/24hr")

    candidates = []

    for ticker in tickers:
        symbol = ticker.get("symbol", "")

        if symbol not in valid:
            continue

        try:
            volume = float(ticker.get("quoteVolume", 0))
        except Exception:
            continue

        if volume >= MIN_24H_VOLUME_USD:
            candidates.append((symbol, volume))

    candidates.sort(key=lambda x: x[1], reverse=True)

    result = []

    # Always prioritize the original watchlist if valid.
    for symbol in STATIC_WATCHLIST:
        if symbol in valid and symbol not in result:
            result.append(symbol)

    for symbol, _ in candidates:
        if symbol not in result:
            result.append(symbol)

        if len(result) >= SCAN_TOP_N:
            break

    return result[:SCAN_TOP_N], valid


def get_klines(symbol, interval, limit=500, start_ms=None, end_ms=None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(int(limit), 1000)
    }

    if start_ms is not None:
        params["startTime"] = int(start_ms)

    if end_ms is not None:
        params["endTime"] = int(end_ms)

    raw = api_get(BINANCE_FAPI, "/fapi/v1/klines", params)

    columns = [
        "open_ms", "open", "high", "low", "close", "volume",
        "close_ms", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"
    ]

    df = pd.DataFrame(raw, columns=columns)

    if df.empty:
        return df

    numeric = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_base", "taker_quote"
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_ms"] = pd.to_numeric(df["open_ms"], errors="coerce").astype("int64")
    df["close_ms"] = pd.to_numeric(df["close_ms"], errors="coerce").astype("int64")

    # CRITICAL: never use the currently forming candle.
    now_ms = int(time.time() * 1000)
    df = df[df["close_ms"] < now_ms].copy()

    return df.reset_index(drop=True)


def get_price(symbol):
    data = api_get(
        BINANCE_FAPI,
        "/fapi/v1/ticker/price",
        {"symbol": symbol}
    )

    return float(data["price"])


def get_depth(symbol, limit=100):
    return api_get(
        BINANCE_FAPI,
        "/fapi/v1/depth",
        {"symbol": symbol, "limit": limit}
    )


def get_funding(symbol):
    try:
        data = api_get(
            BINANCE_FAPI,
            "/fapi/v1/premiumIndex",
            {"symbol": symbol}
        )

        return float(data.get("lastFundingRate", 0) or 0)

    except Exception:
        return 0.0


def get_open_interest(symbol):
    try:
        data = api_get(
            BINANCE_FAPI,
            "/fapi/v1/openInterest",
            {"symbol": symbol}
        )

        return float(data.get("openInterest", 0) or 0)

    except Exception:
        return 0.0


def orderbook_features(depth):
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])

    values = {}

    for level in (5, 10, 20, 50, 100):
        bid_notional = sum(
            float(price) * float(quantity)
            for price, quantity in bids[:level]
        )

        ask_notional = sum(
            float(price) * float(quantity)
            for price, quantity in asks[:level]
        )

        denominator = bid_notional + ask_notional

        values[level] = (
            (bid_notional - ask_notional) / denominator
            if denominator else 0.0
        )

    if bids and asks:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])

        midpoint = (best_bid + best_ask) / 2

        spread_bps = (
            ((best_ask - best_bid) / midpoint) * 10000
            if midpoint else 0.0
        )
    else:
        spread_bps = 0.0

    return values, spread_bps


# ============================================================
# PRICE FEATURES
# ============================================================

def calc_rsi(series, period=14):
    delta = series.diff()

    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gains / losses.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def make_price_features(df):
    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    open_ = x["open"]
    volume = x["volume"]

    log_close = np.log(close.replace(0, np.nan))

    x["ret1"] = log_close.diff(1)
    x["ret3"] = log_close.diff(3)
    x["ret6"] = log_close.diff(6)
    x["ret12"] = log_close.diff(12)

    x["range_pct"] = (high - low) / close
    x["body_pct"] = (close - open_) / open_.replace(0, np.nan)

    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_, close], axis=1).min(axis=1)

    x["upper_wick_pct"] = (high - candle_top) / close
    x["lower_wick_pct"] = (candle_bottom - low) / close

    true_range = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    x["atr_pct"] = (
        true_range.rolling(14).mean() /
        close.replace(0, np.nan)
    )

    volume_median = volume.rolling(48).median()

    x["vol_rel"] = np.log1p(
        volume / volume_median.replace(0, np.nan)
    )

    x["rsi"] = (calc_rsi(close) - 50) / 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26

    x["macd_pct"] = macd / close.replace(0, np.nan)

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    x["bb_width"] = (
        4 * std / middle.replace(0, np.nan)
    )

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(
        (plus_dm > minus_dm) & (plus_dm > 0),
        0.0
    )

    minus_dm = minus_dm.where(
        (minus_dm > plus_dm) & (minus_dm > 0),
        0.0
    )

    atr14 = true_range.rolling(14).mean()

    plus_di = (
        100 *
        plus_dm.rolling(14).mean() /
        atr14.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm.rolling(14).mean() /
        atr14.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    x["adx"] = dx.rolling(14).mean() / 100

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    x["ema_spread"] = (
        (ema20 - ema50) /
        close.replace(0, np.nan)
    )

    direction = np.sign(close.diff()).fillna(0)

    obv = (direction * volume).cumsum()

    x["obv_slope"] = (
        obv.diff(12) /
        volume.rolling(12).sum().replace(0, np.nan)
    )

    features = x[PRICE_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features.clip(-10, 10)


def normalize_price_sequence(array):
    array = np.asarray(array, dtype=np.float32)

    median = np.nanmedian(array, axis=0)

    q1 = np.nanpercentile(array, 25, axis=0)
    q3 = np.nanpercentile(array, 75, axis=0)

    scale = np.maximum(q3 - q1, 1e-4)

    result = (array - median) / scale

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=5.0,
        neginf=-5.0
    )

    return np.clip(result, -8, 8).astype(np.float32)


def btc_regime_as_of(as_of_ms=None):
    # BTC context is always calculated from data available at the sample
    # decision time. This prevents today's BTC regime leaking into old rows.
    df = get_klines("BTCUSDT", "1h", 100, end_ms=as_of_ms)
    if len(df) < 30:
        return 0.0, 0.0, 0.0
    close = df["close"]
    ret24 = float(np.log(close.iloc[-1] / close.iloc[-25])) if len(close) >= 25 else 0.0
    vol24 = float(np.log(close).diff().rolling(24).std().iloc[-1] or 0.0)
    ema24 = close.ewm(span=24, adjust=False).mean().iloc[-1]
    ema72 = close.ewm(span=72, adjust=False).mean().iloc[-1]
    ema_spread = float((ema24 - ema72) / close.iloc[-1])
    return ret24, vol24, ema_spread

def btc_regime():
    return btc_regime_as_of(None)


# ============================================================
# HISTORICAL MICROSTRUCTURE
# ============================================================

def previous_snapshot(symbol, before_ms=None):
    if before_ms is None:
        rows = db_query("""SELECT snapshot_ms, open_interest FROM market_snapshots_v401 WHERE symbol=? ORDER BY snapshot_ms DESC LIMIT 1""", (symbol,))
    else:
        rows = db_query("""SELECT snapshot_ms, open_interest FROM market_snapshots_v401 WHERE symbol=? AND snapshot_ms < ? ORDER BY snapshot_ms DESC LIMIT 1""", (symbol, int(before_ms)))
    return rows[0] if rows else None


def collect_snapshot(symbol, btc_context):
    now_ms = int(time.time() * 1000)

    # One observation per UTC hour.
    bucket_ms = (
        now_ms // 3_600_000
    ) * 3_600_000

    exists = db_query("""
        SELECT id
        FROM market_snapshots_v401
        WHERE symbol=? AND snapshot_ms=?
    """, (symbol, bucket_ms))

    if exists:
        return

    try:
        price = get_price(symbol)

        depth = get_depth(symbol, 100)
        obi, spread_bps = orderbook_features(depth)

        funding = get_funding(symbol)
        open_interest = get_open_interest(symbol)

        previous = previous_snapshot(symbol)

        previous_oi = (
            float(previous[1])
            if previous and previous[1] is not None
            else 0.0
        )

        if previous_oi:
            oi_change = (
                (open_interest - previous_oi) /
                abs(previous_oi)
            )
        else:
            oi_change = 0.0

        btc_ret24, btc_vol24, btc_ema_spread = btc_context

        db_execute("""
        INSERT OR IGNORE INTO market_snapshots_v401 (
            symbol, snapshot_ms, price,
            obi5, obi10, obi20, obi50, obi100,
            spread_bps, funding,
            open_interest, oi_change,
            btc_ret24, btc_vol24, btc_ema_spread
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol,
            bucket_ms,
            price,

            obi[5],
            obi[10],
            obi[20],
            obi[50],
            obi[100],

            spread_bps,
            funding,

            open_interest,
            oi_change,

            btc_ret24,
            btc_vol24,
            btc_ema_spread
        ))

    except Exception as exc:
        log.warning("Snapshot failed for %s: %s", symbol, exc)


def build_micro_sequence(symbol, length, as_of_ms):
    rows = db_query("""
        SELECT snapshot_ms, obi5, obi10, obi20, obi50, obi100,
               spread_bps, funding, oi_change, price
        FROM market_snapshots_v401
        WHERE symbol=? AND snapshot_ms <= ?
        ORDER BY snapshot_ms DESC LIMIT ?
    """, (symbol, int(as_of_ms), length))
    rows = list(reversed(rows))
    if len(rows) < MIN_MICRO_HISTORY:
        return None, None
    values = []
    for index, row in enumerate(rows):
        snapshot_ms, obi5, obi10, obi20, obi50, obi100, spread_bps, funding, oi_change, price = row
        if index == 0:
            price_return = 0.0
        else:
            previous_price = float(rows[index - 1][9]); current_price = float(price)
            price_return = math.log(current_price / previous_price) if previous_price > 0 and current_price > 0 else 0.0
        values.append([float(obi5 or 0), float(obi10 or 0), float(obi20 or 0), float(obi50 or 0), float(obi100 or 0), float(spread_bps or 0) / 100.0, float(funding or 0) * 1000.0, float(oi_change or 0) * 10.0, price_return, 1.0])
    array = np.asarray(values, dtype=np.float32)
    array[:, :9] = np.clip(array[:, :9], -10, 10)
    return array, int(rows[-1][0])

def micro_static_as_of(symbol, as_of_ms):
    rows = db_query("""
        SELECT obi20, spread_bps, funding, oi_change
        FROM market_snapshots_v401
        WHERE symbol=? AND snapshot_ms <= ?
        ORDER BY snapshot_ms DESC LIMIT 1
    """, (symbol, int(as_of_ms)))
    if not rows:
        return [0.0, 0.0, 0.0, 0.0]
    row = rows[0]
    return [float(row[0] or 0) , float(row[1] or 0) / 100.0, float(row[2] or 0) * 1000.0, float(row[3] or 0) * 10.0]

def current_micro_static(symbol):
    return micro_static_as_of(symbol, int(time.time() * 1000))


# ============================================================
# MULTI-TIMEFRAME SAMPLE
# ============================================================

def make_price_sequence(symbol, interval, length, end_ms=None):
    df = get_klines(symbol, interval, min(length + 100, 1000), end_ms=end_ms)
    if len(df) < length + 20:
        return None
    if end_ms is not None:
        df = df[df["close_ms"] <= int(end_ms)].copy()
    if len(df) < length + 20:
        return None
    feature_df = make_price_features(df).iloc[-length:].bfill().ffill().fillna(0)
    sequence = normalize_price_sequence(feature_df.to_numpy())
    source_close_ms = int(df["close_ms"].iloc[-1])
    close_price = float(df["close"].iloc[-1])
    true_range = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])
    return (sequence, source_close_ms, close_price, atr) if np.isfinite(atr) else None

def build_sample(symbol, strategy, btc_context=None, decision_close_ms=None):
    cfg = STRATEGIES[strategy]
    # For live inference, anchor all modalities to the latest CLOSED primary
    # candle. For historical use, caller can provide an explicit anchor.
    if decision_close_ms is None:
        primary_probe = get_klines(symbol, cfg["primary"], 5)
        if primary_probe.empty:
            return None
        decision_close_ms = int(primary_probe["close_ms"].iloc[-1])
    sequences = {}
    entry = None; atr = None
    for interval, length in cfg["seq"].items():
        result = make_price_sequence(symbol, interval, length, end_ms=decision_close_ms)
        if result is None:
            return None
        sequence, source_close, close_price, this_atr = result
        sequences[interval] = sequence
        if interval == cfg["primary"]:
            entry = close_price; atr = this_atr
    as_of_ms = int(decision_close_ms) + 1
    micro, micro_time = build_micro_sequence(symbol, cfg["micro_len"], as_of_ms)
    if micro is None:
        return None
    static = micro_static_as_of(symbol, as_of_ms)
    btc_context = btc_regime_as_of(as_of_ms)
    static.extend(list(btc_context))
    return {
        "seqs": sequences, "micro": micro,
        "static": np.asarray(static, dtype=np.float32),
        "source_close": int(decision_close_ms),
        "entry": float(entry),
        "atr": max(float(atr), float(entry) * 1e-5),
    }


def serialize_sample(sample):
    payload = {
        "1h": sample["seqs"]["1h"].tolist(),
        "4h": sample["seqs"]["4h"].tolist(),
        "1d": sample["seqs"]["1d"].tolist(),
        "micro": sample["micro"].tolist(),
    }

    return json.dumps(
        payload,
        separators=(",", ":")
    )


# ============================================================
# MODEL
# ============================================================

class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        channels=48,
        hidden=48,
        dropout=0.15
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            input_dim,
            channels,
            kernel_size=1
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                dilation=1
            ),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=4,
                dilation=4
            )
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(channels)
            for _ in self.convs
        ])

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(
            hidden * 2,
            1
        )

        self.output = nn.Linear(
            hidden * 2,
            64
        )

    def forward(self, x):
        # x = [batch, time, features]

        z = self.input_projection(
            x.transpose(1, 2)
        )

        for conv, norm in zip(
            self.convs,
            self.norms
        ):
            residual = z

            y = conv(z)
            y = F.gelu(y)

            y = y.transpose(1, 2)
            y = norm(y)
            y = y.transpose(1, 2)

            z = residual + self.dropout(y)

        z = z.transpose(1, 2)

        hidden, _ = self.lstm(z)

        weights = torch.softmax(
            self.attention(hidden).squeeze(-1),
            dim=1
        ).unsqueeze(-1)

        pooled = (
            hidden * weights
        ).sum(dim=1)

        return self.output(pooled)


class V4Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1h = TemporalEncoder(
            PRICE_DIM
        )

        self.encoder_4h = TemporalEncoder(
            PRICE_DIM
        )

        self.encoder_1d = TemporalEncoder(
            PRICE_DIM
        )

        self.encoder_micro = TemporalEncoder(
            MICRO_DIM,
            channels=40,
            hidden=40
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(STATIC_DIM, 32),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(32, 32),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                64 * 4 + 32,
                192
            ),

            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(192, 96),
            nn.GELU()
        )

        self.win_head = nn.Linear(96, 1)
        self.time_head = nn.Linear(96, 1)

    def forward(
        self,
        x1h,
        x4h,
        x1d,
        xmicro,
        static
    ):
        a = self.encoder_1h(x1h)
        b = self.encoder_4h(x4h)
        c = self.encoder_1d(x1d)
        d = self.encoder_micro(xmicro)

        s = self.static_encoder(static)

        combined = torch.cat(
            [a, b, c, d, s],
            dim=1
        )

        fused = self.fusion(combined)

        win_logit = self.win_head(
            fused
        ).squeeze(1)

        event_time = self.time_head(
            fused
        ).squeeze(1)

        return win_logit, event_time


# ============================================================
# CHECKPOINTS
# ============================================================

def load_checkpoint(strategy):
    path = Path(
        STRATEGIES[strategy]["model_path"]
    )

    if not path.exists():
        return (
            None,
            1.0,
            STRATEGIES[strategy]["threshold_default"],
            {}
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=DEVICE
        )

        if checkpoint.get("architecture") != ARCHITECTURE:
            log.warning(
                "Ignoring incompatible checkpoint: %s",
                path
            )

            return (
                None,
                1.0,
                STRATEGIES[strategy]["threshold_default"],
                {}
            )

        model = V4Model().to(DEVICE)

        model.load_state_dict(
            checkpoint["model_state"]
        )

        model.eval()

        checkpoint_temperature = float(checkpoint.get("temperature", 1.0))
        checkpoint_threshold = float(checkpoint.get("threshold", STRATEGIES[strategy]["threshold_default"]))
        state = db_query("SELECT threshold, temperature FROM model_state_v401 WHERE strategy=?", (strategy,))
        if state and state[0][0] is not None:
            db_threshold = float(state[0][0])
        else:
            db_threshold = checkpoint_threshold
        if state and state[0][1] is not None:
            db_temperature = float(state[0][1])
        else:
            db_temperature = checkpoint_temperature
        return model, db_temperature, db_threshold, checkpoint

    except Exception as exc:
        log.warning(
            "Could not load checkpoint %s: %s",
            path,
            exc
        )

        return (
            None,
            1.0,
            STRATEGIES[strategy]["threshold_default"],
            {}
        )


def save_checkpoint(
    strategy,
    model,
    temperature,
    threshold,
    metrics,
    sample_count
):
    path = Path(
        STRATEGIES[strategy]["model_path"]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = path.with_suffix(".tmp")

    checkpoint = {
        "architecture": ARCHITECTURE,
        "version": VERSION,
        "strategy": strategy,
        "feature_version": 2,
        "data_tables": "v401",
        "market_type": "USDT-margined perpetual futures",

        "model_state": model.state_dict(),

        "temperature": float(temperature),
        "threshold": float(threshold),

        "trained_samples": int(sample_count),

        "trained_at": int(
            time.time() * 1000
        ),

        "metrics": metrics
    }

    torch.save(
        checkpoint,
        temporary
    )

    temporary.replace(path)


# ============================================================
# TRAINING DATA
# ============================================================

def decode_sample(sequence_json):
    data = json.loads(sequence_json)

    return (
        np.asarray(
            data["1h"],
            dtype=np.float32
        ),
        np.asarray(
            data["4h"],
            dtype=np.float32
        ),
        np.asarray(
            data["1d"],
            dtype=np.float32
        ),
        np.asarray(
            data["micro"],
            dtype=np.float32
        )
    )


def get_training_rows(strategy):
    rows = db_query("""
        SELECT id, symbol, strategy, predicted_at, source_close_ms,
               decision_entry_price, target_price, stop_price, atr,
               predicted_prob, calibrated_prob, predicted_hours, threshold,
               signaled, resolved, outcome, outcome_type, resolved_at,
               hours_to_result, train_candidate, trained, model_version,
               sequence_json, static_json
        FROM predictions_v401
        WHERE strategy=? AND resolved=1 AND outcome IS NOT NULL AND train_candidate=1
        ORDER BY source_close_ms ASC
    """, (strategy,))
    kept=[]; last_by_symbol={}; stride_ms=(6 if strategy=="day" else 12)*3600*1000
    for row in rows:
        source_time=int(row[4]); previous=last_by_symbol.get(row[1])
        if previous is not None and source_time-previous < stride_ms: continue
        kept.append(row); last_by_symbol[row[1]]=source_time
    return kept[-MAX_TRAIN_SAMPLES:] if len(kept)>MAX_TRAIN_SAMPLES else kept

def make_tensor_dataset(rows):
    x1h=[]; x4h=[]; x1d=[]; micro=[]; static=[]; y_win=[]; y_time=[]; ids=[]
    for row in rows:
        try:
            a,b,c,d=decode_sample(row[22]); s=np.asarray(json.loads(row[23]),dtype=np.float32)
            x1h.append(a); x4h.append(b); x1d.append(c); micro.append(d); static.append(s)
            y_win.append(float(row[15])); y_time.append(math.log1p(max(float(row[18] or 1.0),1e-3))); ids.append(int(row[0]))
        except Exception as exc:
            log.warning("Skipping malformed training row %s: %s", row[0] if row else "?", exc)
    if not x1h: return None
    return tuple(torch.tensor(np.asarray(v),dtype=torch.float32) for v in (x1h,x4h,x1d,micro,static)) + (torch.tensor(np.asarray(y_win),dtype=torch.float32), torch.tensor(np.asarray(y_time),dtype=torch.float32), ids)


def chronological_split(rows):
    if len(rows) < MIN_TRAIN_SAMPLES:
        return None

    rows = sorted(
        rows,
        key=lambda x: int(x[4])
    )

    n = len(rows)

    first_cut = int(n * 0.65)
    second_cut = int(n * 0.80)

    first_time = int(
        rows[first_cut][4]
    )

    second_time = int(
        rows[second_cut][4]
    )

    # Purge a full maximum label horizon around boundaries.
    purge_ms = (
        max(
            STRATEGIES[row[2]]["timeout_h"]
            for row in rows
        )
        * 3600
        * 1000
    )

    train = [
        row
        for row in rows[:first_cut]
        if int(row[4]) <
        first_time - purge_ms
    ]

    validation = [
        row
        for row in rows[first_cut:second_cut]
        if (
            int(row[4]) >
            first_time + purge_ms
        )
    ]

    test = [
        row
        for row in rows[second_cut:]
        if int(row[4]) >
        second_time + purge_ms
    ]

    return train, validation, test


# ============================================================
# MODEL EVALUATION / CALIBRATION
# ============================================================

def evaluate_model(
    model,
    dataset,
    temperature=1.0
):
    if dataset is None:
        return {}

    model.eval()

    (
        x1h,
        x4h,
        x1d,
        micro,
        static,
        y_win,
        y_time,
        _
    ) = dataset

    with torch.no_grad():
        logits, predicted_time = model(
            x1h,
            x4h,
            x1d,
            micro,
            static
        )

    temperature = max(
        float(temperature),
        0.05
    )

    probabilities = torch.sigmoid(
        logits / temperature
    ).numpy()

    labels = y_win.numpy()

    epsilon = 1e-7

    logloss = float(
        -(
            labels *
            np.log(probabilities + epsilon)
            +
            (1 - labels) *
            np.log(
                1 - probabilities + epsilon
            )
        ).mean()
    )

    brier = float(
        np.mean(
            (probabilities - labels) ** 2
        )
    )

    return {
        "logloss": logloss,
        "brier": brier,
        "probabilities": probabilities,
        "labels": labels,
        "logits": logits.numpy(),
        "predicted_time": predicted_time.numpy()
    }


def fit_temperature(model, validation):
    if validation is None:
        return 1.0

    model.eval()

    (
        x1h,
        x4h,
        x1d,
        micro,
        static,
        y_win,
        _,
        _
    ) = validation

    with torch.no_grad():
        logits, _ = model(
            x1h,
            x4h,
            x1d,
            micro,
            static
        )

    if len(
        torch.unique(y_win)
    ) < 2:
        return 1.0

    log_temperature = torch.tensor(
        [0.0],
        dtype=torch.float32,
        requires_grad=True
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50
    )

    loss_function = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        loss = loss_function(
            logits / temperature,
            y_win
        )

        loss.backward()

        return loss

    try:
        optimizer.step(closure)

        return float(
            torch.exp(
                log_temperature
            )
            .detach()
            .clamp(0.2, 5.0)
            .item()
        )

    except Exception:
        return 1.0


def choose_threshold(
    model,
    validation,
    temperature,
    strategy
):
    if validation is None:
        return STRATEGIES[strategy][
            "threshold_default"
        ]

    metrics = evaluate_model(
        model,
        validation,
        temperature
    )

    probabilities = metrics["probabilities"]
    labels = metrics["labels"]

    best_threshold = STRATEGIES[strategy][
        "threshold_default"
    ]

    best_ev = -float("inf")

    tp_multiple = STRATEGIES[strategy][
        "target_atr"
    ]

    sl_multiple = STRATEGIES[strategy][
        "stop_atr"
    ]

    reward_risk = (
        tp_multiple /
        sl_multiple
    )

    for threshold in np.arange(
        0.55,
        0.91,
        0.01
    ):
        selected = (
            probabilities >= threshold
        )

        if (
            selected.sum() <
            max(10, int(len(labels) * 0.03))
        ):
            continue

        ev = float(
            np.mean(
                np.where(
                    labels[selected] > 0.5,
                    reward_risk,
                    -1.0
                )
            )
        ) - 0.03

        if ev > best_ev:
            best_ev = ev
            best_threshold = float(
                threshold
            )

    return best_threshold


# ============================================================
# TRAINING
# ============================================================

def train_strategy(strategy):
    rows = get_training_rows(strategy)

    if len(rows) < MIN_TRAIN_SAMPLES:
        log.info(
            "%s: %d samples; need %d",
            strategy,
            len(rows),
            MIN_TRAIN_SAMPLES
        )
        return False

    wins = sum(
        int(row[15])
        for row in rows
    )

    losses = len(rows) - wins

    if (
        wins < MIN_CLASS_COUNT or
        losses < MIN_CLASS_COUNT
    ):
        log.info(
            "%s: class balance insufficient: wins=%d losses=%d",
            strategy,
            wins,
            losses
        )
        return False

    split = chronological_split(rows)

    if split is None:
        return False

    train_rows, validation_rows, test_rows = split

    if (
        len(train_rows) < 100 or
        len(validation_rows) < 30 or
        len(test_rows) < 30
    ):
        log.info(
            "%s: split too small: train=%d val=%d test=%d",
            strategy,
            len(train_rows),
            len(validation_rows),
            len(test_rows)
        )
        return False

    train = make_tensor_dataset(
        train_rows
    )

    validation = make_tensor_dataset(
        validation_rows
    )

    test = make_tensor_dataset(
        test_rows
    )

    if (
        train is None or
        validation is None or
        test is None
    ):
        return False

    model = V4Model().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0015,
        weight_decay=1e-4
    )

    win_loss = nn.BCEWithLogitsLoss()
    time_loss = nn.SmoothL1Loss()

    (
        x1h,
        x4h,
        x1d,
        micro,
        static,
        y_win,
        y_time,
        _
    ) = train

    (
        vx1h,
        vx4h,
        vx1d,
        vmicro,
        vstatic,
        vy_win,
        vy_time,
        _
    ) = validation

    best_state = None
    best_validation_loss = float("inf")

    patience = 0

    for epoch in range(
        TRAIN_EPOCHS
    ):
        model.train()

        optimizer.zero_grad()

        logits, predicted_time = model(
            x1h,
            x4h,
            x1d,
            micro,
            static
        )

        loss = (
            win_loss(logits, y_win)
            +
            0.20 *
            time_loss(
                predicted_time,
                y_time
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_logits, validation_time = model(
                vx1h,
                vx4h,
                vx1d,
                vmicro,
                vstatic
            )

            validation_loss = float(
                (
                    win_loss(
                        validation_logits,
                        vy_win
                    )
                    +
                    0.20 *
                    time_loss(
                        validation_time,
                        vy_time
                    )
                ).item()
            )

        if (
            validation_loss <
            best_validation_loss - 1e-5
        ):
            best_validation_loss = validation_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience = 0

        else:
            patience += 1

        if patience >= 18:
            break

    if best_state is None:
        return False

    model.load_state_dict(
        best_state
    )

    temperature = fit_temperature(
        model,
        validation
    )

    threshold = choose_threshold(
        model,
        validation,
        temperature,
        strategy
    )

    test_metrics = evaluate_model(
        model,
        test,
        temperature
    )

    selected = (
        test_metrics["probabilities"]
        >= threshold
    )

    if selected.any():
        precision = float(
            np.mean(
                test_metrics["labels"][selected]
            )
        )

        reward_risk = (
            STRATEGIES[strategy]["target_atr"] /
            STRATEGIES[strategy]["stop_atr"]
        )

        test_ev = float(
            np.mean(
                np.where(
                    test_metrics["labels"][selected] > 0.5,
                    reward_risk,
                    -1.0
                )
            )
        )
    else:
        precision = 0.0
        test_ev = 0.0

    metrics = {
        "test_logloss": test_metrics["logloss"],
        "test_brier": test_metrics["brier"],
        "test_precision": precision,
        "test_ev": test_ev,
        "test_signals": int(
            selected.sum()
        ),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows)
    }

    save_checkpoint(
        strategy,
        model,
        temperature,
        threshold,
        metrics,
        len(rows)
    )

    db_execute("""
    INSERT INTO model_state_v401 (
        strategy,
        model_version,
        trained_samples,
        last_train_ms,
        new_resolved_since_train,
        threshold,
        temperature,
        test_logloss,
        test_brier,
        test_precision,
        test_ev
    )
    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(strategy) DO UPDATE SET
        model_version=excluded.model_version,
        trained_samples=excluded.trained_samples,
        last_train_ms=excluded.last_train_ms,
        new_resolved_since_train=0,
        threshold=excluded.threshold,
        temperature=excluded.temperature,
        test_logloss=excluded.test_logloss,
        test_brier=excluded.test_brier,
        test_precision=excluded.test_precision,
        test_ev=excluded.test_ev
    """, (
        strategy,
        VERSION,
        len(rows),
        int(time.time() * 1000),
        0,
        threshold,
        temperature,
        test_metrics["logloss"],
        test_metrics["brier"],
        precision,
        test_ev
    ))

    db_execute("""
    UPDATE predictions_v401
    SET trained=1
    WHERE strategy=?
      AND resolved=1
      AND train_candidate=1
    """, (strategy,))

    log.info(
        "%s model trained | samples=%d | threshold=%.3f | temp=%.3f | "
        "test_brier=%.5f | precision=%.3f | EV=%.3f",
        strategy,
        len(rows),
        threshold,
        temperature,
        test_metrics["brier"],
        precision,
        test_ev
    )

    return True


# ============================================================
# EXPOSURE / POSITION SIZING
# ============================================================

def open_signal_count(symbol=None):
    if symbol:
        rows = db_query("""
            SELECT COUNT(*)
            FROM predictions_v401
            WHERE signaled=1
              AND resolved=0
              AND symbol=?
        """, (symbol,))
    else:
        rows = db_query("""
            SELECT COUNT(*)
            FROM predictions_v401
            WHERE signaled=1
              AND resolved=0
        """)

    return int(rows[0][0])


def current_exposure():
    rows = db_query("""
        SELECT
            symbol,
            live_entry_price,
            live_stop_price
        FROM predictions_v401
        WHERE signaled=1
          AND resolved=0
    """)

    total = 0.0
    major = 0.0
    alt = 0.0

    for symbol, entry, stop in rows:
        entry = float(entry)
        stop = float(stop)

        unit_risk = max(
            entry - stop,
            1e-12
        )

        quantity = (
            ACCOUNT_EQUITY *
            RISK_PER_TRADE /
            unit_risk
        )

        notional = quantity * entry

        notional = min(
            notional,
            ACCOUNT_EQUITY *
            MAX_SINGLE_POSITION
        )

        total += notional

        if symbol in MAJORS:
            major += notional
        else:
            alt += notional

    return (
        total / ACCOUNT_EQUITY,
        major / ACCOUNT_EQUITY,
        alt / ACCOUNT_EQUITY
    )


def get_symbol_filters(
    exchange_info,
    symbol
):
    filters = {
        item["filterType"]: item
        for item in exchange_info[symbol].get(
            "filters",
            []
        )
    }

    lot = filters.get(
        "LOT_SIZE",
        {}
    )

    minimum_notional = filters.get(
        "MIN_NOTIONAL",
        filters.get("NOTIONAL", {})
    )

    return (
        float(
            lot.get(
                "stepSize",
                1.0
            )
        ),

        float(
            lot.get(
                "minQty",
                0.0
            )
        ),

        float(
            minimum_notional.get(
                "minNotional",
                0.0
            )
        )
    )


def floor_step(
    value,
    step
):
    if step <= 0:
        return value

    return math.floor(
        value / step
    ) * step


def position_size(
    symbol,
    entry,
    stop,
    exchange_info,
    probability
):
    risk_amount = (
        ACCOUNT_EQUITY *
        RISK_PER_TRADE
    )

    unit_risk = max(
        entry - stop,
        1e-12
    )

    confidence_factor = min(
        1.25,
        max(
            0.65,
            (probability - 0.5) / 0.25
        )
    )

    quantity = (
        risk_amount /
        unit_risk
    ) * confidence_factor

    max_notional = (
        ACCOUNT_EQUITY *
        MAX_SINGLE_POSITION
    )

    quantity = min(
        quantity,
        max_notional / entry
    )

    step, minimum_qty, minimum_notional = (
        get_symbol_filters(
            exchange_info,
            symbol
        )
    )

    quantity = floor_step(
        quantity,
        step
    )

    if quantity < minimum_qty:
        return 0.0

    if quantity * entry < minimum_notional:
        return 0.0

    return quantity


# ============================================================
# PREDICTIONS
# ============================================================

def latest_source_time(
    symbol,
    strategy
):
    rows = db_query("""
        SELECT source_close_ms
        FROM predictions_v401
        WHERE symbol=?
          AND strategy=?
        ORDER BY source_close_ms DESC
        LIMIT 1
    """, (
        symbol,
        strategy
    ))

    return (
        int(rows[0][0])
        if rows else None
    )


def prediction_exists(symbol, strategy, source_close_ms):
    rows=db_query("SELECT id FROM predictions_v401 WHERE symbol=? AND strategy=? AND source_close_ms=? LIMIT 1",(symbol,strategy,int(source_close_ms)))
    return bool(rows)

def predict_signal(symbol, strategy, exchange_info, btc_context):
    sample=build_sample(symbol,strategy,btc_context=btc_context)
    if sample is None or prediction_exists(symbol,strategy,sample["source_close"]):
        return None
    cfg=STRATEGIES[strategy]; decision_entry=float(sample["entry"]); atr=float(sample["atr"])
    target=decision_entry+cfg["target_atr"]*atr; stop=decision_entry-cfg["stop_atr"]*atr
    model,temperature,threshold,checkpoint=load_checkpoint(strategy)
    now_ms=int(time.time()*1000)
    # Bootstrap-safe: insert the clean causal observation even when no model exists.
    predicted_prob=calibrated_prob=predicted_hours=None; should_signal=False; expected_value=0.0; quantity=0.0
    if model is not None:
        tensors=[torch.tensor(sample[k][None],dtype=torch.float32) for k in ("seqs","micro","static")] if False else None
        with torch.no_grad():
            logits,time_output=model(torch.tensor(sample["seqs"]["1h"][None],dtype=torch.float32),torch.tensor(sample["seqs"]["4h"][None],dtype=torch.float32),torch.tensor(sample["seqs"]["1d"][None],dtype=torch.float32),torch.tensor(sample["micro"][None],dtype=torch.float32),torch.tensor(sample["static"][None],dtype=torch.float32))
        predicted_prob=float(torch.sigmoid(logits).item())
        calibrated_prob=float(torch.sigmoid(logits/max(temperature,0.05)).item())
        predicted_hours=max(1.0,math.expm1(float(time_output.item())))
        rr=cfg["target_atr"]/cfg["stop_atr"]
        expected_value=calibrated_prob*rr-(1-calibrated_prob)-0.03
        should_signal=(calibrated_prob>=threshold and expected_value>=MIN_EV and open_signal_count(symbol)==0)
        total,major,alt=current_exposure()
        if total>=MAX_TOTAL_EXPOSURE or (symbol in MAJORS and major>=MAX_MAJOR_EXPOSURE) or (symbol not in MAJORS and alt>=MAX_ALT_EXPOSURE): should_signal=False
        if should_signal:
            # Actionable entry is LIVE price, fetched only after model/risk gates.
            live_entry=get_price(symbol)
            live_target=live_entry+cfg["target_atr"]*atr
            live_stop=live_entry-cfg["stop_atr"]*atr
            quantity=position_size(symbol,live_entry,live_stop,exchange_info,calibrated_prob)
            if quantity<=0: should_signal=False
        else:
            live_entry=live_target=live_stop=None
    else:
        live_entry=live_target=live_stop=None
        log.info("%s/%s: no checkpoint; recording bootstrap observation",symbol,strategy)
    stride_ms=(6 if strategy=="day" else 12)*3600*1000
    nearby=db_query("SELECT id FROM predictions_v401 WHERE symbol=? AND strategy=? AND train_candidate=1 AND source_close_ms BETWEEN ? AND ? LIMIT 1",(symbol,strategy,sample["source_close"]-stride_ms+1,sample["source_close"]+stride_ms-1))
    train_candidate=0 if nearby else 1
    db_execute("""INSERT INTO predictions_v401 (symbol,strategy,predicted_at,source_close_ms,decision_entry_price,target_price,stop_price,live_entry_price,live_target_price,live_stop_price,atr,predicted_prob,calibrated_prob,predicted_hours,threshold,signaled,resolved,train_candidate,model_version,sequence_json,static_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(symbol,strategy,now_ms,sample["source_close"],decision_entry,target,stop,live_entry,live_target,live_stop,atr,predicted_prob,calibrated_prob,predicted_hours,threshold if model is not None else None,int(should_signal),0,train_candidate,VERSION,serialize_sample(sample),json.dumps(sample["static"].tolist(),separators=(",",":")),now_ms))
    if not should_signal: return None
    return {"symbol":symbol,"strategy":strategy,"entry":live_entry,"target":live_target,"stop":live_stop,"probability":calibrated_prob,"hours":predicted_hours,"quantity":quantity,"expected_value":expected_value,"decision_entry":decision_entry,"source_close":sample["source_close"]}


# ============================================================
# FUTURE-CANDLE LABELING
# ============================================================

def unresolved_predictions():
    rows = db_query("""
        SELECT
            id,
            symbol,
            strategy,
            predicted_at,
            source_close_ms,
            decision_entry_price,
            target_price,
            stop_price
        FROM predictions_v401
        WHERE resolved=0
        ORDER BY symbol, id
    """)

    result = {}

    for row in rows:
        result.setdefault(
            row[1],
            []
        ).append(row)

    return result


def resolve_symbol(symbol,predictions):
    if not predictions: return
    earliest_source=min(int(r[4]) for r in predictions)
    try:
        future=get_klines(symbol,"1h",1000,start_ms=earliest_source+1)
    except Exception as exc:
        log.warning("Could not resolve %s: %s",symbol,exc); return
    if future.empty: return
    now_ms=int(time.time()*1000)
    for row in predictions:
        prediction_id,_,strategy,predicted_at,source_close,entry,target,stop=row
        source_close=int(source_close); horizon_ms=int(STRATEGIES[strategy]["timeout_h"]*3600*1000)
        deadline=min(now_ms,source_close+horizon_ms)
        bars=future[(future["open_ms"]>source_close)&(future["open_ms"]<=deadline)]
        outcome=outcome_type=resolved_at=hours=None
        for _,bar in bars.iterrows():
            high=float(bar["high"]); low=float(bar["low"]); hit_target=high>=float(target); hit_stop=low<=float(stop)
            if hit_target and hit_stop: outcome=0; outcome_type="both_same_candle_conservative_stop"
            elif hit_stop: outcome=0; outcome_type="stop"
            elif hit_target: outcome=1; outcome_type="target"
            else: continue
            resolved_at=int(bar["close_ms"]); hours=(resolved_at-source_close)/3600000; break
        if outcome is None and now_ms>=source_close+horizon_ms:
            outcome=0; outcome_type="timeout"; resolved_at=source_close+horizon_ms; hours=STRATEGIES[strategy]["timeout_h"]
        if outcome is not None:
            db_execute("UPDATE predictions_v401 SET resolved=1,outcome=?,outcome_type=?,resolved_at=?,hours_to_result=? WHERE id=?",(outcome,outcome_type,resolved_at,hours,prediction_id))


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):
    if not TELEGRAM_TOKEN:
        return False

    destinations = [
        YOUR_CHAT_ID
    ]

    if CHANNEL_ID:
        destinations.append(
            CHANNEL_ID
        )

    success = True

    for chat_id in destinations:
        if not chat_id:
            continue

        try:
            response = session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={
                    "chat_id": chat_id,
                    "text": message
                },
                timeout=HTTP_TIMEOUT
            )

            if not response.ok:
                success = False

                log.warning(
                    "Telegram failed: %s",
                    response.text[:300]
                )

        except Exception as exc:
            success = False

            log.warning(
                "Telegram exception: %s",
                exc
            )

    return success


def format_signal(signal):
    return (
        f"🧠 V4.0 {signal['strategy'].upper()} SIGNAL\n"
        f"Coin: {signal['symbol']}\n"
        f"Live entry: {signal['entry']:.8g}\n"
        f"Target: {signal['target']:.8g}\n"
        f"Stop: {signal['stop']:.8g}\n"
        f"Win probability: {signal['probability'] * 100:.1f}%\n"
        f"Predicted event time: {signal['hours']:.1f}h\n"
        f"Expected value: {signal['expected_value']:.3f}\n"
        f"Suggested quantity: {signal['quantity']:.8g}\n"
        f"Model: {VERSION}\n"
        f"Decision close: {signal['decision_entry']:.8g} | no exchange order is placed."
    )


# ============================================================
# TRAINING TRIGGER
# ============================================================

def new_training_examples(strategy):
    rows = db_query("""
        SELECT COUNT(*)
        FROM predictions_v401
        WHERE strategy=?
          AND resolved=1
          AND trained=0
          AND train_candidate=1
    """, (strategy,))

    return int(
        rows[0][0]
    )


def last_attempt_count(strategy):
    rows = db_query(
        "SELECT new_resolved_since_train FROM model_state_v401 WHERE strategy=?",
        (strategy,)
    )
    return int(rows[0][0]) if rows and rows[0][0] is not None else 0


def mark_attempt(strategy, count):
    db_execute("""
    INSERT INTO model_state_v401 (strategy, new_resolved_since_train)
    VALUES (?, ?)
    ON CONFLICT(strategy) DO UPDATE SET
        new_resolved_since_train=excluded.new_resolved_since_train
    """, (strategy, count))


def maybe_train():
    trained_any = False

    for strategy in STRATEGIES:
        count = new_training_examples(
            strategy
        )

        attempted_at = last_attempt_count(strategy)

        # Only attempt once another full RETRAIN_EVERY worth of new
        # examples has accumulated since the LAST ATTEMPT (successful or
        # not) — not just since the last success. Without this, a
        # persistent failure condition (e.g. class imbalance that isn't
        # resolving) causes a full training pipeline to be re-attempted
        # every single hourly run indefinitely, burning CI time on a
        # doomed-to-fail retry each time.
        if count - attempted_at >= RETRAIN_EVERY:
            log.info(
                "%s: attempting retrain (%d new examples since last attempt)",
                strategy,
                count - attempted_at
            )

            mark_attempt(strategy, count)

            if train_strategy(strategy):
                trained_any = True

    return trained_any


# ============================================================
# MAIN
# ============================================================

def main():
    init_db()

    log.info(
        "Starting V4.0 | device=%s",
        DEVICE
    )

    symbols, exchange_info = (
        get_top_symbols()
    )

    btc_context = btc_regime()

    log.info(
        "Universe: %d symbols",
        len(symbols)
    )

    # Collect one historical microstructure observation per symbol
    # per hourly bucket.
    for symbol in symbols:
        try:
            collect_snapshot(
                symbol,
                btc_context
            )
        except Exception as exc:
            log.warning(
                "Snapshot loop failed for %s: %s",
                symbol,
                exc
            )

    # Resolve mature observations using future candle OHLC.
    grouped = unresolved_predictions()

    for symbol, predictions in grouped.items():
        resolve_symbol(
            symbol,
            predictions
        )

    # Retrain only after enough NEW resolved observations exist.
    maybe_train()

    signals = []

    for symbol in symbols:
        for strategy in STRATEGIES:
            try:
                signal = predict_signal(
                    symbol,
                    strategy,
                    exchange_info,
                    btc_context
                )

                if signal:
                    signals.append(
                        signal
                    )

                    telegram_send(
                        format_signal(
                            signal
                        )
                    )

            except Exception as exc:
                log.exception(
                    "Prediction failed %s/%s: %s",
                    symbol,
                    strategy,
                    exc
                )

    log.info(
        "V4 run completed | signals=%d",
        len(signals)
    )


if __name__ == "__main__":
    main()