import os
import time
import json
import math
import random
import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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

KRAKEN_SPOT_BASE = "https://api.kraken.com"
KRAKEN_FUTURES_BASE = "https://futures.kraken.com/derivatives/api/v3"

KRAKEN_INTERVAL_MINUTES = {"1h": 60, "4h": 240, "1d": 1440}

STATIC_WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SHIBUSDT",
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "TRXUSDT", "BNBUSDT"
]

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
    "spread", "book_age", "funding", "oi_change", "price_ret", "valid"
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

# Snapshots are meant to land on a fixed 1-hour grid (see collect_snapshot's
# bucket_ms). Anything pulled in dynamically via SCAN_TOP_N — as opposed to
# the STATIC_WATCHLIST, which is always included — can drop out of the top
# ranking for a stretch and miss snapshots entirely. Allow one occasional
# missed run (a transient API failure) without penalty, but reject a
# sequence whose rows aren't roughly evenly spaced, rather than silently
# feed the model a multi-day price move mislabeled as a normal ~1h step.
MAX_SNAPSHOT_GAP_MS = 2 * 3_600_000

# I/O-bound network waits, not CPU work — Python releases the GIL during
# a blocking request, so this doesn't need to match the runner's CPU
# core count (GitHub's free ubuntu-latest runners have 2). Kept modest
# specifically to stay well inside Kraken's own rate limits rather than
# to respect any CPU constraint.
KRAKEN_THREAD_WORKERS = int(os.getenv("KRAKEN_THREAD_WORKERS", "8"))


def parallel_fetch(items, fetch_fn, max_workers=KRAKEN_THREAD_WORKERS):
    """Runs fetch_fn(item) concurrently across items. ONLY ever used for
    pure Kraken network calls — never anything touching the Turso
    connection, since its thread-safety under concurrent use isn't
    confirmed and a race there would be a far worse outcome than a slow
    run. Returns {item: result_or_None}; a failed fetch logs a warning
    and maps to None rather than aborting the whole batch."""
    results = {}
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_fn, item): item for item in items}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                results[item] = future.result()
            except Exception as exc:
                log.warning("Parallel fetch failed for %s: %s", item, exc)
                results[item] = None
    return results

# ------------------------------------------------------------
# Candidate lifecycle: Core / Scout / Incubator / Promoted / Dormant
# ------------------------------------------------------------
# A coin sitting right at a ranking boundary shouldn't flap in and out
# of tracking every run — that's what produced the snapshot gaps fixed
# earlier. Entry and exit use DIFFERENT thresholds (hysteresis): easier
# to fall below the exit bar than to clear the entry bar in the first
# place, so genuine boundary noise doesn't cause repeated cold starts.
PROMOTE_SCORE = float(os.getenv("PROMOTE_SCORE", "0.55"))
DEMOTE_SCORE = float(os.getenv("DEMOTE_SCORE", "0.35"))
CONSECUTIVE_TO_INCUBATE = int(os.getenv("CONSECUTIVE_TO_INCUBATE", "3"))
CONSECUTIVE_TO_DORMANT = int(os.getenv("CONSECUTIVE_TO_DORMANT", "3"))

# How many brand-new candidates can be discovered per run, and the hard
# cap on total tracked universe size (beyond Core) — both bound the
# extra per-run API cost of scoring (one daily-OHLC call per tracked
# candidate), since that's now the main new cost this mechanism adds.
NEW_DISCOVERIES_PER_RUN = int(os.getenv("NEW_DISCOVERIES_PER_RUN", "20"))
MAX_TRACKED_CANDIDATES = int(os.getenv("MAX_TRACKED_CANDIDATES", "120"))

# Scoring weights. "Persistence" was deliberately dropped as a scored
# component — it's now the CONSECUTIVE_TO_* state-transition logic
# above instead, which is a more direct and separately-tunable way to
# get the same anti-flapping effect than diluting it into a weighted
# average. The 5% freed up went to liquidity, since thin books directly
# undermine ATR-based target/stop reliability and position sizing —
# the most "hard constraint"-like of the components.
CANDIDATE_WEIGHTS = {
    "liquidity": 0.35,
    "relative_volume": 0.20,
    "trend_momentum": 0.20,
    "volatility_quality": 0.15,
    "market_relative_strength": 0.10,
}

# Retention — the main sources of unbounded growth in this system.
SNAPSHOT_RETENTION_DAYS = int(os.getenv("SNAPSHOT_RETENTION_DAYS", "30"))
PREDICTION_RETENTION_MONTHS = int(os.getenv("PREDICTION_RETENTION_MONTHS", "18"))
DORMANT_RETENTION_DAYS = int(os.getenv("DORMANT_RETENTION_DAYS", "14"))

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
        spread_bps REAL, book_age_sec REAL, funding REAL,
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

    db_execute("""
    CREATE TABLE IF NOT EXISTS candidate_state_v401 (
        symbol TEXT PRIMARY KEY,
        state TEXT NOT NULL DEFAULT 'scout',
        score REAL,
        consecutive_qualify INTEGER DEFAULT 0,
        consecutive_disqualify INTEGER DEFAULT 0,
        first_seen_ms INTEGER,
        state_changed_ms INTEGER,
        last_scored_ms INTEGER
    )
    """)

    # Append-only — candidate_state_v401 only ever stores each symbol's
    # CURRENT score (upsert), so without this there is no way to later
    # ask "did coins that scored well at promotion time actually turn
    # out well." This is what makes PROMOTE_SCORE/DEMOTE_SCORE
    # calibration possible at all.
    db_execute("""
    CREATE TABLE IF NOT EXISTS candidate_score_history_v401 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        score REAL,
        state_before TEXT,
        state_after TEXT,
        logged_ms INTEGER
    )
    """)

    db_execute("""
    CREATE TABLE IF NOT EXISTS system_state_v401 (
        key TEXT PRIMARY KEY,
        value INTEGER
    )
    """)

    db_execute("""CREATE INDEX IF NOT EXISTS idx_v401_predictions_resolution
    ON predictions_v401(resolved, strategy)""")
    db_execute("""CREATE INDEX IF NOT EXISTS idx_v401_predictions_source
    ON predictions_v401(symbol, strategy, source_close_ms)""")
    db_execute("""CREATE INDEX IF NOT EXISTS idx_v401_snapshots
    ON market_snapshots_v401(symbol, snapshot_ms)""")


# ============================================================
# KRAKEN HTTP
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

    raise RuntimeError(f"Kraken API failed: {path}: {last_error}")


def get_exchange_info():
    return api_get(KRAKEN_SPOT_BASE, "/0/public/AssetPairs")


_kraken_pairs_cache = None


def valid_kraken_symbols():
    """Every symbol elsewhere in this file (DB rows, Telegram messages,
    the STATIC_WATCHLIST) stays in familiar 'BTCUSDT'-style canonical
    form. This function is the ONLY place that talks to Kraken's own
    (fairly messy, legacy-encumbered) pair-naming — everything else just
    calls kraken_pair_name(symbol) to translate at the HTTP boundary.
    We derive the canonical name from Kraken's own 'wsname' field
    (documented, human-readable 'BASE/QUOTE' form) rather than trying to
    manually strip Kraken's legacy X/Z asset-code prefixes ourselves,
    since wsname exists specifically to sidestep that mess."""
    data = get_exchange_info()
    pairs = data.get("result", {})

    result = {}

    for info in pairs.values():
        wsname = info.get("wsname", "")

        if "/" not in wsname:
            continue

        base, quote = wsname.split("/")

        if quote not in ("USD", "USDT"):
            continue

        base = "BTC" if base == "XBT" else base
        canonical = f"{base}USDT"

        # Prefer a native USDT-quoted pair over a USD one if both exist
        # for the same coin, since the rest of this file assumes "USDT".
        if canonical in result and quote != "USDT":
            continue

        result[canonical] = {
            "kraken_pair": info.get("altname", ""),
            "lot_decimals": info.get("lot_decimals", 8),
            "ordermin": float(info.get("ordermin", 0.0) or 0.0),
            "costmin": float(info.get("costmin", 0.0) or 0.0),
        }

    return result


def kraken_pairs():
    global _kraken_pairs_cache

    if _kraken_pairs_cache is None:
        _kraken_pairs_cache = valid_kraken_symbols()

    return _kraken_pairs_cache


def kraken_pair_name(symbol):
    info = kraken_pairs().get(symbol)

    if not info:
        raise RuntimeError(f"Unknown or unsupported symbol on Kraken: {symbol}")

    return info["kraken_pair"]


def score_liquidity(usd_volume_24h):
    if usd_volume_24h <= 0:
        return 0.0
    # log-scaled, saturating around $50M/24h as "excellent" liquidity
    return float(min(1.0, math.log10(usd_volume_24h + 1) / math.log10(50_000_000)))


def score_relative_volume(daily):
    if daily is None or len(daily) < 8:
        return 0.0
    volumes = daily["volume"].astype(float).values
    today = volumes[-1]
    baseline = float(np.mean(volumes[-8:-1]))
    if baseline <= 0:
        return 0.0
    ratio = today / baseline
    return float(1.0 / (1.0 + math.exp(-2.0 * (ratio - 1.0))))


def score_trend_momentum(daily):
    # Deliberately coarse and multi-day — a distinct timescale from the
    # model's own hourly/4h momentum features (RSI/MACD/ADX). This is
    # "is this coin alive enough to deserve tracking," not "is this a
    # good entry right now" — using the same signal for both would bias
    # what the model ever gets to learn from toward coins that already
    # look like they're trending.
    if daily is None or len(daily) < 8:
        return 0.5
    close = daily["close"].astype(float).values
    ret7 = math.log(close[-1] / close[-8]) if close[-8] > 0 else 0.0
    return float(1.0 / (1.0 + math.exp(-8.0 * ret7)))


def score_volatility_quality(daily):
    if daily is None or len(daily) < 8:
        return 0.0
    high = daily["high"].astype(float).values[-8:]
    low = daily["low"].astype(float).values[-8:]
    close = daily["close"].astype(float).values[-8:]
    ranges = (high - low) / np.where(close > 0, close, 1.0)
    avg_range = float(np.mean(ranges))
    consistency = 1.0 - min(1.0, float(np.std(ranges) / (avg_range + 1e-9)))
    # Want SOME real movement (not a dead coin), but consistent day to
    # day rather than one wild outlier candle driving the whole window
    # — the latter is the signature of a thin-liquidity flash move, not
    # genuine price discovery.
    magnitude = 1.0 - math.exp(-15.0 * avg_range)
    return float(max(0.0, min(1.0, 0.5 * magnitude + 0.5 * consistency)))


def score_market_relative_strength(daily, btc_context):
    if daily is None or len(daily) < 8:
        return 0.5
    close = daily["close"].astype(float).values
    coin_ret = math.log(close[-1] / close[-8]) if close[-8] > 0 else 0.0
    btc_ret24, _, _ = btc_context
    relative = coin_ret - btc_ret24
    return float(1.0 / (1.0 + math.exp(-8.0 * relative)))


def compute_candidate_score(usd_volume_24h, daily, btc_context):
    parts = {
        "liquidity": score_liquidity(usd_volume_24h),
        "relative_volume": score_relative_volume(daily),
        "trend_momentum": score_trend_momentum(daily),
        "volatility_quality": score_volatility_quality(daily),
        "market_relative_strength": score_market_relative_strength(daily, btc_context),
    }
    score = sum(CANDIDATE_WEIGHTS[k] * v for k, v in parts.items())
    return float(score), parts


def has_enough_micro_history(symbol, as_of_ms):
    micro, _ = build_micro_sequence(symbol, MIN_MICRO_HISTORY, as_of_ms)
    return micro is not None


def get_candidate_state(symbol):
    rows = db_query("""
        SELECT state, score, consecutive_qualify, consecutive_disqualify,
               first_seen_ms, state_changed_ms
        FROM candidate_state_v401 WHERE symbol=?
    """, (symbol,))
    return rows[0] if rows else None


def upsert_candidate_state(symbol, state, score, cq, cd, first_seen_ms, state_changed_ms, now_ms):
    db_execute("""
    INSERT INTO candidate_state_v401
        (symbol, state, score, consecutive_qualify, consecutive_disqualify,
         first_seen_ms, state_changed_ms, last_scored_ms)
    VALUES (?,?,?,?,?,?,?,?)
    ON CONFLICT(symbol) DO UPDATE SET
        state=excluded.state, score=excluded.score,
        consecutive_qualify=excluded.consecutive_qualify,
        consecutive_disqualify=excluded.consecutive_disqualify,
        state_changed_ms=excluded.state_changed_ms,
        last_scored_ms=excluded.last_scored_ms
    """, (symbol, state, score, cq, cd, first_seen_ms, state_changed_ms, now_ms))


def advance_candidate(symbol, score, now_ms):
    """One coin, one run: update its score, consecutive counters, and
    state. Promotion out of Incubator requires BOTH a healthy score AND
    enough contiguous snapshot history — the data-availability half of
    'don't predict on a new coin until enough data has been collected',
    checked by directly attempting a micro-sequence build rather than
    re-deriving the gap logic here."""
    existing = get_candidate_state(symbol)

    if existing is None:
        state, cq, cd, first_seen, changed = "scout", 0, 0, now_ms, now_ms
    else:
        state, _prev_score, cq, cd, first_seen, changed = existing

    if state == "core":
        upsert_candidate_state(symbol, "core", score, cq, cd, first_seen, changed, now_ms)
        return "core"

    qualifies = score >= PROMOTE_SCORE
    disqualifies = score < DEMOTE_SCORE

    cq = cq + 1 if qualifies else 0
    cd = cd + 1 if disqualifies else 0

    new_state = state

    if state in ("scout", "dormant") and cq >= CONSECUTIVE_TO_INCUBATE:
        new_state = "incubator"
    elif state == "incubator":
        if cd >= CONSECUTIVE_TO_DORMANT:
            new_state = "dormant"
        elif has_enough_micro_history(symbol, now_ms):
            new_state = "promoted"
    elif state == "promoted":
        if cd >= CONSECUTIVE_TO_DORMANT:
            new_state = "dormant"

    if new_state != state:
        changed = now_ms
        log.info("%s: %s -> %s (score=%.3f)", symbol, state, new_state, score)

    upsert_candidate_state(symbol, new_state, score, cq, cd, first_seen, changed, now_ms)

    db_execute("""
        INSERT INTO candidate_score_history_v401 (symbol, score, state_before, state_after, logged_ms)
        VALUES (?,?,?,?,?)
    """, (symbol, score, state, new_state, now_ms))

    return new_state


def discover_and_score():
    """Replaces the old flat top-N cutoff. Returns (snapshot_targets,
    predict_targets, exchange_info, state_counts) where snapshot_targets
    = everyone who should get a market snapshot this run (core, incubator,
    promoted — NOT scout, which is deliberately cheap and depth-call-free,
    and NOT dormant, which has been actively de-prioritized), and
    predict_targets = everyone eligible for actual signals (core, promoted)."""
    valid = kraken_pairs()

    try:
        tickers = api_get(KRAKEN_SPOT_BASE, "/0/public/Ticker")
        ticker_result = tickers.get("result", {})
    except Exception as exc:
        log.warning("Could not fetch Kraken tickers: %s", exc)
        ticker_result = {}

    volumes = {}
    for canonical, info in valid.items():
        ticker = ticker_result.get(info["kraken_pair"])
        if not ticker:
            continue
        try:
            last_price = float(ticker["c"][0])
            base_volume_24h = float(ticker["v"][1])
            volumes[canonical] = last_price * base_volume_24h
        except Exception:
            continue

    tracked = {row[0] for row in db_query("SELECT symbol FROM candidate_state_v401")}

    # Discover a bounded number of new candidates per run, ranked by raw
    # 24h volume as a cheap first filter (the full score needs a daily
    # OHLC pull, which is the real per-symbol cost — no point spending it
    # on obviously-illiquid pairs).
    if len(tracked) < MAX_TRACKED_CANDIDATES:
        room = MAX_TRACKED_CANDIDATES - len(tracked)
        discoverable = sorted(
            (s for s in volumes if s not in tracked and s not in STATIC_WATCHLIST
             and volumes[s] >= MIN_24H_VOLUME_USD),
            key=lambda s: volumes[s], reverse=True
        )
        for symbol in discoverable[:min(NEW_DISCOVERIES_PER_RUN, room)]:
            tracked.add(symbol)

    btc_context = btc_regime()
    now_ms = int(time.time() * 1000)

    all_symbols = set(STATIC_WATCHLIST) | tracked
    state_counts = {"core": 0, "scout": 0, "incubator": 0, "promoted": 0, "dormant": 0}

    # Pure network fetch, no DB involved — safe to parallelize. This is
    # the main per-run cost of discovery (one call per tracked candidate),
    # now done concurrently instead of one at a time.
    daily_cache = parallel_fetch(
        [s for s in all_symbols if s in valid],
        lambda s: get_klines(s, "1d", 10)
    )

    for symbol in all_symbols:
        if symbol not in valid:
            continue

        existing = get_candidate_state(symbol)
        is_core = symbol in STATIC_WATCHLIST

        if is_core and existing is None:
            upsert_candidate_state(symbol, "core", 1.0, 0, 0, now_ms, now_ms, now_ms)

        daily = daily_cache.get(symbol)

        score, _parts = compute_candidate_score(volumes.get(symbol, 0.0), daily, btc_context)
        state = advance_candidate(symbol, score, now_ms)
        state_counts[state] = state_counts.get(state, 0) + 1

    snapshot_targets = [
        row[0] for row in db_query(
            "SELECT symbol FROM candidate_state_v401 WHERE state IN ('core','incubator','promoted')"
        )
    ]
    predict_targets = [
        row[0] for row in db_query(
            "SELECT symbol FROM candidate_state_v401 WHERE state IN ('core','promoted')"
        )
    ]

    return snapshot_targets, predict_targets, valid, state_counts, btc_context


def cleanup_old_data():
    """The main sources of unbounded growth: hourly snapshots, resolved
    predictions, and dormant candidates that never returned.

    Snapshots are unaffected by training concerns — every consumer of
    market_snapshots_v401 (build_micro_sequence, previous_snapshot,
    micro_static_as_of) only ever looks back MIN_MICRO_HISTORY hours.
    Training never touches this table at all; it reads its own frozen
    sequence_json/static_json copy stored in predictions_v401 at
    prediction time, so pruning old snapshots can't affect retraining.

    A resolved prediction is only ever pruned once it satisfies ALL of:
      - trained=1 (already used in at least one training run) — a row
        that hasn't been learned from yet is NEVER deleted regardless
        of age, since new_training_examples() counts exactly these
        trained=0 rows to decide when to retrain; deleting one before
        it's had its turn would silently discard real training signal
      - outside the most recent MAX_TRAIN_SAMPLES for its strategy —
        the actual working set get_training_rows() draws from, so
        nothing currently in active use is ever at risk even if total
        volume stays below that cap for longer than the retention window
      - older than the age-based retention window
    """
    now_ms = int(time.time() * 1000)

    db_execute(
        "DELETE FROM market_snapshots_v401 WHERE snapshot_ms < ?",
        (now_ms - SNAPSHOT_RETENTION_DAYS * 24 * 3600 * 1000,)
    )

    db_execute(
        "DELETE FROM candidate_score_history_v401 WHERE logged_ms < ?",
        (now_ms - PREDICTION_RETENTION_MONTHS * 30 * 24 * 3600 * 1000,)
    )

    age_cutoff = now_ms - PREDICTION_RETENTION_MONTHS * 30 * 24 * 3600 * 1000
    pruned_predictions = 0

    for strategy in STRATEGIES:
        # train_candidate=1 rows are the ones that can actually become
        # training examples — protect them until trained=1 (see
        # train_strategy's final UPDATE) AND until they're outside the
        # recent MAX_TRAIN_SAMPLES window get_training_rows() draws from.
        keep_rows = db_query("""
            SELECT id FROM predictions_v401
            WHERE strategy=? AND resolved=1 AND train_candidate=1
            ORDER BY source_close_ms DESC LIMIT ?
        """, (strategy, MAX_TRAIN_SAMPLES))
        keep_ids = {row[0] for row in keep_rows}

        trainable_old = db_query("""
            SELECT id FROM predictions_v401
            WHERE strategy=? AND resolved=1 AND train_candidate=1
              AND trained=1 AND resolved_at < ?
        """, (strategy, age_cutoff))

        # train_candidate=0 rows are deliberately excluded from training
        # by the temporal-stride dedup (predict_signal) and — critically —
        # NEVER get trained=1 set by anything, since train_strategy's
        # UPDATE only touches train_candidate=1 rows. Gating their
        # deletion on trained=1 would mean they never get pruned at all,
        # regardless of age. Their only remaining purpose is calibration
        # (which reads all resolved rows, not just train_candidate=1 —
        # see calibration_report.py), so plain age governs them.
        non_trainable_old = db_query("""
            SELECT id FROM predictions_v401
            WHERE strategy=? AND resolved=1 AND train_candidate=0
              AND resolved_at < ?
        """, (strategy, age_cutoff))

        to_delete = [row[0] for row in trainable_old if row[0] not in keep_ids]
        to_delete += [row[0] for row in non_trainable_old]

        for i in range(0, len(to_delete), 500):
            chunk = to_delete[i:i + 500]
            placeholders = ",".join("?" for _ in chunk)
            db_execute(f"DELETE FROM predictions_v401 WHERE id IN ({placeholders})", tuple(chunk))

        pruned_predictions += len(to_delete)

    # Separate issue found while auditing this: resolve_symbol() silently
    # gives up (logs a warning, returns) if get_klines() ever fails for a
    # symbol — most commonly because it's been delisted mid-holding-period
    # — leaving that prediction unresolved forever with no age-based
    # pruning path able to touch it (the block above only ever looks at
    # resolved=1 rows). A row sitting unresolved for many multiples of
    # its own strategy's timeout window is almost certainly stuck, not
    # legitimately still pending — mark it abandoned (no fabricated
    # outcome, excluded from training) instead of deleting it outright,
    # consistent with this file's existing preference for marking over
    # silent removal (see the trained flag itself).
    abandoned_total = 0

    for strategy, cfg in STRATEGIES.items():
        abandon_cutoff = now_ms - 5 * cfg["timeout_h"] * 3600 * 1000

        stuck = db_query("""
            SELECT id FROM predictions_v401
            WHERE strategy=? AND resolved=0 AND source_close_ms < ?
        """, (strategy, abandon_cutoff))
        stuck_ids = [row[0] for row in stuck]

        for i in range(0, len(stuck_ids), 500):
            chunk = stuck_ids[i:i + 500]
            placeholders = ",".join("?" for _ in chunk)
            db_execute(
                f"""UPDATE predictions_v401 SET resolved=1, outcome_type='abandoned',
                    resolved_at=?, train_candidate=0 WHERE id IN ({placeholders})""",
                (now_ms, *chunk)
            )

        abandoned_total += len(stuck_ids)

    dormant_cutoff = now_ms - DORMANT_RETENTION_DAYS * 24 * 3600 * 1000
    long_dormant = db_query(
        "SELECT symbol FROM candidate_state_v401 WHERE state='dormant' AND state_changed_ms < ?",
        (dormant_cutoff,)
    )

    for (symbol,) in long_dormant:
        # Fully forgotten — if it ever recovers, it re-enters as a new
        # Scout from scratch, same as any coin we've never seen before.
        db_execute("DELETE FROM market_snapshots_v401 WHERE symbol=?", (symbol,))
        db_execute("DELETE FROM candidate_state_v401 WHERE symbol=?", (symbol,))

    # Delisted symbols: no longer a valid Kraken pair at all.
    valid = kraken_pairs()
    tracked = db_query("SELECT symbol FROM candidate_state_v401")
    for (symbol,) in tracked:
        if symbol not in valid and symbol not in STATIC_WATCHLIST:
            db_execute("DELETE FROM market_snapshots_v401 WHERE symbol=?", (symbol,))
            db_execute("DELETE FROM candidate_state_v401 WHERE symbol=?", (symbol,))

    log.info(
        "Cleanup: pruned %d resolved predictions, abandoned %d stuck-unresolved, "
        "forgot %d long-dormant candidates",
        pruned_predictions,
        abandoned_total,
        len(long_dormant)
    )

    return len(long_dormant)


def get_klines(symbol, interval, limit=500, start_ms=None, end_ms=None):
    pair = kraken_pair_name(symbol)
    minutes = KRAKEN_INTERVAL_MINUTES.get(interval)

    if minutes is None:
        raise ValueError(f"Unsupported interval for Kraken: {interval}")

    params = {"pair": pair, "interval": minutes}

    if start_ms is not None:
        params["since"] = int(start_ms // 1000)

    raw = api_get(KRAKEN_SPOT_BASE, "/0/public/OHLC", params)
    result = raw.get("result", {})

    # Kraken echoes back its OWN internal pair key in the response, which
    # doesn't always match the altname we queried with — so we take
    # whichever series is present rather than re-indexing by name.
    series = None
    for key, value in result.items():
        if key != "last":
            series = value
            break

    columns = [
        "open_ms", "open", "high", "low", "close", "volume",
        "close_ms", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"
    ]

    if not series:
        return pd.DataFrame(columns=columns)

    interval_ms = minutes * 60 * 1000
    rows = []

    for candle in series:
        # Kraken candle: [time, open, high, low, close, vwap, volume, count]
        open_ms = int(float(candle[0]) * 1000)

        rows.append([
            open_ms,
            candle[1], candle[2], candle[3], candle[4],
            candle[6],
            open_ms + interval_ms - 1,   # Kraken gives only open time
            float(candle[6]) * float(candle[5]),  # approx quote volume (vol * vwap)
            candle[7],                    # trade count
            0.0, 0.0,                     # taker base/quote — not exposed by Kraken
            None
        ])

    df = pd.DataFrame(rows, columns=columns)

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

    if end_ms is not None:
        df = df[df["open_ms"] <= end_ms].copy()

    # Kraken's OHLC endpoint doesn't take a limit param — it just returns
    # everything since `since` (up to its own internal cap), so we
    # truncate client-side to preserve the old call-site contract.
    if limit:
        df = df.tail(int(limit)).copy()

    return df.reset_index(drop=True)


def get_price(symbol):
    pair = kraken_pair_name(symbol)
    data = api_get(KRAKEN_SPOT_BASE, "/0/public/Ticker", {"pair": pair})
    result = data.get("result", {})

    for value in result.values():
        return float(value["c"][0])

    raise RuntimeError(f"No ticker data returned for {symbol}")


def get_depth(symbol, limit=100):
    pair = kraken_pair_name(symbol)
    data = api_get(KRAKEN_SPOT_BASE, "/0/public/Depth", {"pair": pair, "count": limit})
    result = data.get("result", {})

    for value in result.values():
        # Kraken returns [price, volume, timestamp] per level. We keep
        # all three now — the timestamp becomes a genuine feature
        # (book_age_sec, see orderbook_features) instead of being
        # discarded.
        return {"bids": value.get("bids", []), "asks": value.get("asks", [])}

    return {"bids": [], "asks": []}


_kraken_futures_tickers_cache = None


def _kraken_futures_tickers():
    """Fetched once per run and reused across every symbol — the same
    fix applied earlier to btc_regime() being recomputed per symbol."""
    global _kraken_futures_tickers_cache

    if _kraken_futures_tickers_cache is None:
        try:
            data = api_get(KRAKEN_FUTURES_BASE, "/tickers")
            _kraken_futures_tickers_cache = data.get("tickers", [])
        except Exception as exc:
            log.warning("Could not fetch Kraken Futures tickers: %s", exc)
            _kraken_futures_tickers_cache = []

    return _kraken_futures_tickers_cache


def _find_futures_ticker(symbol):
    base = symbol[:-4]
    base = "XBT" if base == "BTC" else base

    # Kraken Futures perpetual symbols use a prefix (commonly PF_ or
    # PI_ depending on contract generation) that we can't fully confirm
    # without a live test — matching on the base asset suffix rather
    # than assuming one exact prefix is deliberately more forgiving of
    # that uncertainty.
    for ticker in _kraken_futures_tickers():
        tsym = ticker.get("symbol", "")

        if tsym.startswith(("PF_", "PI_")) and tsym.endswith(f"{base}USD"):
            return ticker

    return None


def get_funding(symbol):
    try:
        ticker = _find_futures_ticker(symbol)

        if not ticker:
            return 0.0

        return float(ticker.get("fundingRate", 0) or 0)

    except Exception:
        return 0.0


def get_open_interest(symbol):
    try:
        ticker = _find_futures_ticker(symbol)

        if not ticker:
            return 0.0

        return float(ticker.get("openInterest", 0) or 0)

    except Exception:
        return 0.0


def orderbook_features(depth):
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])

    values = {}

    for level in (5, 10, 20, 50, 100):
        bid_notional = sum(
            float(price) * float(quantity)
            for price, quantity, *_rest in bids[:level]
        )

        ask_notional = sum(
            float(price) * float(quantity)
            for price, quantity, *_rest in asks[:level]
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

        # Kraken includes a per-level "last updated" timestamp we were
        # previously discarding entirely. The age of the top-of-book
        # quote is genuine signal: a small value means the market is
        # actively quoting right now (liquid, attentive), a large one
        # means the book has gone stale (thin, quiet). We only have this
        # timestamp on Kraken's depth response — Binance's didn't expose
        # it, which is presumably why it was dropped in the first place.
        if len(bids[0]) >= 3 and len(asks[0]) >= 3:
            now_sec = time.time()
            best_bid_ts = float(bids[0][2])
            best_ask_ts = float(asks[0][2])
            book_age_sec = max(0.0, now_sec - ((best_bid_ts + best_ask_ts) / 2))
        else:
            book_age_sec = 0.0
    else:
        spread_bps = 0.0
        book_age_sec = 0.0

    return values, spread_bps, book_age_sec


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


def collect_snapshot(symbol, btc_context, price=None, depth=None):
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
        if price is None:
            price = get_price(symbol)

        if depth is None:
            depth = get_depth(symbol, 100)
        obi, spread_bps, book_age_sec = orderbook_features(depth)

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
            spread_bps, book_age_sec, funding,
            open_interest, oi_change,
            btc_ret24, btc_vol24, btc_ema_spread
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            book_age_sec,
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
               spread_bps, book_age_sec, funding, oi_change, price
        FROM market_snapshots_v401
        WHERE symbol=? AND snapshot_ms <= ?
        ORDER BY snapshot_ms DESC LIMIT ?
    """, (symbol, int(as_of_ms), length))
    rows = list(reversed(rows))
    if len(rows) < MIN_MICRO_HISTORY:
        return None, None

    gaps = [rows[i][0] - rows[i - 1][0] for i in range(1, len(rows))]
    max_gap = max(gaps) if gaps else 0
    if max_gap > MAX_SNAPSHOT_GAP_MS:
        log.warning(
            "%s: micro sequence rejected — %.1fh gap between snapshots "
            "(likely dropped out of the scanned watchlist for a stretch)",
            symbol,
            max_gap / 3_600_000,
        )
        return None, None

    values = []
    for index, row in enumerate(rows):
        snapshot_ms, obi5, obi10, obi20, obi50, obi100, spread_bps, book_age_sec, funding, oi_change, price = row
        if index == 0:
            price_return = 0.0
        else:
            previous_price = float(rows[index - 1][10]); current_price = float(price)
            price_return = math.log(current_price / previous_price) if previous_price > 0 and current_price > 0 else 0.0
        # book_age_sec is raw seconds since the top-of-book quote last
        # moved — scale into units of 5 minutes and cap so one very stale
        # reading (e.g. a thinly-traded symbol overnight) doesn't dominate.
        book_age_scaled = min((float(book_age_sec or 0) / 300.0), 5.0)
        values.append([float(obi5 or 0), float(obi10 or 0), float(obi20 or 0), float(obi50 or 0), float(obi100 or 0), float(spread_bps or 0) / 100.0, book_age_scaled, float(funding or 0) * 1000.0, float(oi_change or 0) * 10.0, price_return, 1.0])
    array = np.asarray(values, dtype=np.float32)
    array[:, :10] = np.clip(array[:, :10], -10, 10)
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

_klines_cache = {}


def _max_seq_length(interval):
    return max((cfg["seq"].get(interval, 0) for cfg in STRATEGIES.values()), default=0)


def get_klines_cached(symbol, interval):
    """Raw candle data for a given (symbol, interval) doesn't depend on
    which strategy is asking, or on what end_ms cutoff gets applied
    afterward — Kraken always just returns 'most recent N candles', and
    end_ms is a client-side post-filter (get_klines/make_price_sequence
    both already applied it this way; Kraken's OHLC endpoint has no
    'until' parameter). day and swing both use 1h/4h/1d, with 4h and 1d
    even wanting the exact same length — so without this cache, every
    symbol's candles were being fetched from Kraken TWICE per run, once
    per strategy, for identical data. This is the single biggest
    fixable cost driver behind long run times."""
    key = (symbol, interval)
    if key not in _klines_cache:
        limit = min(_max_seq_length(interval) + 100, 1000)
        _klines_cache[key] = get_klines(symbol, interval, limit)
    return _klines_cache[key]


def prewarm_klines_cache(symbols):
    """Fetches every (symbol, interval) combination the prediction phase
    will need, concurrently, BEFORE the sequential predict_signal loop
    runs — so every get_klines_cached() call during that loop is a pure
    cache hit with no network wait. This is the parallel half of the
    same cache get_klines_cached reads from; nothing here touches Turso."""
    intervals = set()
    for cfg in STRATEGIES.values():
        intervals.update(cfg["seq"].keys())

    tasks = [(symbol, interval) for symbol in symbols for interval in intervals]

    def fetch_one(task):
        symbol, interval = task
        limit = min(_max_seq_length(interval) + 100, 1000)
        return get_klines(symbol, interval, limit)

    results = parallel_fetch(tasks, fetch_one)
    for task, df in results.items():
        if df is not None:
            _klines_cache[task] = df


def make_price_sequence(symbol, interval, length, end_ms=None):
    df = get_klines_cached(symbol, interval)
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
        primary_probe = get_klines_cached(symbol, cfg["primary"])
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

    telegram_send(
        format_model_retrained(strategy, threshold, temperature, metrics)
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
    info = exchange_info.get(symbol, {})

    lot_decimals = int(info.get("lot_decimals", 8))
    step = 10 ** (-lot_decimals)

    return (
        float(step),
        float(info.get("ordermin", 0.0) or 0.0),
        float(info.get("costmin", 0.0) or 0.0)
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
            stop_price,
            signaled,
            live_entry_price,
            live_target_price,
            live_stop_price
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


def resolve_symbol(symbol,predictions,future=None):
    if not predictions: return
    earliest_source=min(int(r[4]) for r in predictions)
    if future is None:
        try:
            future=get_klines(symbol,"1h",1000,start_ms=earliest_source+1)
        except Exception as exc:
            log.warning("Could not resolve %s: %s",symbol,exc); return
    if future is None or future.empty: return
    now_ms=int(time.time()*1000)
    resolved_count=0
    for row in predictions:
        prediction_id,_,strategy,predicted_at,source_close,entry,target,stop,signaled,live_entry,live_target,live_stop=row
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
            resolved_count+=1
            # Only alert on trades the user was actually told about —
            # shadow-logged (non-signaled) observations resolve silently,
            # same principle as never alerting on them in the first place.
            if signaled:
                telegram_send(format_trade_resolved(symbol,strategy,outcome_type,live_entry,live_target,live_stop,hours))
    return resolved_count


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


def format_run_started():
    return (
        f"🟢 V4.0.1 run started\n"
        f"Device: {DEVICE} | Time: {datetime.now(timezone.utc).isoformat()}"
    )


def format_run_completed(stats):
    return (
        f"🟢 V4.0.1 run completed\n"
        f"Duration: {stats['duration_sec']:.0f}s\n"
        f"Universe: {stats['universe']} symbols\n"
        f"Signals sent: {stats['signals']}\n"
        f"Trades resolved: {stats['resolved']}\n"
        f"Errors: {stats['errors']}"
    )


def format_data_collection(ok_count, fail_count, failed_symbols, universe_size):
    message = (
        f"📊 Data collection\n"
        f"Universe: {universe_size} symbols\n"
        f"Snapshots: {ok_count} ok, {fail_count} failed"
    )

    if failed_symbols:
        shown = ", ".join(failed_symbols[:15])
        extra = f" (+{len(failed_symbols) - 15} more)" if len(failed_symbols) > 15 else ""
        message += f"\nFailed: {shown}{extra}"

    return message


def format_signal(signal):
    return (
        f"🔔 V4.0 {signal['strategy'].upper()} SIGNAL\n"
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


def format_trade_resolved(symbol, strategy, outcome_type, live_entry, live_target, live_stop, hours):
    won = outcome_type == "target"
    icon = "✅" if won else "❌"

    label = {
        "target": "TARGET HIT",
        "stop": "STOPPED OUT",
        "both_same_candle_conservative_stop": "STOPPED OUT (target+stop same candle)",
        "timeout": "TIMED OUT"
    }.get(outcome_type, outcome_type.upper())

    entry = live_entry if live_entry is not None else None
    exit_price = live_target if won else live_stop

    pct = None
    if entry and exit_price:
        pct = ((exit_price - entry) / entry) * 100 if won else ((exit_price - entry) / entry) * 100

    message = (
        f"{icon} Trade resolved — {label}\n"
        f"Coin: {symbol} [{strategy}]\n"
    )

    if entry is not None and exit_price is not None:
        message += f"Entry: {entry:.8g} → Exit: {exit_price:.8g}"
        if pct is not None:
            message += f" ({pct:+.2f}%)"
        message += "\n"

    message += f"Held: {hours:.1f}h"

    return message


def format_model_retrained(strategy, threshold, temperature, metrics):
    return (
        f"🧠 Model retrained — {strategy}\n"
        f"Samples: {metrics['train_rows']} train / {metrics['validation_rows']} val / {metrics['test_rows']} test\n"
        f"Test logloss: {metrics['test_logloss']:.4f} | Brier: {metrics['test_brier']:.4f}\n"
        f"Test precision @ threshold: {metrics['test_precision']:.1%} ({metrics['test_signals']} signals)\n"
        f"Test EV: {metrics['test_ev']:+.3f}\n"
        f"Threshold: {threshold:.1%} | Temperature: {temperature:.3f}\n"
        f"Model: {VERSION}"
    )


def format_error(context, exc):
    return (
        f"⚠️ Error — {context}\n"
        f"{type(exc).__name__}: {str(exc)[:300]}"
    )


def get_system_marker(key):
    rows = db_query("SELECT value FROM system_state_v401 WHERE key=?", (key,))
    return int(rows[0][0]) if rows and rows[0][0] is not None else 0


def set_system_marker(key, value):
    db_execute("""
    INSERT INTO system_state_v401 (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, int(value)))


def format_health_summary():
    lines = ["📈 Periodic health summary"]

    for strategy in STRATEGIES:
        state = db_query("""
            SELECT trained_samples, threshold, temperature,
                   test_precision, test_ev, last_train_ms
            FROM model_state_v401 WHERE strategy=?
        """, (strategy,))

        open_rows = db_query("""
            SELECT COUNT(*) FROM predictions_v401
            WHERE strategy=? AND signaled=1 AND resolved=0
        """, (strategy,))
        open_count = int(open_rows[0][0]) if open_rows else 0

        window_ms = int(time.time() * 1000) - 30 * 24 * 3600 * 1000
        recent = db_query("""
            SELECT outcome FROM predictions_v401
            WHERE strategy=? AND signaled=1 AND resolved=1 AND resolved_at >= ?
        """, (strategy, window_ms))

        win_rate = (
            sum(1 for r in recent if r[0] == 1) / len(recent)
            if recent else None
        )

        lines.append(f"\n[{strategy}]")
        lines.append(f"Open signaled trades: {open_count}")
        lines.append(
            f"30d win rate: {win_rate:.1%} ({len(recent)} resolved)"
            if win_rate is not None
            else "30d win rate: no resolved trades yet"
        )

        if state:
            samples, threshold, temperature, precision, ev, last_train_ms = state[0]
            trained_ago_h = (
                (int(time.time() * 1000) - last_train_ms) / 3600000
                if last_train_ms else None
            )
            lines.append(
                f"Model: {samples or 0} samples | threshold {threshold or 0:.1%} | "
                f"last trained {trained_ago_h:.0f}h ago" if trained_ago_h is not None
                else f"Model: {samples or 0} samples | threshold {threshold or 0:.1%} | never trained"
            )
        else:
            lines.append("Model: not trained yet")

    return "\n".join(lines)


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
    start_time = time.time()
    init_db()

    log.info(
        "Starting V4.0.1 | device=%s",
        DEVICE
    )

    telegram_send(format_run_started())

    errors = []

    # Wrapping this specifically because it's exactly what crashed last
    # time (the Binance/Kraken 451 failure) — before, that crash happened
    # before any Telegram code ever ran, so there was no notification at
    # all. Now a failure here is reported before the process exits.
    try:
        snapshot_targets, predict_targets, exchange_info, state_counts, btc_context = discover_and_score()
    except Exception as exc:
        log.exception("Fatal setup failure: %s", exc)
        telegram_send(format_error("startup (discover_and_score)", exc))
        raise

    log.info(
        "Universe: %d snapshot targets, %d prediction-eligible | states=%s",
        len(snapshot_targets),
        len(predict_targets),
        state_counts
    )

    # Collect one historical microstructure observation per symbol
    # per hourly bucket — core/incubator/promoted only. Scout coins are
    # scored using cheap ticker+daily-OHLC data alone (no order-book
    # calls) until they've proven themselves worth the extra cost;
    # dormant coins are skipped entirely.
    snapshot_ok = 0
    snapshot_failed = []

    # Pure network fetches, no DB — safe to parallelize. Pre-warms price
    # and depth for every snapshot target before the sequential loop
    # that actually writes to Turso runs.
    price_cache = parallel_fetch(snapshot_targets, get_price)
    depth_cache = parallel_fetch(snapshot_targets, lambda s: get_depth(s, 100))

    for symbol in snapshot_targets:
        try:
            collect_snapshot(
                symbol,
                btc_context,
                price=price_cache.get(symbol),
                depth=depth_cache.get(symbol)
            )
            snapshot_ok += 1
        except Exception as exc:
            log.warning(
                "Snapshot loop failed for %s: %s",
                symbol,
                exc
            )
            snapshot_failed.append(symbol)

    telegram_send(
        format_data_collection(snapshot_ok, len(snapshot_failed), snapshot_failed, len(snapshot_targets))
        + f"\nStates: core={state_counts.get('core',0)} scout={state_counts.get('scout',0)} "
          f"incubator={state_counts.get('incubator',0)} promoted={state_counts.get('promoted',0)} "
          f"dormant={state_counts.get('dormant',0)}"
    )

    # A large fraction of failures suggests something systemic (an API
    # change, a widespread outage) rather than a handful of unlucky
    # symbols — worth a distinct error alert on top of the routine
    # data-collection summary above.
    if snapshot_targets and len(snapshot_failed) / len(snapshot_targets) > 0.5:
        telegram_send(format_error(
            "data collection",
            RuntimeError(f"{len(snapshot_failed)}/{len(snapshot_targets)} snapshots failed this run")
        ))

    # Resolve mature observations using future candle OHLC.
    grouped = unresolved_predictions()
    resolved_total = 0

    # Each symbol needs a different start_ms (its own earliest unresolved
    # prediction), but the fetches themselves are still independent, pure
    # network calls — safe to run concurrently before the sequential
    # resolve loop that writes outcomes to Turso.
    future_cache = parallel_fetch(
        list(grouped.keys()),
        lambda s: get_klines(s, "1h", 1000, start_ms=min(int(r[4]) for r in grouped[s]) + 1)
    )

    for symbol, predictions in grouped.items():
        try:
            count = resolve_symbol(symbol, predictions, future=future_cache.get(symbol))
            resolved_total += count or 0
        except Exception as exc:
            log.warning("Resolution failed for %s: %s", symbol, exc)
            errors.append((f"resolve/{symbol}", exc))

    # Retrain only after enough NEW resolved observations exist.
    # (sends its own 🧠 message internally on success)
    try:
        maybe_train()
    except Exception as exc:
        log.exception("Training failed: %s", exc)
        telegram_send(format_error("training", exc))
        errors.append(("training", exc))

    signals = []

    # Pre-warms 1h/4h/1d candles for every prediction target, concurrently,
    # before the sequential loop below — so every predict_signal() call
    # hits get_klines_cached() as a pure cache read with no network wait.
    prewarm_klines_cache(predict_targets)

    for symbol in predict_targets:
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
                errors.append((f"predict/{symbol}/{strategy}", exc))

    # One aggregated error alert instead of one per failure, to avoid
    # spamming — but every failure still gets logged individually above.
    if errors:
        summary = "\n".join(f"- {ctx}: {type(exc).__name__}: {str(exc)[:120]}" for ctx, exc in errors[:10])
        extra = f"\n(+{len(errors) - 10} more)" if len(errors) > 10 else ""
        telegram_send(f"⚠️ {len(errors)} error(s) this run\n{summary}{extra}")

    # Self-scheduling health summary — fires roughly once every 24h
    # regardless of how often this script itself gets triggered, so it
    # works the same whether the scheduler is hourly, every 2 hours, etc.
    last_summary_ms = get_system_marker("last_health_summary_ms")
    now_ms = int(time.time() * 1000)

    if now_ms - last_summary_ms >= 24 * 3600 * 1000:
        try:
            telegram_send(format_health_summary())
            set_system_marker("last_health_summary_ms", now_ms)
        except Exception as exc:
            log.warning("Health summary failed: %s", exc)

    duration_sec = time.time() - start_time

    purged_dormant = cleanup_old_data()

    log.info(
        "V4 run completed | signals=%d | duration=%.0fs | purged_dormant=%d",
        len(signals),
        duration_sec,
        purged_dormant
    )

    telegram_send(format_run_completed({
        "duration_sec": duration_sec,
        "universe": len(snapshot_targets),
        "signals": len(signals),
        "resolved": resolved_total,
        "errors": len(errors)
    }))


if __name__ == "__main__":
    main()