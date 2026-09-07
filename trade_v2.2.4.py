import os
import time

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import turso_serverless

# ──────────────────────────────────────────────
# CONFIG
#   export TELEGRAM_TOKEN="..."
#   export YOUR_CHAT_ID="6039461297"
#   export CHANNEL_ID="@your_channel"        (optional)
#   export TURSO_DATABASE_URL="turso://your-db-yourorg.turso.io"
#   export TURSO_AUTH_TOKEN="..."
# ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
YOUR_CHAT_ID = int(os.environ['YOUR_CHAT_ID'])
CHANNEL_ID = os.environ.get('CHANNEL_ID')

STATIC_WATCHLIST = [
    {"name": "Bitcoin", "symbol_binance": "BTCUSDT"},
    {"name": "Ethereum", "symbol_binance": "ETHUSDT"},
    {"name": "Dogecoin", "symbol_binance": "DOGEUSDT"},
    {"name": "Shiba Inu", "symbol_binance": "SHIBUSDT"},
    {"name": "Solana", "symbol_binance": "SOLUSDT"},
    {"name": "Ripple", "symbol_binance": "XRPUSDT"},
    {"name": "Cardano", "symbol_binance": "ADAUSDT"},
    {"name": "Tron", "symbol_binance": "TRXUSDT"},
    {"name": "Binance Coin", "symbol_binance": "BNBUSDT"},
]

SCAN_TOP_N = 40
MIN_24H_VOLUME_USD = 5_000_000

# timeout_hours: if neither target nor stop is hit within this window, the
# prediction resolves as a loss (conservative) so it still gets a label for
# training/calibration instead of staying open forever.
STRATEGIES = {
    "day": {"interval": "1h", "limit": 168, "target_atr_mult": 1.8, "stop_atr_mult": 1.0,
             "timeout_hours": 240, "model_path": "brain_day_v224.pth"},
    "swing": {"interval": "4h", "limit": 180, "target_atr_mult": 3.5, "stop_atr_mult": 1.8,
               "timeout_hours": 720, "model_path": "brain_swing_v224.pth"},
}
N_FEATURES = 11
DEFAULT_THRESHOLD = 0.70
RETENTION_MONTHS = 18

MAJOR_SYMBOLS = {"BTCUSDT", "ETHUSDT"}
MAX_OPEN_TRADES_TOTAL = int(os.environ.get('MAX_OPEN_TRADES_TOTAL', 6))
MAX_OPEN_MAJORS = int(os.environ.get('MAX_OPEN_MAJORS', 2))
MAX_OPEN_ALTS = int(os.environ.get('MAX_OPEN_ALTS', 4))

ACCOUNT_SIZE_USD = float(os.environ.get('ACCOUNT_SIZE_USD', 1000))
RISK_PER_TRADE_PCT = float(os.environ.get('RISK_PER_TRADE_PCT', 1.0))

# ──────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────
def send_telegram(text):
    for chat_id in [YOUR_CHAT_ID] + ([CHANNEL_ID] if CHANNEL_ID else []):
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                           json={"chat_id": chat_id, "text": text}, timeout=10)
        except Exception as e:
            print(f"Telegram send error ({chat_id}): {e}")

# ──────────────────────────────────────────────
# Database — turso_serverless, the current (non-deprecated) driver for
# remote Turso databases. It implements the standard Python DB-API, same
# shape as the built-in sqlite3 module: conn.execute(sql, params) returns
# a cursor you call .fetchall() on, and writes need an explicit .commit().
# ──────────────────────────────────────────────
conn = turso_serverless.connect(
    os.environ['TURSO_DATABASE_URL'],
    auth_token=os.environ['TURSO_AUTH_TOKEN'],
)


def db_execute(sql, params=()):
    """Run a write statement (CREATE/INSERT/UPDATE/DELETE) and commit."""
    conn.execute(sql, params)
    conn.commit()


def db_query(sql, params=()):
    """Run a SELECT and return all rows as a list of tuples."""
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    conn.commit()
    return rows


db_execute('''
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT, symbol TEXT, strategy TEXT,
    start_price REAL, end_price REAL, stop_price REAL,
    actual_outcome TEXT, opened_at DATETIME DEFAULT CURRENT_TIMESTAMP, closed_at DATETIME,
    hours_to_result REAL,
    rsi REAL, macd REAL, vol_change REAL, price_change REAL, atr REAL, bb_width REAL,
    adx REAL, obv_slope REAL, ob_imbalance REAL, funding_rate REAL, oi_change REAL,
    suggested_qty REAL, position_value_usd REAL,
    predicted_prob REAL,          -- the model's win_prob output at prediction time
    signaled INTEGER,             -- 1 if this prediction crossed the live threshold and was alerted
    threshold_at_time REAL,       -- what the threshold was when this row was created
    trained INTEGER DEFAULT 0     -- has this resolved trade been included in the last retrain
)
''')

db_execute('''
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value REAL
)
''')


def get_threshold():
    rows = db_query("SELECT value FROM config WHERE key='threshold'")
    return float(rows[0][0]) if rows else DEFAULT_THRESHOLD


def prune_old_trades():
    db_execute(f"""DELETE FROM trades
        WHERE opened_at < datetime('now', '-{RETENTION_MONTHS} months') AND actual_outcome IS NOT NULL""")

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class TradePredictor(nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.shared = nn.Linear(n_features, 16)
        self.act = nn.ReLU()
        self.win_head = nn.Linear(16, 1)
        self.time_head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.act(self.shared(x))
        return torch.sigmoid(self.win_head(h)), self.time_head(h)


models = {}
for strat_name, cfg in STRATEGIES.items():
    m = TradePredictor()
    if os.path.exists(cfg["model_path"]):
        ckpt = torch.load(cfg["model_path"])
        m.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded saved model for strategy '{strat_name}'")
    models[strat_name] = m

# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────
def get_ohlcv(symbol, interval, limit):
    try:
        r = requests.get('https://api.binance.com/api/v3/klines',
                          params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        raw = r.json()
        df = pd.DataFrame(raw, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'quote_vol', 'trades', 'taker_base', 'taker_quote', 'ignore'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"OHLCV fetch error ({symbol}): {e}")
        return pd.DataFrame()


def get_current_price(symbol):
    try:
        r = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}', timeout=10)
        if r.status_code == 200:
            return float(r.json()['price'])
    except Exception as e:
        print(f"Price fetch error ({symbol}): {e}")
    return None


def get_top_symbols(n, min_volume_usd):
    try:
        r = requests.get('https://api.binance.com/api/v3/ticker/24hr', timeout=15)
        data = r.json()
    except Exception as e:
        print(f"Top symbols fetch error: {e}")
        return []
    known = {c['symbol_binance'] for c in STATIC_WATCHLIST}
    pairs = [d for d in data if d['symbol'].endswith('USDT') and d['symbol'] not in known]
    pairs.sort(key=lambda d: float(d['quoteVolume']), reverse=True)
    extra = []
    for d in pairs:
        if float(d['quoteVolume']) < min_volume_usd:
            continue
        extra.append({"name": d['symbol'][:-4], "symbol_binance": d['symbol']})
        if len(extra) >= n:
            break
    return extra


def build_watchlist():
    wl = list(STATIC_WATCHLIST)
    if SCAN_TOP_N > 0:
        wl += get_top_symbols(SCAN_TOP_N, MIN_24H_VOLUME_USD)
    return wl


def get_order_book_imbalance(symbol, limit=100):
    try:
        r = requests.get('https://api.binance.com/api/v3/depth',
                          params={'symbol': symbol, 'limit': limit}, timeout=10)
        if r.status_code != 200:
            return 0.0
        data = r.json()
        bid_vol = sum(float(qty) for _, qty in data['bids'])
        ask_vol = sum(float(qty) for _, qty in data['asks'])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
    except Exception as e:
        print(f"Order book fetch error ({symbol}): {e}")
        return 0.0


def get_funding_rate(symbol):
    try:
        r = requests.get('https://fapi.binance.com/fapi/v1/fundingRate',
                          params={'symbol': symbol, 'limit': 1}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                return float(data[-1]['fundingRate'])
    except Exception as e:
        print(f"Funding rate fetch error ({symbol}): {e}")
    return 0.0


def get_oi_change(symbol):
    try:
        r = requests.get('https://fapi.binance.com/futures/data/openInterestHist',
                          params={'symbol': symbol, 'period': '1h', 'limit': 2}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if len(data) == 2:
                prev, curr = float(data[0]['sumOpenInterest']), float(data[1]['sumOpenInterest'])
                if prev > 0:
                    return (curr - prev) / prev
    except Exception as e:
        print(f"Open interest fetch error ({symbol}): {e}")
    return 0.0

# ──────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────
def calculate_indicators(df):
    if len(df) < 40:
        return None
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    prev_close = close.shift(1)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi = (100 - (100 / (1 + gain / loss))).iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).iloc[-1]

    vol_change = volume.pct_change().iloc[-1]
    price_change = close.pct_change().iloc[-1]

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = ((sma20 + 2 * std20 - (sma20 - 2 * std20)) / sma20).iloc[-1]

    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr14 = tr.rolling(14).sum()
    plus_di14 = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14)
    adx = dx.rolling(14).mean().iloc[-1]

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / 10 if len(obv) > 10 else 0.0

    values = [rsi, macd, vol_change, price_change, atr, bb_width, adx, obv_slope]
    return [float(v) if not (v is None or np.isnan(v)) else 0.0 for v in values]

# ──────────────────────────────────────────────
# Exposure limits & position sizing — apply only to alerted ("signaled")
# open positions, never to shadow-logged observational rows
# ──────────────────────────────────────────────
def count_open_exposure():
    rows = db_query("SELECT symbol FROM trades WHERE actual_outcome IS NULL AND signaled=1")
    symbols = [row[0] for row in rows]
    total = len(symbols)
    majors = sum(1 for s in symbols if s in MAJOR_SYMBOLS)
    return total, majors, total - majors


def exposure_allows(symbol):
    total, majors, alts = count_open_exposure()
    if total >= MAX_OPEN_TRADES_TOTAL:
        return False
    return majors < MAX_OPEN_MAJORS if symbol in MAJOR_SYMBOLS else alts < MAX_OPEN_ALTS


def has_open_signaled_trade(coin_name, strategy):
    rows = db_query(
        "SELECT 1 FROM trades WHERE coin=? AND strategy=? AND actual_outcome IS NULL AND signaled=1 LIMIT 1",
        (coin_name, strategy))
    return len(rows) > 0


def calculate_position_size(start_price, stop_price, win_prob, threshold):
    risk_amount = ACCOUNT_SIZE_USD * (RISK_PER_TRADE_PCT / 100)
    risk_per_unit = abs(start_price - stop_price)
    if risk_per_unit <= 0:
        return 0.0, 0.0, risk_amount
    headroom = max(1.0 - threshold, 0.01)  # avoid divide-by-zero if threshold is ever set to 1.0
    confidence_factor = 1.0 + (min(max(win_prob - threshold, 0.0) / headroom, 1.0)) * 0.5
    qty = (risk_amount / risk_per_unit) * confidence_factor
    return qty, qty * start_price, risk_amount

# ──────────────────────────────────────────────
# Trade persistence — every prediction gets logged (shadow logging);
# `signaled` marks which ones crossed the live threshold and were alerted
# ──────────────────────────────────────────────
def save_trade(coin_name, symbol, strategy, start_price, end_price, stop_price,
               features, qty, position_value, predicted_prob, signaled, threshold):
    rsi, macd, vol_change, price_change, atr, bb_width, adx, obv_slope, ob_imbalance, funding_rate, oi_change = features
    db_execute("""INSERT INTO trades
        (coin, symbol, strategy, start_price, end_price, stop_price, rsi, macd, vol_change,
         price_change, atr, bb_width, adx, obv_slope, ob_imbalance, funding_rate, oi_change,
         suggested_qty, position_value_usd, predicted_prob, signaled, threshold_at_time, trained, opened_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0, datetime('now'))""",
        (coin_name, symbol, strategy, start_price, end_price, stop_price,
         rsi, macd, vol_change, price_change, atr, bb_width, adx, obv_slope,
         ob_imbalance, funding_rate, oi_change, qty, position_value, predicted_prob, int(signaled), threshold))


def close_trade(trade_id, outcome):
    rows = db_query("SELECT opened_at FROM trades WHERE id=?", (trade_id,))
    opened_at = rows[0][0]
    db_execute("""UPDATE trades SET actual_outcome=?, closed_at=datetime('now'),
        hours_to_result=(julianday('now') - julianday(?)) * 24 WHERE id=?""",
        (outcome, opened_at, trade_id))

# ──────────────────────────────────────────────
# Retraining — full retrain on the whole 18-month window whenever any
# resolved-but-untrained row exists; `trained` is a pure trigger flag,
# never used to select a training subset.
# ──────────────────────────────────────────────
def retrain_model(strategy):
    rows = db_query(
        "SELECT 1 FROM trades WHERE strategy=? AND actual_outcome IS NOT NULL AND trained=0 LIMIT 1",
        (strategy,))
    if not rows:
        print(f"[{strategy}] No new resolved trades since last retrain — skipping")
        return

    rows = db_query("""SELECT rsi, macd, vol_change, price_change, atr, bb_width, adx, obv_slope,
        ob_imbalance, funding_rate, oi_change, actual_outcome, hours_to_result
        FROM trades WHERE strategy=? AND actual_outcome IS NOT NULL""", (strategy,))

    if len(rows) < 5:
        print(f"[{strategy}] Not enough completed trades to retrain yet")
        return

    X, y_win, y_hours = [], [], []
    for rsi, macd, vol, pchange, atr, bb, adx, obv, obi, fund, oi, outcome, hours in rows:
        X.append([rsi or 50.0, macd or 0.0, vol or 0.0, pchange or 0.0, atr or 0.0, bb or 0.0,
                   adx or 0.0, obv or 0.0, obi or 0.0, fund or 0.0, oi or 0.0])
        y_win.append(1.0 if outcome == 'win' else 0.0)
        y_hours.append(np.log1p(hours) if hours else 0.0)

    X = torch.tensor(X, dtype=torch.float32)
    y_win = torch.tensor(y_win, dtype=torch.float32).unsqueeze(1)
    y_hours = torch.tensor(y_hours, dtype=torch.float32).unsqueeze(1)

    n = X.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, int(n * 0.2))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, yw_train, yh_train = X[train_idx], y_win[train_idx], y_hours[train_idx]
    X_val, yw_val, yh_val = X[val_idx], y_win[val_idx], y_hours[val_idx]

    model = TradePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion_win, criterion_time = nn.BCELoss(), nn.MSELoss()

    best_val_loss, best_state, no_improve = float('inf'), None, 0
    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        win_pred, hours_pred = model(X_train)
        loss = criterion_win(win_pred, yw_train) + 0.3 * criterion_time(hours_pred, yh_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vw, vh = model(X_val)
            val_loss = (criterion_win(vw, yw_val) + 0.3 * criterion_time(vh, yh_val)).item()
        if val_loss < best_val_loss:
            best_val_loss, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= 20:
                break

    model.load_state_dict(best_state)
    models[strategy] = model

    torch.save({'model_state_dict': model.state_dict()}, STRATEGIES[strategy]["model_path"])
    db_execute("UPDATE trades SET trained=1 WHERE strategy=? AND actual_outcome IS NOT NULL", (strategy,))
    print(f"[{strategy}] Retrained on {len(rows)} examples (best val loss {best_val_loss:.4f}) → saved")

# ──────────────────────────────────────────────
# Core analysis — logs every prediction (shadow logging); only alerts and
# opens a tracked position for ones that cross the threshold AND clear
# the dedup/exposure gates
# ──────────────────────────────────────────────
def analyze_and_suggest(coin, strategy_name, threshold):
    cfg = STRATEGIES[strategy_name]
    symbol = coin['symbol_binance']
    df = get_ohlcv(symbol, cfg["interval"], cfg["limit"])
    if df.empty or len(df) < 40:
        return

    price_features = calculate_indicators(df)
    if price_features is None:
        return

    ob_imbalance = get_order_book_imbalance(symbol)
    funding_rate = get_funding_rate(symbol)
    oi_change = get_oi_change(symbol)
    features = price_features + [ob_imbalance, funding_rate, oi_change]

    inputs = torch.tensor([features], dtype=torch.float32)
    model = models[strategy_name]
    model.eval()
    with torch.no_grad():
        win_prob, log_hours = model(inputs)
    win_prob = win_prob.item()
    predicted_hours = max(float(np.expm1(log_hours.item())), 0.0)

    current_price = get_current_price(symbol)
    if not current_price:
        return

    atr = features[4] if features[4] > 0 else current_price * 0.01
    start_price = current_price
    end_price = start_price + atr * cfg["target_atr_mult"]
    stop_price = start_price - atr * cfg["stop_atr_mult"]

    signaled = win_prob > threshold
    qty = position_value = 0.0
    risk_amount = 0.0
    should_alert = False

    if signaled and not has_open_signaled_trade(coin['name'], strategy_name) and exposure_allows(symbol):
        should_alert = True
        qty, position_value, risk_amount = calculate_position_size(start_price, stop_price, win_prob, threshold)

    # every prediction gets logged, whether or not it was alerted
    save_trade(coin['name'], symbol, strategy_name, start_price, end_price, stop_price,
               features, qty, position_value, win_prob, should_alert, threshold)

    if should_alert:
        gain_pct = (end_price - start_price) / start_price * 100
        loss_pct = (start_price - stop_price) / start_price * 100
        msg = (
            f"[{strategy_name.upper()}] BUY signal {coin['name']}\n"
            f"Now: ${current_price:,.4f}\n"
            f"Target: ${end_price:,.4f}  (+{gain_pct:.1f}%)\n"
            f"Stop: ${stop_price:,.4f}  (-{loss_pct:.1f}%)\n"
            f"Model confidence: {win_prob:.1%} (threshold {threshold:.1%})\n"
            f"Est. time to target: ~{predicted_hours:.0f}h\n"
            f"Order book: {'buy-heavy' if ob_imbalance > 0.05 else 'sell-heavy' if ob_imbalance < -0.05 else 'balanced'} ({ob_imbalance:+.2f})\n"
            f"Funding rate: {funding_rate:+.4%} | OI change: {oi_change:+.2%}\n"
            f"Suggested size: {qty:.4f} units (~${position_value:,.2f}, risking ~${risk_amount:,.2f})"
        )
        send_telegram(msg)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Run started")
    threshold = get_threshold()
    watchlist = build_watchlist()

    for coin in watchlist:
        for strategy_name, cfg in STRATEGIES.items():
            analyze_and_suggest(coin, strategy_name, threshold)

            open_rows = db_query("""SELECT id, opened_at, end_price, stop_price, signaled
                FROM trades WHERE coin=? AND strategy=? AND actual_outcome IS NULL""",
                (coin['name'], strategy_name))

            if not open_rows:
                continue
            current = get_current_price(coin['symbol_binance'])
            if not current:
                continue

            for tid, opened_at, end_price, stop_price, signaled in open_rows:
                hours_row = db_query("SELECT (julianday('now') - julianday(?)) * 24", (opened_at,))
                hours_open = hours_row[0][0]
                outcome = None
                if current >= end_price:
                    outcome = 'win'
                elif current <= stop_price:
                    outcome = 'loss'
                elif hours_open >= cfg["timeout_hours"]:
                    outcome = 'loss'  # timed out — conservative label, still resolved for training

                if outcome:
                    close_trade(tid, outcome)
                    if signaled:
                        label = "WIN!" if outcome == 'win' else "STOPPED OUT"
                        send_telegram(f"{label} [{strategy_name}] {coin['name']} at ${current:,.4f}")

    for strategy_name in STRATEGIES:
        retrain_model(strategy_name)

    prune_old_trades()
    print("Run finished")


if __name__ == '__main__':
    main()
