# Crypto Bot V4.0.1

Clean causal V4 data/training revision.

## What is fixed
- Uses **USDT-margined Binance perpetual futures consistently** for universe, candles, price, depth, funding and open interest. No spot/futures price/order-book mismatch.
- All multi-timeframe sequences are explicitly cut off at the same primary closed-candle decision time.
- Historical microstructure is queried `<= decision time`; no latest-snapshot leakage.
- Historical BTC regime is computed as-of each sample time.
- Clean V4.0.1 tables are isolated from older V4 observations/models.
- Observations are inserted even when no model checkpoint exists, removing bootstrap deadlock.
- Model-independent decision entry/TP/SL are stored for causal labels.
- Live price is fetched only after model + EV + exposure filters pass; live target/stop/quantity are calculated from that live price.
- Turso threshold/temperature are authoritative when present, with checkpoint fallback.
- Critical Turso commit failures are raised instead of silently ignored.
- Future labels begin strictly after the decision candle close.
- Same-candle TP+SL resolves conservatively as a stop.

## Deployment status
This revision is suitable for GitHub Actions deployment as a **data-collection / paper-signal bot**. It does not place exchange orders.

Do not interpret successful deployment as proof of trading profitability. A live-money decision should wait until V4.0.1 has accumulated enough clean resolved observations, passed chronological validation/test checks, and produced stable calibration/EV results.

## Required GitHub Secrets
- `TELEGRAM_TOKEN`
- `YOUR_CHAT_ID`
- `CHANNEL_ID` (optional)
- `TURSO_DATABASE_URL`
- `TURSO_AUTH_TOKEN`
- `ACCOUNT_EQUITY` (optional)
- `RISK_PER_TRADE` (optional)

The workflow installs Python dependencies and commits `models/*.pth` after successful training.
