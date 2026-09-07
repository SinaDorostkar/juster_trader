# Crypto Bot V4.0

V4.0 is the new production candidate. V2.2.4 and V3.1.4 should remain in the repository as testing/baseline implementations but are not run by the production workflow.

## Architecture

- 1H price sequence: TCN + BiLSTM + attention
- 4H price sequence: TCN + BiLSTM + attention
- 1D price sequence: TCN + BiLSTM + attention
- Historical order-book sequence: TCN + BiLSTM + attention
- BTC market-regime context
- Funding and open-interest context
- Calibrated probability
- Time-to-event prediction
- Expected-value filter
- Risk/exposure limits
- Future-candle TP/SL labeling
- Chronological train/validation/test split
- Purged split boundaries
- Sparse training observations to reduce overlapping samples

## Files

trade_v4.py
models/brain_v4_day.pth
models/brain_v4_swing.pth
.github/workflows/crypto-v4.yml

## Important

The first phase intentionally does NOT start with a pretrained V2/V3 brain.

V4 needs to collect its own clean observations and historical order-book snapshots.

A model checkpoint will not exist immediately. Once enough resolved V4 observations accumulate, GitHub Actions trains the model and commits the resulting .pth file back to the repository.

The bot is long-only and does not place exchange orders.

## GitHub Actions

The workflow requires:

- TELEGRAM_TOKEN
- YOUR_CHAT_ID
- CHANNEL_ID (optional)
- TURSO_DATABASE_URL
- TURSO_AUTH_TOKEN

The workflow has contents: write permission because it commits updated model checkpoints.

No VPS or persistent server is required.
