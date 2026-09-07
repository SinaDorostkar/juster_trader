import os
import io

import numpy as np
import pandas as pd
import requests
import turso_serverless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
YOUR_CHAT_ID = int(os.environ['YOUR_CHAT_ID'])

MIN_SAMPLE_PER_BUCKET = 30     # guardrail 1: don't propose based on a bucket this thin
MAX_THRESHOLD_MOVE = 0.05      # guardrail 3: cap any single proposal to ±5 points
DEFAULT_THRESHOLD = 0.70

BUCKET_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
BUCKET_LABELS = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%',
                   '80-85%', '85-90%', '90-95%', '95-100%']

conn = turso_serverless.connect(
    os.environ['TURSO_DATABASE_URL'],
    auth_token=os.environ['TURSO_AUTH_TOKEN'],
)


def db_query(sql, params=()):
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    conn.commit()
    return rows


def send_telegram(text):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                   json={"chat_id": YOUR_CHAT_ID, "text": text}, timeout=15)


def send_telegram_photo(image_bytes, caption):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                   data={"chat_id": YOUR_CHAT_ID, "caption": caption},
                   files={"photo": ("calibration.png", image_bytes, "image/png")}, timeout=30)


def get_current_threshold(config_table):
    rows = db_query(f"SELECT value FROM {config_table} WHERE key='threshold'")
    return float(rows[0][0]) if rows else DEFAULT_THRESHOLD


def load_resolved(table_name):
    """Pulls every resolved, shadow-logged prediction — signaled or not —
    for the full retained window (the trades table is already pruned to
    18 months by the live bot, so no extra date filter is needed here)."""
    rows = db_query(f"""SELECT strategy, predicted_prob, actual_outcome, start_price, end_price, stop_price
        FROM {table_name} WHERE actual_outcome IS NOT NULL AND predicted_prob IS NOT NULL""")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['strategy', 'predicted_prob', 'actual_outcome',
                                       'start_price', 'end_price', 'stop_price'])
    df['win'] = (df['actual_outcome'] == 'win').astype(int)
    gain_frac = (df['end_price'] - df['start_price']) / df['start_price']
    loss_frac = (df['start_price'] - df['stop_price']) / df['start_price']
    df['realized_pct'] = np.where(df['win'] == 1, gain_frac, -loss_frac)
    return df


def propose_threshold(bucket_summary, current_threshold):
    """Find the lowest bucket (by minimum edge) with enough samples and
    positive average expectancy — that's the proposed new floor. Capped
    to a max move from the current threshold either direction."""
    candidates = bucket_summary[bucket_summary['n'] >= MIN_SAMPLE_PER_BUCKET]
    candidates = candidates[candidates['avg_realized_pct'] > 0]
    if candidates.empty:
        return None
    lowest_viable = candidates.index.min()
    idx = BUCKET_LABELS.index(lowest_viable)
    proposed = BUCKET_EDGES[idx]
    delta = max(-MAX_THRESHOLD_MOVE, min(MAX_THRESHOLD_MOVE, proposed - current_threshold))
    return round(current_threshold + delta, 3)


def report_for_strategy(df_strategy, strategy_name, current_threshold, bot_label):
    if df_strategy.empty:
        send_telegram(f"[{bot_label}] [{strategy_name}] No resolved shadow-logged predictions yet — nothing to calibrate.")
        return

    df_strategy = df_strategy.copy()
    df_strategy['bucket'] = pd.cut(df_strategy['predicted_prob'], bins=BUCKET_EDGES,
                                     labels=BUCKET_LABELS, right=False)
    summary = df_strategy.groupby('bucket', observed=True).agg(
        n=('win', 'size'), actual_win_rate=('win', 'mean'), avg_realized_pct=('realized_pct', 'mean'))

    total_n = len(df_strategy)
    overall_win_rate = df_strategy['win'].mean()
    overall_expectancy = df_strategy['realized_pct'].mean()

    proposed = propose_threshold(summary, current_threshold)

    fig, ax = plt.subplots()
    valid = summary.dropna(subset=['actual_win_rate'])
    midpoints = [(BUCKET_EDGES[i] + BUCKET_EDGES[i + 1]) / 2 for i in range(len(BUCKET_LABELS))]
    x = [midpoints[BUCKET_LABELS.index(b)] for b in valid.index]
    ax.plot([0.5, 1.0], [0.5, 1.0], '--', color='gray', label='perfectly calibrated')
    ax.scatter(x, valid['actual_win_rate'], s=[max(10, n) for n in valid['n']], label='actual (size = sample count)')
    ax.axvline(current_threshold, color='orange', linestyle=':', label=f'current threshold ({current_threshold:.0%})')
    if proposed is not None:
        ax.axvline(proposed, color='green', linestyle=':', label=f'proposed ({proposed:.0%})')
    ax.set_xlabel('Predicted confidence (bucket midpoint)')
    ax.set_ylabel('Actual win rate')
    ax.set_title(f'{bot_label} — {strategy_name} calibration')
    ax.legend(fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    table_text = summary.to_string(float_format=lambda x: f"{x:.3f}")
    caption = (
        f"[{bot_label}] [{strategy_name}] Semi-annual calibration\n"
        f"Total resolved predictions: {total_n}\n"
        f"Overall win rate: {overall_win_rate:.1%}\n"
        f"Overall expectancy: {overall_expectancy:+.2%}\n"
        f"Current threshold: {current_threshold:.1%}\n"
    )
    if proposed is None:
        caption += "Proposed threshold: no change — no bucket yet has enough samples with positive expectancy."
    elif abs(proposed - current_threshold) < 0.001:
        caption += f"Proposed threshold: {proposed:.1%} (no change from current)."
    else:
        direction = "lower" if proposed < current_threshold else "raise"
        caption += (f"Proposed threshold: {proposed:.1%} ({direction} from {current_threshold:.1%})\n"
                     f"This is a PROPOSAL ONLY — nothing is applied automatically. "
                     f"Review the chart/table, then update the config table yourself if you agree.")

    send_telegram_photo(buf.getvalue(), caption[:1024])
    send_telegram(f"[{bot_label}] [{strategy_name}] Bucket detail:\n{table_text}")


def main():
    # v2.2.4 (flat-vector model)
    threshold_v22 = get_current_threshold('config')
    df_v22 = load_resolved('trades')
    for strategy_name in (df_v22['strategy'].unique() if not df_v22.empty else ['day', 'swing']):
        report_for_strategy(df_v22[df_v22['strategy'] == strategy_name] if not df_v22.empty else pd.DataFrame(),
                             strategy_name, threshold_v22, 'v2.2.4')

    # v3.1.4 (LSTM sequence model)
    threshold_v31 = get_current_threshold('config_v3')
    df_v31 = load_resolved('trades_v3')
    for strategy_name in (df_v31['strategy'].unique() if not df_v31.empty else ['day', 'swing']):
        report_for_strategy(df_v31[df_v31['strategy'] == strategy_name] if not df_v31.empty else pd.DataFrame(),
                             strategy_name, threshold_v31, 'v3.1.4')

    print("Calibration report finished")


if __name__ == '__main__':
    main()
