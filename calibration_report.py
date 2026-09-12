import os
import io
import math

import numpy as np
import pandas as pd
import requests
import turso_serverless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
YOUR_CHAT_ID = int(os.environ['YOUR_CHAT_ID'])

TURSO_DATABASE_URL = os.environ['TURSO_DATABASE_URL']
TURSO_AUTH_TOKEN = os.environ['TURSO_AUTH_TOKEN']

# Mirrors STRATEGIES' reward:risk ratio from trade_v4_0_1.py — this
# script deliberately doesn't import the main file (which has real
# side effects on load, like connecting to Torch/DB); the two numbers
# actually needed here are duplicated instead.
STRATEGY_RR = {
    "day": 1.8 / 1.0,
    "swing": 3.5 / 1.8,
}

MIN_SAMPLE_PER_BUCKET = 30       # guardrail: don't propose from a thin bucket
MAX_THRESHOLD_MOVE = 0.05        # guardrail: cap any single proposal
MIN_PROMOTIONS_PER_BUCKET = 10   # PROMOTE_SCORE needs enough distinct promotion events, not just rows
MIN_RESOLVED_PER_PROMOTION = 5   # a promoted symbol needs a few resolved trades to count at all

PROB_BUCKET_EDGES = [i / 20 for i in range(21)]           # 0.00 .. 1.00 in 0.05 steps
PROB_BUCKET_LABELS = [f"{PROB_BUCKET_EDGES[i]:.2f}-{PROB_BUCKET_EDGES[i+1]:.2f}" for i in range(20)]

SCORE_BUCKET_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
SCORE_BUCKET_LABELS = [f"{SCORE_BUCKET_EDGES[i]:.2f}-{SCORE_BUCKET_EDGES[i+1]:.2f}" for i in range(10)]

conn = turso_serverless.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)


def db_query(sql, params=()):
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    conn.commit()
    return rows


def send_telegram(text):
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": YOUR_CHAT_ID, "text": text},
        timeout=15
    )
    response.raise_for_status()


def send_telegram_photo(image_bytes, caption):
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
        data={"chat_id": YOUR_CHAT_ID, "caption": caption[:1024]},
        files={"photo": ("calibration.png", image_bytes, "image/png")},
        timeout=30
    )
    response.raise_for_status()


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ──────────────────────────────────────────────
# PART 1 — model win-probability threshold calibration
# (same methodology as the original calibration_report.py, re-pointed
# at v4's actual columns: calibrated_prob instead of predicted_prob,
# model_state_v401.threshold instead of a separate config table.
# Reads ALL resolved rows regardless of train_candidate or signaled —
# calibration wants the full shadow-logged range, not just the subset
# that got alerted or the subset training happens to use.)
# ──────────────────────────────────────────────
def get_current_threshold(strategy):
    rows = db_query("SELECT threshold FROM model_state_v401 WHERE strategy=?", (strategy,))
    return float(rows[0][0]) if rows and rows[0][0] is not None else 0.70


def load_resolved_predictions(strategy):
    rows = db_query("""
        SELECT calibrated_prob, outcome
        FROM predictions_v401
        WHERE strategy=? AND resolved=1 AND outcome IS NOT NULL AND calibrated_prob IS NOT NULL
    """, (strategy,))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['calibrated_prob', 'outcome'])
    df['win'] = (df['outcome'] > 0.5).astype(int)
    rr = STRATEGY_RR[strategy]
    df['realized_pct'] = np.where(df['win'] == 1, rr, -1.0)
    return df


def propose_threshold(bucket_summary, current_threshold):
    candidates = bucket_summary[bucket_summary['n'] >= MIN_SAMPLE_PER_BUCKET]
    candidates = candidates[candidates['avg_realized_pct'] > 0]
    if candidates.empty:
        return None
    lowest_viable = candidates.index.min()
    idx = PROB_BUCKET_LABELS.index(lowest_viable)
    proposed = PROB_BUCKET_EDGES[idx]
    delta = max(-MAX_THRESHOLD_MOVE, min(MAX_THRESHOLD_MOVE, proposed - current_threshold))
    return round(current_threshold + delta, 3)


def report_model_threshold(strategy):
    df = load_resolved_predictions(strategy)
    current_threshold = get_current_threshold(strategy)

    if df.empty:
        send_telegram(f"🎯 [{strategy}] Model threshold calibration\nNo resolved predictions yet — nothing to calibrate.")
        return

    df['bucket'] = pd.cut(df['calibrated_prob'], bins=PROB_BUCKET_EDGES, labels=PROB_BUCKET_LABELS, right=False)
    summary = df.groupby('bucket', observed=True).agg(
        n=('win', 'size'), actual_win_rate=('win', 'mean'), avg_realized_pct=('realized_pct', 'mean'))

    total_n = len(df)
    overall_win_rate = df['win'].mean()
    overall_expectancy = df['realized_pct'].mean()
    proposed = propose_threshold(summary, current_threshold)

    fig, ax = plt.subplots()
    valid = summary.dropna(subset=['actual_win_rate'])
    midpoints = [(PROB_BUCKET_EDGES[i] + PROB_BUCKET_EDGES[i + 1]) / 2 for i in range(len(PROB_BUCKET_LABELS))]
    x = [midpoints[PROB_BUCKET_LABELS.index(b)] for b in valid.index]
    ax.plot([0, 1], [0, 1], '--', color='gray', label='perfectly calibrated')
    ax.scatter(x, valid['actual_win_rate'], s=[max(10, n) for n in valid['n']], label='actual (size = sample count)')
    ax.axvline(current_threshold, color='orange', linestyle=':', label=f'current threshold ({current_threshold:.0%})')
    if proposed is not None:
        ax.axvline(proposed, color='green', linestyle=':', label=f'proposed ({proposed:.0%})')
    ax.set_xlabel('Calibrated win probability (bucket midpoint)')
    ax.set_ylabel('Actual win rate')
    ax.set_title(f'V4 — {strategy} model threshold calibration')
    ax.legend(fontsize=8)

    caption = (
        f"🎯 [{strategy}] Model threshold calibration\n"
        f"Resolved predictions analyzed: {total_n}\n"
        f"Overall win rate: {overall_win_rate:.1%}\n"
        f"Overall expectancy: {overall_expectancy:+.2f}R\n"
        f"Current threshold: {current_threshold:.1%}\n"
    )
    if proposed is None:
        caption += "Proposed threshold: no change — no bucket yet has enough samples with positive expectancy."
    elif abs(proposed - current_threshold) < 0.001:
        caption += f"Proposed threshold: {proposed:.1%} (no change from current)."
    else:
        direction = "lower" if proposed < current_threshold else "raise"
        caption += (f"Proposed threshold: {proposed:.1%} ({direction} from {current_threshold:.1%})\n"
                     f"PROPOSAL ONLY — nothing is applied automatically. Review, then update "
                     f"model_state_v401.threshold yourself if you agree.")

    send_telegram_photo(fig_to_bytes(fig), caption)
    send_telegram(f"[{strategy}] Bucket detail:\n{summary.to_string(float_format=lambda v: f'{v:.3f}')}")

# ──────────────────────────────────────────────
# PART 2 — PROMOTE_SCORE calibration
#
# candidate_state_v401 only ever stores each symbol's CURRENT score
# (upsert) — with no history, there was no way to ask "did coins that
# scored well at promotion time actually turn out well." That's what
# candidate_score_history_v401 (added alongside this report) exists
# for: an append-only log of every score, with the state transition it
# produced, letting us isolate the exact score at each promotion event
# and check it against what that symbol's predictions actually did
# afterward.
#
# DEMOTE_SCORE is deliberately NOT calibrated the same way — see the
# note in report_demote_score() for why that's a harder, structurally
# different problem, not just an omission.
# ──────────────────────────────────────────────
def load_promotion_events():
    rows = db_query("""
        SELECT symbol, score, logged_ms
        FROM candidate_score_history_v401
        WHERE state_before != 'promoted' AND state_after = 'promoted'
    """)
    return pd.DataFrame(rows, columns=['symbol', 'score', 'promoted_ms']) if rows else pd.DataFrame()


def subsequent_performance(symbol, promoted_ms):
    """Win rate and expectancy across every strategy's resolved
    predictions for this symbol made at or after its promotion."""
    rows = db_query("""
        SELECT strategy, outcome FROM predictions_v401
        WHERE symbol=? AND source_close_ms >= ? AND resolved=1 AND outcome IS NOT NULL
    """, (symbol, promoted_ms))
    if len(rows) < MIN_RESOLVED_PER_PROMOTION:
        return None
    realized = [STRATEGY_RR.get(strategy, 1.0) if outcome > 0.5 else -1.0 for strategy, outcome in rows]
    return float(np.mean(realized)), len(rows)


def get_promote_score():
    return 0.55  # PROMOTE_SCORE's current default — see note below


def propose_score_threshold(bucket_summary, current_score):
    candidates = bucket_summary[bucket_summary['n_promotions'] >= MIN_PROMOTIONS_PER_BUCKET]
    candidates = candidates[candidates['avg_expectancy'] > 0]
    if candidates.empty:
        return None
    lowest_viable = candidates.index.min()
    idx = SCORE_BUCKET_LABELS.index(lowest_viable)
    proposed = SCORE_BUCKET_EDGES[idx]
    delta = max(-MAX_THRESHOLD_MOVE, min(MAX_THRESHOLD_MOVE, proposed - current_score))
    return round(current_score + delta, 3)


def report_promote_score():
    events = load_promotion_events()

    if events.empty:
        send_telegram(
            "📈 PROMOTE_SCORE calibration\n"
            "No promotion events logged yet — this analysis needs candidate_score_history_v401 "
            "to accumulate real promotions first. Expected on early reports; check back once the "
            "candidate lifecycle has been running for a while."
        )
        return

    rows = []
    for _, event in events.iterrows():
        perf = subsequent_performance(event['symbol'], event['promoted_ms'])
        if perf is not None:
            expectancy, n_resolved = perf
            rows.append({'symbol': event['symbol'], 'score': event['score'],
                          'expectancy': expectancy, 'n_resolved': n_resolved})

    if not rows:
        send_telegram(
            "📈 PROMOTE_SCORE calibration\n"
            f"{len(events)} promotion event(s) logged, but none yet have "
            f"{MIN_RESOLVED_PER_PROMOTION}+ resolved predictions since promotion — too early to say anything."
        )
        return

    df = pd.DataFrame(rows)
    df['bucket'] = pd.cut(df['score'], bins=SCORE_BUCKET_EDGES, labels=SCORE_BUCKET_LABELS, right=False)
    summary = df.groupby('bucket', observed=True).agg(
        n_promotions=('symbol', 'size'), avg_expectancy=('expectancy', 'mean'))

    current_score = get_promote_score()
    proposed = propose_score_threshold(summary, current_score)

    fig, ax = plt.subplots()
    valid = summary.dropna(subset=['avg_expectancy'])
    midpoints = [(SCORE_BUCKET_EDGES[i] + SCORE_BUCKET_EDGES[i + 1]) / 2 for i in range(len(SCORE_BUCKET_LABELS))]
    x = [midpoints[SCORE_BUCKET_LABELS.index(b)] for b in valid.index]
    ax.axhline(0, color='gray', linestyle='--', label='breakeven')
    ax.scatter(x, valid['avg_expectancy'], s=[max(10, n) * 3 for n in valid['n_promotions']],
               label='actual (size = # promotions)')
    ax.axvline(current_score, color='orange', linestyle=':', label=f'current PROMOTE_SCORE ({current_score:.2f})')
    if proposed is not None:
        ax.axvline(proposed, color='green', linestyle=':', label=f'proposed ({proposed:.2f})')
    ax.set_xlabel('Score at time of promotion (bucket midpoint)')
    ax.set_ylabel('Avg expectancy since promotion (R)')
    ax.set_title('PROMOTE_SCORE calibration')
    ax.legend(fontsize=7)

    caption = (
        f"📈 PROMOTE_SCORE calibration\n"
        f"Promotion events with enough data: {len(df)} / {len(events)} logged\n"
        f"Current PROMOTE_SCORE: {current_score:.2f}\n"
    )
    if proposed is None:
        caption += "Proposed: no change — no score bucket yet has enough promotions with positive expectancy."
    elif abs(proposed - current_score) < 0.001:
        caption += f"Proposed: {proposed:.2f} (no change from current)."
    else:
        direction = "raise" if proposed > current_score else "lower"
        caption += (f"Proposed: {proposed:.2f} ({direction} from {current_score:.2f})\n"
                     f"PROPOSAL ONLY — update PROMOTE_SCORE in trade_v4_0_1.py yourself if you agree. "
                     f"This methodology is newer and less proven than the model-threshold calibration "
                     f"above — treat it with more skepticism until it's been checked across a few reports.")

    send_telegram_photo(fig_to_bytes(fig), caption)
    send_telegram(f"PROMOTE_SCORE bucket detail:\n{summary.to_string(float_format=lambda v: f'{v:.3f}')}")


def report_demote_score():
    """DEMOTE_SCORE deliberately gets a descriptive report, not a
    calibrated proposal. The reason is structural, not laziness: the
    moment a symbol demotes, prediction generation for it stops
    immediately — there's no way to observe what would have happened
    had it kept trading a bit longer past that point. Any 'calibration'
    here would really be comparing demoted coins against a counterfactual
    we deliberately never collected, which isn't a comparison worth
    trusting enough to act on automatically."""
    rows = db_query("""
        SELECT score FROM candidate_score_history_v401
        WHERE state_before != 'dormant' AND state_after = 'dormant'
    """)
    if not rows:
        send_telegram("📉 DEMOTE_SCORE\nNo demotion events logged yet.")
        return

    scores = [r[0] for r in rows]
    send_telegram(
        f"📉 DEMOTE_SCORE — descriptive only, no proposal\n"
        f"{len(scores)} demotion event(s) logged. Avg score at demotion: {np.mean(scores):.3f}\n"
        f"Not calibrated the way PROMOTE_SCORE is: once a coin demotes, we stop generating "
        f"predictions for it, so there's no way to see what its outcomes would have been had it "
        f"stayed promoted a bit longer. Treat this as a status note, not a tuning signal."
    )


def main():
    for strategy in STRATEGY_RR:
        report_model_threshold(strategy)

    report_promote_score()
    report_demote_score()

    print("Calibration report finished")


if __name__ == '__main__':
    main()
