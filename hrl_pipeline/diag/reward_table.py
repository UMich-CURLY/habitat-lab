"""Per-outcome reward decomposition table from per-episode eval stats.

Usage: python3 reward_table.py <stats.json> [<stats.json> ...]

Requires evals run with the training reward config and the
`social_nav_reward_breakdown` measure enabled. Components:
  terminal   = SUCCESS_REWARD * [success]        (added in core/environments.py)
  slack      = SLACK_REWARD  * num_steps         (idem)
  collide / goal_progress / backoff / proximity  (episode sums from the measure)
`total` is the sum of the parts; `reward` is what the evaluator actually
accumulated -- printing both makes any mismatch (i.e. an unaccounted term)
visible instead of silently absorbed.
"""
import json
import sys
from collections import defaultdict

SUCCESS_REWARD = 50.0
SLACK_REWARD = -0.01
BD = "social_nav_reward_breakdown."
PARTS = ["terminal", "slack", "collide", "goal_progress", "backoff", "proximity"]


def outcome(v):
    if v["social_nav_to_pos_success"] > 0.5:
        return "SUCCESS"
    if v["did_collide"] > 0.5:
        return "COLLISION"
    return "TIMEOUT"


def table(path):
    stats = json.load(open(path))
    groups = defaultdict(list)
    for v in stats.values():
        row = {
            "terminal": SUCCESS_REWARD * (v["social_nav_to_pos_success"] > 0.5),
            "slack": SLACK_REWARD * v["num_steps"],
            "collide": v.get(BD + "collide", float("nan")),
            "goal_progress": v.get(BD + "goal_progress", float("nan")),
            "backoff": v.get(BD + "backoff", float("nan")),
            "proximity": v.get(BD + "proximity", float("nan")),
            "reward": v["reward"],
            "steps": v["num_steps"],
        }
        groups[outcome(v)].append(row)

    print(f"\n=== {path} ===")
    hdr = f"{'':14}" + "".join(f"{g:>12}" for g in ("SUCCESS", "COLLISION", "TIMEOUT"))
    print(hdr)
    print(f"{'n episodes':14}" + "".join(
        f"{len(groups.get(g, [])):>12}" for g in ("SUCCESS", "COLLISION", "TIMEOUT")))
    print(f"{'mean steps':14}" + "".join(
        f"{sum(r['steps'] for r in groups[g])/len(groups[g]):>12.0f}"
        if groups.get(g) else f"{'-':>12}" for g in ("SUCCESS", "COLLISION", "TIMEOUT")))
    print("-" * len(hdr))
    for p in PARTS:
        print(f"{p:14}" + "".join(
            f"{sum(r[p] for r in groups[g])/len(groups[g]):>12.1f}"
            if groups.get(g) else f"{'-':>12}" for g in ("SUCCESS", "COLLISION", "TIMEOUT")))
    print("-" * len(hdr))
    for label, keyf in (
        ("total (parts)", lambda r: sum(r[p] for p in PARTS)),
        ("reward (meas.)", lambda r: r["reward"]),
    ):
        print(f"{label:14}" + "".join(
            f"{sum(keyf(r) for r in groups[g])/len(groups[g]):>12.1f}"
            if groups.get(g) else f"{'-':>12}" for g in ("SUCCESS", "COLLISION", "TIMEOUT")))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        table(p)
