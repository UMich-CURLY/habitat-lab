"""Offline conflict-rate / observability analysis of hl_decisions_*.jsonl.

Usage: python analyze_hl_decisions.py <jsonl> [<jsonl> ...]

Per file, grouped by env, reports:
  - action distribution and per-episode action transitions
  - memoryless-rule replay disagreement: fraction of decisions a pure function
    of the CURRENT 6-dim feat cannot explain (old teacher's hidden state)
  - (obs, prev_action)-conditioned replay for rule_markov logs (needs the
    "approach"/"prev_action" keys): residual = only hysteresis-tick rows
  - grid mixed-label-cell rate + irreducible BC error (Bayes floor)
  - yield_branch shares when present

Baselines to reproduce on hl_decisions_rule_yield.jsonl (2026-08 audit):
  memoryless disagreement 7.5%, GO-in-conflict 17.5%, grid-0.2 mixed 14.9%.
"""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

DIST_T = 3.0


def old_conflict(feat):
    """Legacy rule_yield conflict predicate over the 6-dim feat."""
    hd, hb, hh, hs = feat[0], feat[1], feat[2], feat[3]
    return hd < DIST_T and abs(hb) < 1.75 and abs(hh) > math.pi / 2 and hs > 0.1


def markov_action(feat, approach, prev_a, ticks, P):
    """Replay of the rule_markov teacher (must mirror scripted_hl_eval.py).
    ticks is the mutable {"trig": n, "rel": n} state; returns action."""
    d, b, rh = feat[0], feat[1], feat[2]
    yielding = prev_a in (0, 1)
    if not yielding:
        trig = (
            d < P["trig_dist"]
            and abs(b) < P["trig_bearing"]
            and abs(rh) > P["trig_rel_heading"]
            and approach > P["trig_approach"]
        )
        ticks["trig"] = ticks["trig"] + 1 if trig else 0
        ticks["rel"] = 0
        ticks["boff"] = 0
        return 0 if ticks["trig"] >= P["trig_ticks"] else 2
    still = (d < P["rel_dist"] and approach > P["rel_approach"]) or (
        d < P["rel_min_dist"] and ticks["boff"] < P["wait_after_ticks"]
    )
    ticks["rel"] = ticks["rel"] + 1 if not still else 0
    ticks["trig"] = 0
    if ticks["rel"] >= P["rel_ticks"]:
        return 2
    if ticks["boff"] >= P["wait_after_ticks"]:
        return 1
    ticks["boff"] += 1
    return 0


MARKOV_PARAMS = dict(
    trig_dist=3.0, trig_bearing=1.75, trig_rel_heading=math.pi / 2,
    trig_approach=0.15, trig_ticks=1,
    rel_dist=3.5, rel_approach=0.05, rel_min_dist=1.0, rel_ticks=2,
    wait_after_ticks=6,
)

ACT2IDX = {"backoff": 0, "wait": 1, "go_to_goal": 2}


def grid_stats(X, y, g):
    med = np.median(X, 0)
    scale = np.percentile(np.abs(X - med), 90, axis=0)
    scale[scale < 1e-6] = 1.0
    cells = defaultdict(Counter)
    for xi, yi in zip(np.floor((X - med) / scale / g).astype(int), y):
        cells[tuple(xi)][yi] += 1
    mixed = [c for c in cells.values() if len(c) > 1]
    n_mixed = sum(sum(c.values()) for c in mixed)
    irr = sum(sum(c.values()) - max(c.values()) for c in mixed)
    return len(cells), len(mixed), n_mixed / len(y), irr / len(y)


def analyze(path):
    recs = [json.loads(l) for l in open(path)]
    print(f"\n=== {path}: {len(recs)} decisions ===")
    by_env = defaultdict(list)
    for r in recs:
        by_env[r.get("env", 0)].append(r)
    print(f"envs: {sorted(by_env)}  actions: {Counter(r['action'] for r in recs)}")

    has_markov_keys = all(k in recs[0] for k in ("approach", "prev_action"))
    X = np.array([r["feat"] for r in recs], dtype=np.float64)
    y = np.array([ACT2IDX[r["action"]] for r in recs])

    # --- memoryless legacy replay (conflict -> backoff else go) ---
    conf = np.array([old_conflict(f) for f in X])
    memless = np.where(conf, 0, 2)
    dis = memless != y
    n_conf = max(int(conf.sum()), 1)
    print(
        f"memoryless-legacy replay: disagree {dis.sum()}/{len(y)}"
        f" ({100*dis.mean():.1f}%) | conflict-true {conf.sum()},"
        f" GO-in-conflict {int((conf & (y == 2)).sum())}"
        f" ({100*(conf & (y == 2)).sum()/n_conf:.1f}%)"
    )

    # --- (obs, prev_action)-conditioned rule_markov replay ---
    if has_markov_keys:
        bad = total = 0
        for env, rows in by_env.items():
            ticks = {"trig": 0, "rel": 0, "boff": 0}
            for r in rows:
                if r["mask"] == 0.0:
                    ticks = {"trig": 0, "rel": 0, "boff": 0}
                a = markov_action(
                    r["feat"], r["approach"], r["prev_action"], ticks, MARKOV_PARAMS
                )
                total += 1
                if a != ACT2IDX[r["action"]]:
                    bad += 1
        print(
            f"(obs,prev_action)-conditioned markov replay:"
            f" disagree {bad}/{total} ({100*bad/max(total,1):.2f}%)"
        )

    # --- grid mixed-cell / irreducible error ---
    feats_used = X if not has_markov_keys else np.column_stack(
        [X, [r["approach"] for r in recs], [r["prev_action"] for r in recs]]
    )
    for g in (0.1, 0.2):
        nc, nm, pim, irr = grid_stats(feats_used, y, g)
        print(
            f"grid={g}: cells={nc} mixed={nm}"
            f" samples-in-mixed={100*pim:.1f}% irreducible-err={100*irr:.2f}%"
        )

    # --- transitions per episode ---
    n_ep = trans = 0
    for env, rows in by_env.items():
        prev = None
        for r in rows:
            if r["mask"] == 0.0:
                n_ep += 1
                prev = None
            if prev is not None and r["action"] != prev:
                trans += 1
            prev = r["action"]
    print(f"episodes={n_ep} action-transitions={trans}"
          f" ({trans/max(n_ep,1):.1f}/ep)")

    if "yield_branch" in recs[0]:
        print(f"yield_branch shares: {Counter(r['yield_branch'] for r in recs)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
