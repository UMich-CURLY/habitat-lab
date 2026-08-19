"""Yield-geometry diagnosis for the layer-3 reward design.

Usage: python analyze_yield_geometry.py <tag> [<tag> ...]
  (tags reference hrl_pipeline/trace_<tag>.jsonl + hl_dec_trace_<tag>.jsonl
   + stats_trace_<tag>.json produced by _collect_traces.sh)

Answers, from per-step traces of several policies on train36_v1:
  Q1 geometry over time (robot/human positions, lateral offset to the human
     path, in-corridor flag)   Q2 risk-metric candidates (MPD-hold, MPD-go
     kinematic counterfactual, TTC, future-corridor overlap)   Q3 extra
     backoff beyond first-safe   Q4 extra wait beyond earliest-safe-release
  Q5 the too-early-GO boundary (state at final GO before collisions vs
     successes)   Q6 outcome-stratified waste decomposition   Q7 per-decision
     state deltas by action
  A/B/C: which metric says "safe to stop backing off" / "safe to GO" /
     whether slow successes lose steps to extra_backoff or extra_wait.

Human/goal world positions are reconstructed from the polar feat (sensor
frame: bearing=atan2(lat,fwd), lateral=[fwd_z,-fwd_x]); the yaw->forward
convention is self-calibrated per episode by minimizing the spread of the
reconstructed (static) nav goal. calib_std is reported -- treat episodes
above 0.5 m with suspicion. The GO counterfactual uses the episode's ACTUAL
human future trajectory (deterministic replay) + a straight-line constant
speed robot model: crude about walls, honest about timing.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("HAB", "/habitat-lab")
SD = f"{ROOT}/hrl_pipeline/results"
CORRIDOR_W = 1.1     # LL yield sensor's human-corridor clearance
SAFE_MPD = 0.9       # robot half-len 0.45 + human 0.3 + slack
PASS_MARGIN = 0.5    # m past the conflict point to call it "passed"

SKILL = {0: "backoff", 1: "wait", 2: "go"}


def load_tag(tag):
    """-> list of episodes: dict(id, rows, outcome)."""
    stats = list(json.load(open(f"{SD}/stats_trace_{tag}.json")).items())
    eps, cur = [], []
    for ln in open(f"{SD}/trace_{tag}.jsonl"):
        d = json.loads(ln)
        # Drop meta and agent_1 rows (the class-level act patch also fired for
        # the human's hierarchical policy in early collections; those rows
        # carry no robot feat/loc).
        if d.get("meta") or d["feat"] is None or d["loc"] is None:
            continue
        if d["mask"] == 0.0 and cur:
            eps.append(cur)
            cur = []
        cur.append(d)
    if cur:
        eps.append(cur)
    assert len(eps) == len(stats), (tag, len(eps), len(stats))
    out = []
    for rows, (k, v) in zip(eps, stats):
        out.append(
            dict(
                id=int(k.split("|")[1]),
                rows=rows,
                success=v["social_nav_to_pos_success"] > 0.5,
                collided=v["did_collide"] > 0.5,
                steps=int(v["num_steps"]),
            )
        )
        assert abs(len(rows) - out[-1]["steps"]) <= 2, (
            tag, out[-1]["id"], len(rows), out[-1]["steps"])
    return out


def calibrate(rows):
    """Pick the yaw->forward convention minimizing reconstructed-goal spread."""
    loc = np.array([r["loc"] for r in rows])
    feat = np.array([r["feat"] for r in rows])
    rob = loc[:, [0, 2]]
    yaw = loc[:, 3]
    cands = []
    for sx, sy, swap in [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                         (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]:
        f = np.stack([sx * np.cos(yaw), sy * np.sin(yaw)], 1)
        if swap:
            f = f[:, ::-1]
        cands.append(f)
    best, best_std, best_goal = None, np.inf, None
    m = feat[:, 4] > 0.5
    if m.sum() < 10:
        m = np.ones(len(rows), bool)
    for f in cands:
        lat = np.stack([f[:, 1], -f[:, 0]], 1)
        goal = (rob + feat[:, [4]] * (np.cos(feat[:, [5]]) * f
                                      + np.sin(feat[:, [5]]) * lat))
        std = float(goal[m].std(axis=0).mean())
        if std < best_std:
            best, best_std, best_goal = (f, lat), std, goal[m].mean(0)
    f, lat = best
    human = (rob + feat[:, [0]] * (np.cos(feat[:, [1]]) * f
                                   + np.sin(feat[:, [1]]) * lat))
    return rob, human, best_goal, best_std


def poly_dist(p, poly):
    """Min distance from point p to a polyline (vertex approximation)."""
    if len(poly) == 0:
        return np.inf
    return float(np.min(np.linalg.norm(poly - p, axis=1)))


def analyze_episode(ep, v_r):
    rows = ep["rows"]
    rob, hum, goal, calib_std = calibrate(rows)
    T = len(rows)
    d = np.array([r["feat"][0] for r in rows])
    gd = np.array([r["feat"][4] for r in rows])
    skill = np.array([r["skill"] for r in rows])

    # Human arc-length + conflict point (closest approach of the two actual
    # trajectories, in human-path arc-length terms).
    seg = np.linalg.norm(np.diff(hum, axis=0), axis=1)
    s_h = np.concatenate([[0.0], np.cumsum(seg)])
    # subsample robot traj for the pairwise min (speed)
    rsub = rob[:: max(1, T // 200)]
    dm = np.linalg.norm(hum[:, None, :] - rsub[None, :, :], axis=2)
    s_conflict = s_h[int(np.argmin(dm.min(axis=1)))]

    # Per-step metrics.
    mpd_hold = np.empty(T)
    mpd_go = np.empty(T)
    in_corr = np.zeros(T, bool)
    human_passed = np.zeros(T, bool)
    to_goal = goal[None, :] - rob
    tg_n = np.linalg.norm(to_goal, axis=1)
    for t in range(T):
        fut = hum[t:]
        mpd_hold[t] = poly_dist(rob[t], fut)
        in_corr[t] = poly_dist(rob[t], fut) < CORRIDOR_W
        human_passed[t] = (s_h[t] > s_conflict + PASS_MARGIN) and (
            t + 1 >= T or s_h[min(t + 20, T - 1)] >= s_h[t]
        )
        # GO counterfactual: straight line to goal at v_r vs actual human future.
        H = min(T - t, 250)
        tau = np.arange(H)
        adv = np.minimum(v_r * tau, tg_n[t])
        pos = rob[t] + (to_goal[t] / (tg_n[t] + 1e-9)) * adv[:, None]
        mpd_go[t] = float(np.min(np.linalg.norm(pos - hum[t: t + H], axis=1)))

    return dict(
        rob=rob, hum=hum, d=d, gd=gd, skill=skill, mpd_hold=mpd_hold,
        mpd_go=mpd_go, in_corr=in_corr, human_passed=human_passed,
        calib_std=calib_std, T=T,
    )


def runs_of(skill, want):
    """Contiguous index runs where skill in want."""
    out, s = [], None
    for t, k in enumerate(skill):
        if k in want and s is None:
            s = t
        elif k not in want and s is not None:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, len(skill)))
    return out


STOP_METRICS = {
    "d>=3.5": lambda m, t: m["d"][t] >= 3.5,
    "mpd_hold>=1.0": lambda m, t: m["mpd_hold"][t] >= 1.0,
    "out_of_corridor": lambda m, t: not m["in_corr"][t],
}
GO_METRICS = {
    "human_passed": lambda m, t: bool(m["human_passed"][t]),
    "mpd_go>=0.9": lambda m, t: m["mpd_go"][t] >= SAFE_MPD,
    "mpd_go>=0.6": lambda m, t: m["mpd_go"][t] >= 0.6,
    "d>=3.5": lambda m, t: m["d"][t] >= 3.5,
}


def first_true(metric, m, a, b):
    for t in range(a, b):
        if metric(m, t):
            return t
    return None


def main(tags):
    all_ep = {}
    for tag in tags:
        eps = load_tag(tag)
        # robot speed while in go-skill (median step displacement)
        disp = []
        for ep in eps:
            loc = np.array([r["loc"] for r in ep["rows"]])[:, [0, 2]]
            sk = np.array([r["skill"] for r in ep["rows"]])
            dd = np.linalg.norm(np.diff(loc, axis=0), axis=1)
            disp += list(dd[(sk[:-1] == 2) & (dd > 1e-4)])
        v_r = float(np.median(disp)) if disp else 0.05
        for ep in eps:
            ep["m"] = analyze_episode(ep, v_r)
        all_ep[tag] = eps
        print(f"[{tag}] v_r={v_r:.3f} m/step  "
              f"calib_std med={np.median([e['m']['calib_std'] for e in eps]):.3f} "
              f"max={max(e['m']['calib_std'] for e in eps):.3f}")

    ep_csv = open(f"{SD}/yield_geom_episodes.csv", "w", newline="")
    w = csv.writer(ep_csv)
    w.writerow(["tag", "id", "outcome", "steps", "go_steps", "backoff_steps",
                "wait_steps", "extra_backoff", "extra_wait", "calib_std"])

    print("\n== Q3/Q4/Q6: waste decomposition "
          "(extra_backoff: mpd_hold>=1.0; extra_wait: mpd_go>=0.9) ==")
    for tag, eps in all_ep.items():
        strata = defaultdict(list)
        succ_steps = [e["steps"] for e in eps if e["success"]]
        med = np.median(succ_steps) if succ_steps else 0
        for e in eps:
            m = e["m"]
            xb = xw = 0
            for a, b in runs_of(m["skill"], {0}):
                t0 = first_true(STOP_METRICS["mpd_hold>=1.0"], m, a, b)
                if t0 is not None:
                    xb += b - t0
            for a, b in runs_of(m["skill"], {0, 1}):
                t0 = first_true(GO_METRICS["mpd_go>=0.9"], m, a, b)
                if t0 is not None and b < m["T"]:
                    xw += b - t0
            e["extra_backoff"], e["extra_wait"] = xb, xw
            oc = ("collision" if e["collided"] else
                  "timeout" if not e["success"] else
                  "fast_success" if e["steps"] <= med else "slow_success")
            e["outcome"] = oc
            sk = m["skill"]
            e["go_steps"] = int((sk == 2).sum())
            e["backoff_steps"] = int((sk == 0).sum())
            e["wait_steps"] = int((sk == 1).sum())
            strata[oc].append(e)
            w.writerow([tag, e["id"], oc, e["steps"], e["go_steps"],
                        e["backoff_steps"], e["wait_steps"], xb, xw,
                        f"{m['calib_std']:.3f}"])
        for oc in ("fast_success", "slow_success", "collision", "timeout"):
            g = strata.get(oc, [])
            if not g:
                continue
            print(f"  {tag:<8} {oc:<13} n={len(g):<3} "
                  f"steps={np.mean([e['steps'] for e in g]):7.0f}  "
                  f"go={np.mean([e['go_steps'] for e in g]):6.0f}  "
                  f"backoff={np.mean([e['backoff_steps'] for e in g]):6.0f}  "
                  f"wait={np.mean([e['wait_steps'] for e in g]):5.0f}  "
                  f"extra_bo={np.mean([e['extra_backoff'] for e in g]):6.0f}  "
                  f"extra_wait={np.mean([e['extra_wait'] for e in g]):6.0f}")

    print("\n== Q3: stop-backoff metric comparison (per backoff run) ==")
    for tag, eps in all_ep.items():
        agg = {k: [0, 0, 0] for k in STOP_METRICS}  # fired, extra_steps, runs
        for e in eps:
            m = e["m"]
            for a, b in runs_of(m["skill"], {0}):
                if b - a < 3:
                    continue
                for name, fn in STOP_METRICS.items():
                    agg[name][2] += 1
                    t0 = first_true(fn, m, a, b)
                    if t0 is not None:
                        agg[name][0] += 1
                        agg[name][1] += b - t0
        for name, (fired, extra, runs) in agg.items():
            if runs:
                print(f"  {tag:<8} {name:<16} fires {fired}/{runs} runs, "
                      f"mean extra-steps-after-fire "
                      f"{extra / max(fired, 1):6.1f}")

    print("\n== Q5/B: state at the FINAL go-onset (collision vs success) ==")
    feats = ["d", "mpd_go", "mpd_hold", "human_passed", "in_corr"]
    pool = defaultdict(list)
    for tag, eps in all_ep.items():
        for e in eps:
            m = e["m"]
            on = [t for t in range(1, m["T"])
                  if m["skill"][t] == 2 and m["skill"][t - 1] != 2]
            if not on:
                continue
            t = on[-1]
            row = dict(tag=tag, id=e["id"], coll=e["collided"],
                       d=m["d"][t], mpd_go=m["mpd_go"][t],
                       mpd_hold=m["mpd_hold"][t],
                       human_passed=int(m["human_passed"][t]),
                       in_corr=int(m["in_corr"][t]))
            pool[e["collided"]].append(row)
    for coll, rows in sorted(pool.items()):
        lab = "collision" if coll else "success"
        print(f"  final-GO before {lab:<9} n={len(rows):<3} " + "  ".join(
            f"{f}={np.mean([r[f] for r in rows]):.2f}" for f in feats))
    # 1-D separation power of each metric
    print("  -- separation (best threshold accuracy, pooled) --")
    for f in ("d", "mpd_go", "mpd_hold", "human_passed"):
        xs = [(r[f], r["coll"]) for rows in pool.values() for r in rows]
        vals = sorted(set(x for x, _ in xs))
        best = 0.0
        bthr = None
        for thr in vals:
            acc = np.mean([(x >= thr) != c for x, c in xs])
            if max(acc, 1 - acc) > best:
                best, bthr = max(acc, 1 - acc), thr
        print(f"     {f:<14} best_acc={best:.2f} @ thr={bthr:.2f}")

    print("\n== Q7: per-HL-decision state deltas by action ==")
    for tag, eps in all_ep.items():
        by_act = defaultdict(list)
        for e in eps:
            m = e["m"]
            sk = m["skill"]
            bounds = [0] + [t for t in range(1, m["T"]) if sk[t] != sk[t - 1]] \
                + [m["T"] - 1]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b <= a:
                    continue
                by_act[SKILL.get(sk[a], sk[a])].append(
                    (m["gd"][b] - m["gd"][a], m["mpd_hold"][b] - m["mpd_hold"][a],
                     m["d"][b] - m["d"][a], b - a))
        for act, rows in sorted(by_act.items()):
            r = np.array(rows)
            print(f"  {tag:<8} {act:<8} n={len(rows):<4} "
                  f"dGoal={r[:, 0].mean():+.2f}  dMPDhold={r[:, 1].mean():+.2f} "
                  f"dHumanDist={r[:, 2].mean():+.2f}  span={r[:, 3].mean():.0f}")

    print(f"\nper-episode CSV -> {SD}/yield_geom_episodes.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or ["teacher"])
