"""Publish every deterministic-eval result into TensorBoard.

Training curves already live in tb_*/ , but the numbers that actually decide
anything -- the per-episode deterministic evaluations -- only exist as
stats_*.json / bd_*.json / cert_*.json. This AUTO-DISCOVERS all of them (any
new eval is picked up with no edit) and writes them to `tb_eval_summary/`:

  tb_eval_summary/<group>__<run>/   one run per eval file, scalars at step 0
                                   (overall + per pool-label + reward parts)
  tb_eval_summary/                 a `summary` text tab: full markdown table

Run names come from the filename (`stats_nr1_ck19_holdout.json` ->
`nr1__ck19_holdout`), except for files in ALIASES which get a spelled-out name.

Usage (in the container):  python hrl_pipeline/eval_to_tb.py [--keep]
  --keep : append to an existing tb_eval_summary instead of rebuilding it
"""
import glob
import gzip
import json
import os
import shutil
import sys
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

SD = "/habitat-lab/hrl_pipeline/results"
OUT = "/habitat-lab/hrl_pipeline/tb/eval_summary"
TRAIN36 = "/habitat-lab/data/social_nav_episode_train36_v1.json.gz"
SUCCESS_REWARD, SLACK_REWARD = 50.0, -0.01
BD = "social_nav_reward_breakdown."

# Spelled-out names for the runs whose provenance is documented; everything
# else falls back to the filename.
ALIASES = {
    "stats_teacher_train36bc.json": "teacher__rule_markov_train36_31",
    "stats_teacher_wait_t36.json": "teacher__wait_train36",
    "stats_teacher_wait_bc.json": "teacher__wait_demoset30",
    "stats_bc_train36.json": "BC__nowait_train36",
    "bd_bc.json": "BC__nowait_train36_TRAINREWARD",
    "stats_b2_final.json": "PPO__B2final_train36",
    "stats_g3_final.json": "PPO__G3final_criticlr_train36",
    "stats_prod1_ckpt29.json": "PPO__prod1_final_train36",
    "bd_prod1e.json": "PPO__prod1_final_train36_TRAINREWARD",
    "stats_prod1_ckpt6.json": "PPO__prod1_ckpt6_train36",
    "stats_learned_g3.json": "PPO__G3final_learnedmode_train36",
    "stats_rule_markov_cd17.json": "teacher__rule_markov_cd17",
    "stats_rule_yield_cd17.json": "teacher__rule_yield_legacy_cd17",
    "stats_always_go_cd17.json": "baseline__always_go_cd17",
}


def labels():
    try:
        d = json.load(gzip.open(TRAIN36))
        return {
            e["episode_id"]: e.get("info", {}).get("pool", {}).get("label", "?")
            for e in d["episodes"]
        }
    except Exception:
        return {}


def run_name(fname):
    if fname in ALIASES:
        return ALIASES[fname]
    base = fname[:-5]
    for p in ("stats_", "bd_", "cert3_", "cert_", "recheck_"):  # cert3_ before cert_
        if base.startswith(p):
            base = base[len(p) :]
            break
    head, _, tail = base.partition("_")
    return f"{head}__{tail}" if tail else base


def load_eps(path):
    """-> {episode_id: metrics} or None if this is not an eval-stats file."""
    try:
        s = json.load(open(path))
    except Exception:
        return None
    if not isinstance(s, dict) or not s:
        return None
    # Key is "<scene>|<episode_id>|<eval_idx>": keep the FULL key so repeated
    # evaluations of the same episode all count (keying on episode_id alone
    # silently collapsed evals_per_ep=3 runs down to one sample per episode).
    eps = {}
    for k, v in s.items():
        if not isinstance(v, dict) or "num_steps" not in v:
            return None
        eps[k] = v
    return eps or None


def summarize(eps, lab):
    n = len(eps)
    succ = [v for v in eps.values() if v.get("social_nav_to_pos_success", 0) > 0.5]
    coll = [v for v in eps.values() if v.get("did_collide", 0) > 0.5]
    tmo = [
        v for v in eps.values()
        if v.get("social_nav_to_pos_success", 0) <= 0.5 and v.get("did_collide", 0) <= 0.5
    ]
    out = {
        "eval/n_episodes": n,
        "eval/success_rate": len(succ) / n,
        "eval/collision_rate": len(coll) / n,
        "eval/timeout_rate": len(tmo) / n,
        "eval/n_success": len(succ),
        "eval/n_collision": len(coll),
        "eval/mean_steps": sum(v["num_steps"] for v in eps.values()) / n,
    }
    for tag, rows in (("success", succ), ("collision", coll), ("timeout", tmo)):
        if rows:
            out[f"eval/mean_steps_{tag}"] = sum(r["num_steps"] for r in rows) / len(rows)
    if any("reward" in v for v in eps.values()):
        rw = [v["reward"] for v in eps.values() if "reward" in v]
        out["eval/mean_episode_reward"] = sum(rw) / len(rw)

    by = defaultdict(list)
    for e, v in eps.items():
        if e in lab:
            by[lab[e]].append(v.get("social_nav_to_pos_success", 0))
    for k, vals in by.items():
        out[f"by_label/{k}_success_rate"] = sum(vals) / len(vals)
        out[f"by_label/{k}_n"] = len(vals)

    if any(BD + "collide" in v for v in eps.values()):
        for tag, rows in (("success", succ), ("collision", coll), ("timeout", tmo)):
            if not rows:
                continue
            m = len(rows)
            parts = {
                "terminal": sum(
                    SUCCESS_REWARD * (r.get("social_nav_to_pos_success", 0) > 0.5)
                    for r in rows) / m,
                "slack": sum(SLACK_REWARD * r["num_steps"] for r in rows) / m,
                "collide": sum(r.get(BD + "collide", 0.0) for r in rows) / m,
                "goal_progress": sum(r.get(BD + "goal_progress", 0.0) for r in rows) / m,
                "backoff": sum(r.get(BD + "backoff", 0.0) for r in rows) / m,
                "proximity": sum(r.get(BD + "proximity", 0.0) for r in rows) / m,
            }
            for k, v in parts.items():
                out[f"reward_{tag}/{k}"] = v
            out[f"reward_{tag}/total_parts"] = sum(parts.values())
            if any("reward" in r for r in rows):
                out[f"reward_{tag}/reward_measured"] = sum(
                    r.get("reward", 0.0) for r in rows) / m
    return out


def main():
    if "--keep" not in sys.argv and os.path.isdir(OUT):
        shutil.rmtree(OUT)
    lab = labels()
    files = sorted(
        set(glob.glob(f"{SD}/stats_*.json"))
        | set(glob.glob(f"{SD}/bd_*.json"))
        | set(glob.glob(f"{SD}/cert_*.json"))
        | set(glob.glob(f"{SD}/cert3_*.json"))
        | set(glob.glob(f"{SD}/recheck_*.json"))
    )
    rows, skipped = [], []
    for path in files:
        fname = os.path.basename(path)
        eps = load_eps(path)
        if eps is None:
            skipped.append(fname)
            continue
        res = summarize(eps, lab)
        name = run_name(fname)
        w = SummaryWriter(os.path.join(OUT, name))
        for tag, val in res.items():
            w.add_scalar(tag, float(val), 0)
        w.add_text("source", fname, 0)
        w.close()
        rows.append((name, res))

    hdr = ("| run | n | success | collisions | timeouts | mean steps | "
           "steps(succ) | steps(coll) | ep reward |\n"
           "|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for name, r in sorted(rows, key=lambda x: x[0]):
        g = lambda k, f="{:.0f}": f.format(r[k]) if k in r else "-"  # noqa: E731
        body += (
            f"| {name} | {r['eval/n_episodes']:.0f} | "
            f"{r['eval/n_success']:.0f} ({100*r['eval/success_rate']:.0f}%) | "
            f"{r['eval/n_collision']:.0f} | {r['eval/timeout_rate']*r['eval/n_episodes']:.0f} | "
            f"{r['eval/mean_steps']:.0f} | {g('eval/mean_steps_success')} | "
            f"{g('eval/mean_steps_collision')} | {g('eval/mean_episode_reward', '{:.1f}')} |\n"
        )
    w = SummaryWriter(OUT)
    w.add_text("summary/all_evals", hdr + body, 0)
    w.close()

    print(f"published {len(rows)} eval runs -> {OUT}")
    for name, r in sorted(rows, key=lambda x: x[0]):
        print(f"  {name:46} n={r['eval/n_episodes']:.0f} "
              f"succ={100*r['eval/success_rate']:.0f}% coll={r['eval/n_collision']:.0f}")
    if skipped:
        print(f"\nskipped {len(skipped)} non-eval json: {', '.join(skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))


if __name__ == "__main__":
    main()
