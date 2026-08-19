"""Build `tb_clean/` -- the TensorBoard you actually want to look at.

`tb/` is the raw archive: 30 run dirs, 7 duplicate symlinks, 125 eval sub-runs and
48-61 scalar tags per training run, which is why the board is unreadable. Nothing
there is deleted; this script COPIES a curated subset into `tb_clean/`:

  * only runs that back a kept checkpoint or a documented result (11 + 15 evals,
    down from ~280 run entries)
  * only tags worth plotting (~20 per run, down from 48-61) -- the dropped ones are
    profiler timings, scene-collision counters and the inherited habitat social-nav
    stats, three of which are misspelled upstream (`frist_ecnounter_steps`)
  * readable run names, so the run list reads like the results table

Tag names are NOT renamed: every doc, csv and stats file refers to them as they are.

  DEX 'python hrl_pipeline/diag/tb_curate.py'          # rebuild tb_clean/
  tensorboard --logdir hrl_pipeline/tb_clean --port 6006
"""
import glob
import os
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
from torch.utils.tensorboard import SummaryWriter

TB = "/habitat-lab/hrl_pipeline/tb"
OUT = "/habitat-lab/hrl_pipeline/tb_clean"

# run dir -> name in tb_clean. Order fixes the display order.
TRAIN_RUNS = [
    ("nr1", "HL_1_main_nr1_3M"),          # main HL result; ck14 (1.5M) is the answer
    ("nr1_s200", "HL_2_seed200"),         # same recipe, other seeds -> reproducibility
    ("nr1_s300", "HL_3_seed300"),
    ("mechB", "HL_4_ablation_B_upsample"),   # B grew a new capability
    ("mechC", "HL_5_ablation_C_dapg"),       # C's demo anchor suppressed it
    ("lrhalf", "HL_6_probe_lr_half"),
    ("b2", "HL_7_recipe_b2"),             # the diagnostic line that produced the recipe
    ("g3", "HL_8_recipe_g3"),
    ("train36", "HL_9_recipe_train36"),
    ("curves", "LL_1_bc_and_es"),         # ll_yield_es_v3, the LL result
]

# Evaluations of the SAME run at different checkpoints are one CURVE, not four
# dots. eval_to_tb.py writes every stats file at step 0, so re-key by the training
# step the checkpoint came from -- that is what makes "peaks at 1.5 M, then
# degrades" visible instead of something you have to read off four separate cards.
# Members must share an eval set or the line is meaningless. `u<N>` is the PPO
# update index; step = N * 8192 env steps, so u122/u183/u244/u366 are exactly the
# ck9/ck14/ck19/final checkpoints -- the same four models, evaluated on two
# different sets. Only the n=108 one resolves anything: the n=6 holdout reads
# 0.8333 at all four checkpoints because 5/6 is the only value it can take.
EVAL_SERIES = [
    # BC is step 0: it IS the initialisation of this PPO run (bc_stable_hl.pth),
    # so including it is what makes the rise-then-fall visible. Without it the
    # curve starts at 1.0 M and looks purely monotone-degrading.
    ("EVAL_1_nr1_over_training_n108", [
        ("mechBC", 0),                # BC start
        ("nr1__u122", 1_000_000),     # = ck9   -- best evaluated point
        ("nr1__u183", 1_500_000),     # = ck14
        ("nr1__u244", 2_000_000),     # = ck19
        ("nr1__u366", 3_000_000),     # = final
    ]),
    ("EVAL_2_nr1_over_training_holdout_n6", [  # zero-leak, but too small to resolve
        ("nr1__ck9_holdout", 1_000_000),
        ("nr1__ck14_holdout", 1_500_000),
        ("nr1__ck19_holdout", 2_000_000),
        ("nr1__u366_holdout", 3_000_000),
    ]),
]

# Baselines the learned policy is judged against. Emitted at every step of the
# series x-range so they draw a flat line across the plot instead of a lone dot
# at step 0 -- "did we pass the teacher" has to be readable without arithmetic.
REFERENCE_LINES = [
    ("EVAL_0_ref_teacher_n108", "teacher__cert3"),
]
REFERENCE_STEPS = [0, 1_000_000, 1_500_000, 2_000_000, 3_000_000]

# One-off evaluations. The step encodes where the checkpoint sits in training so
# they land on the same x-axis as the series above; 0 means "not from a PPO run".
# Everything else under eval_summary is certification archaeology on retired
# datasets (cd17, m0805, solid6, dev20_parked, ...) -- 110 sub-runs, left in tb/.
EVAL_RUNS = [
    ("nr1__s200", "EVAL_3_seed200_n108", 1_500_000),
    ("nr1__s300", "EVAL_4_seed300_n108", 1_500_000),
    ("mechB", "EVAL_5_ablation_B_n108", 1_000_000),
    ("mechC", "EVAL_6_ablation_C_n108", 1_500_000),
    ("holdout2__teacher", "EVAL_8_ref_teacher_holdout_n6", 0),
    ("holdout2__BC", "EVAL_8_ref_BC_holdout_n6", 0),
    ("esv3__holdout2", "EVAL_9_LL_esv3_holdout_n6", 0),
    ("esv3__train36_v1", "EVAL_9_LL_esv3_train36", 0),
    ("PPO__B2final_train36", "EVAL_9_recipe_b2_train36", 0),
    ("PPO__G3final_criticlr_train36", "EVAL_9_recipe_g3_train36", 0),
]

# Exact tags and prefixes worth plotting. Everything else is dropped.
KEEP_EXACT = {
    "reward",
    "metrics/social_nav_to_pos_success",     # success rate
    "metrics/did_collide",                   # collision rate
    "metrics/num_steps",                     # episode length
    "metrics/min_agent_clearance",           # closest approach to the human
    "metrics/human_delay",                   # social cost imposed on the human
    "metrics/human_passed_door",
    "metrics/social_nav_reward",
    "metrics/social_nav_stats.yield_ratio",
    "metrics/social_nav_stats.backup_ratio",
    "learner/agent_0_action_loss",
    "learner/agent_0_value_loss",
    "learner/agent_0_value_pred_mean",       # the critic-scale diagnosis lives here
    "learner/agent_0_dist_entropy",
    "learner/agent_0_grad_norm",
    "learner/agent_0_aux_bc_anchor_loss",
}
KEEP_PREFIX = (
    "metrics/social_nav_reward_breakdown.",  # per-component reward decomposition
    "metrics/rollout_",                      # stable-success / stable-failure fractions
    "learner/agent_0_aux_dapg_",             # DAPG arm only
    "learner/agent_0_dapg_grad_",
    "eval/",                                 # post-hoc scorecard
    "reward_success/", "reward_collision/", "reward_timeout/",
    "es/", "train/", "data/action_count_",   # LL: evolution + BC curves
)


def wanted(tag):
    return tag in KEEP_EXACT or tag.startswith(KEEP_PREFIX)


def read_run(src, force_step=None):
    """-> ({tag: {step: value}}, tags_seen). force_step overrides the recorded step."""
    series, seen = {}, set()
    for f in sorted(glob.glob(os.path.join(src, "**", "events.out.tfevents.*"),
                              recursive=True)):
        try:
            acc = EA(f, size_guidance={"scalars": 0})
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            seen.add(tag)
            if not wanted(tag):
                continue
            pts = {(force_step if force_step is not None else s.step): s.value
                   for s in acc.Scalars(tag)}
            # later event files win on a repeated (tag, step): a rerun supersedes
            series.setdefault(tag, {}).update(pts)
    return series, seen


def write_run(dst, series):
    if not series:
        return 0
    os.makedirs(dst, exist_ok=True)
    n = 0
    with SummaryWriter(dst) as w:
        for tag, pts in sorted(series.items()):
            for step in sorted(pts):
                w.add_scalar(tag, pts[step], step)
                n += 1
    return n


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)          # rebuilt from tb/ every time; never edit by hand
    os.makedirs(OUT)

    jobs = [(name, [(os.path.join(TB, s), None)]) for s, name in TRAIN_RUNS]
    jobs += [(name, [(os.path.join(TB, "eval_summary", m), st) for m, st in members])
             for name, members in EVAL_SERIES]
    jobs += [(name, [(os.path.join(TB, "eval_summary", s), st)])
             for s, name, st in EVAL_RUNS]
    jobs += [(name, [(os.path.join(TB, "eval_summary", s), st)
                     for st in REFERENCE_STEPS])
             for name, s in REFERENCE_LINES]

    print("%-34s %-6s %-6s %-7s %s" % ("run", "kept", "seen", "points", "x-axis"))
    tk = ts = tp = 0
    for name, members in jobs:
        merged, seen = {}, set()
        for src, step in members:
            if not os.path.isdir(src):
                print("%-34s MISSING %s" % (name, src))
                continue
            s, sn = read_run(src, step)
            seen |= sn
            for tag, pts in s.items():
                merged.setdefault(tag, {}).update(pts)
        pts = write_run(os.path.join(OUT, name), merged)
        steps = sorted({k for v in merged.values() for k in v})
        span = "-" if not steps else (format(steps[0], ",") if len(steps) == 1
                                      else "%s..%s (%d)" % (format(steps[0], ","),
                                                            format(steps[-1], ","),
                                                            len(steps)))
        print("%-34s %-6d %-6d %-7d %s" % (name, len(merged), len(seen), pts, span))
        tk, ts, tp = tk + len(merged), ts + len(seen), tp + pts
    print("\n%d runs, %d tags kept of %d seen, %d points -> %s"
          % (len(jobs), tk, ts, tp, OUT))
    print("tensorboard --logdir hrl_pipeline/tb_clean --port 6006")


if __name__ == "__main__":
    main()
