"""One-shot PPO training-health check, in the project's diagnostic order:
critic working? -> entropy melting? -> reward-vs-success direction -> steps
diving (yield-trimming slide)? Prints ALERT lines when a pathology fires,
else OK lines. Exit code 2 = alert, 0 = healthy/insufficient data.

Usage: python check_train_health.py <tb_dir> <train_log> [warmup_updates=60]
"""
import sys

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

tb_dir, log_path = sys.argv[1], sys.argv[2]
warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 60

ea = event_accumulator.EventAccumulator(tb_dir)
ea.Reload()
tags = ea.Tags()["scalars"]


def series(frag):
    for t in tags:
        if frag in t:
            return np.array([s.value for s in ea.Scalars(t)])
    return None


alerts = []
ent = series("dist_entropy")
vpred = series("value_pred_mean")
vloss = series("value_loss")
succ = series("social_nav_to_pos_success")
rew = series("reward")
steps = series("num_steps")

n = len(ent) if ent is not None else 0
post = max(warmup, 10)
if n <= post + 20:
    print(f"OK insufficient-data updates={n} (need >{post + 20})")
    sys.exit(0)

# 1. critic: V_pred should leave zero and value_loss should shrink post-warmup
if vpred is not None and abs(np.median(vpred[post + 10 :])) < 1.0:
    alerts.append(f"CRITIC_DEAD |V_pred| median {np.median(vpred[post+10:]):.2f} < 1 after warmup")
# 2. entropy melt: sustained rise vs post-warmup baseline
if ent is not None:
    base = np.median(ent[post : post + 20])
    late = np.median(ent[-20:])
    if base > 0 and late > 3.0 * base and late > 0.05:
        alerts.append(f"ENTROPY_MELT {base:.3f} -> {late:.3f} (x{late/base:.1f})")
# 3. direction: reward up + success down = hacking; both down = optimization failure
if succ is not None and rew is not None and len(succ) > post + 40:
    s0, s1 = np.median(succ[post : post + 20]), np.median(succ[-20:])
    r0, r1 = np.median(rew[post : post + 20]), np.median(rew[-20:])
    if s1 < s0 - 0.12:
        kind = "REWARD_HACKING" if r1 > r0 + 5 else "OPTIMIZATION_FAILURE"
        alerts.append(f"{kind} succ {s0:.2f}->{s1:.2f} reward {r0:.1f}->{r1:.1f}")
# 4. steps dive = yield-trimming slide
if steps is not None and len(steps) > post + 40:
    st0, st1 = np.median(steps[post : post + 20]), np.median(steps[-20:])
    if st1 < 0.75 * st0:
        alerts.append(f"STEPS_DIVE {st0:.0f} -> {st1:.0f}")

if alerts:
    for a in alerts:
        print("ALERT", a)
    sys.exit(2)
print(
    f"OK updates={n} succ={np.median(succ[-20:]):.2f} "
    f"ent={np.median(ent[-20:]):.3f} vpred={np.median(vpred[-20:]) if vpred is not None else float('nan'):.1f} "
    f"steps={np.median(steps[-20:]):.0f}"
)
