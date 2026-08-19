"""Summarize DAPG probe + rollout-mix curves from a tensorboard dir.

Usage: python _probe_summary.py <tb_dir>
"""
import sys

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(sys.argv[1])
ea.Reload()
TAGS = [
    "learner/agent_0_dapg_grad_ratio",
    "learner/agent_0_dapg_grad_cos",
    "learner/agent_0_aux_dapg_ce",
    "learner/agent_0_aux_dapg_acc",
    "learner/agent_0_aux_dapg_coef",
    "metrics/rollout_fail_frac",
]
for tag in TAGS:
    try:
        v = np.array([s.value for s in ea.Scalars(tag)])
    except KeyError:
        print(f"{tag}: MISSING")
        continue
    n = len(v)
    thirds = [v[: n // 3], v[n // 3 : 2 * n // 3], v[2 * n // 3 :]]
    med = [f"{np.median(t):.3f}" for t in thirds]
    frac_neg = float((v < 0).mean())
    print(
        f"{tag}: n={n} median_by_third={med} "
        f"min={v.min():.3f} max={v.max():.3f} frac_neg={frac_neg:.2f}"
    )
