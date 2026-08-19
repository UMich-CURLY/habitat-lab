"""Print the PPO learner curves (losses/entropy/grad-norm/clip) from a tb dir.

Usage: python dump_losses.py <tb_dir> [<tb_dir> ...]
x-axis is env steps; printed as update index (= steps / (num_envs*num_steps)).
"""
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

def _steps_per_update(steps):
    """Infer env-steps per update from the scalar step axis (it differs per
    run: 8 envs x ppo.num_steps, which we vary across experiments)."""
    return min(b - a for a, b in zip(steps[:-1], steps[1:])) if len(steps) > 1 else 1

TAGS = {
    "succ": "metrics/social_nav_to_pos_success",
    "vl": "learner/agent_0_value_loss",
    "al": "learner/agent_0_action_loss",
    "ent": "learner/agent_0_dist_entropy",
    "gn": "learner/agent_0_grad_norm",
    "clip": "learner/agent_0_ppo_fraction_clipped",
    "vp": "learner/agent_0_value_pred_mean",
    "pr": "learner/agent_0_prob_ratio_mean",
}


def dump(tb_dir, n_rows=22):
    f = sorted(glob.glob(f"{tb_dir}/events.out.tfevents.*"))[-1]
    acc = EventAccumulator(f, size_guidance={"scalars": 0})
    acc.Reload()
    have = set(acc.Tags()["scalars"])
    d = {
        k: {e.step: e.value for e in acc.Scalars(v)}
        for k, v in TAGS.items()
        if v in have
    }
    steps = sorted(d["vl"])
    spu = _steps_per_update(steps)
    print(f"\n=== {tb_dir} ({len(steps)} updates, {spu} env-steps/update) ===")
    print(
        f"{'Msteps':>6} {'succ':>5} {'value_loss':>10} {'action_loss':>11} "
        f"{'entropy':>7} {'grad_norm':>9} {'clip%':>6} {'V_pred':>7} {'ratio':>6}"
    )
    sel = steps[:: max(1, len(steps) // n_rows)] + [steps[-1]]
    for s in sel:
        g = lambda k: d.get(k, {}).get(s, float("nan"))  # noqa: E731
        print(
            f"{s / 1e6:6.2f} {g('succ'):5.2f} {g('vl'):10.2f} "
            f"{g('al'):11.4f} {g('ent'):7.3f} {g('gn'):9.3f} "
            f"{100 * g('clip'):6.1f} {g('vp'):7.1f} {g('pr'):6.3f}"
        )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        dump(p)
