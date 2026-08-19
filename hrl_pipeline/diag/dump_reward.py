"""ASCII reward/entropy curves from tb dirs, for chat-friendly inspection.

Usage: python dump_reward.py <tb_dir> [<tb_dir> ...]
"""
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)



def dump(tb_dir, rows=20):
    f = sorted(glob.glob(f"{tb_dir}/events.out.tfevents.*"))[-1]
    acc = EventAccumulator(f, size_guidance={"scalars": 0})
    acc.Reload()
    have = set(acc.Tags()["scalars"])
    get = lambda t: (  # noqa: E731
        {e.step: e.value for e in acc.Scalars(t)} if t in have else {}
    )
    rew = get("reward")
    snr = get("metrics/social_nav_reward")
    succ = get("metrics/social_nav_to_pos_success")
    ent = get("learner/agent_0_dist_entropy")
    steps = sorted(rew)
    lo, hi = min(rew.values()), max(rew.values())
    span = max(hi - lo, 1e-6)
    print(f"\n=== {tb_dir}: reward {lo:.0f} .. {hi:.0f} over {len(steps)} updates ===")
    print(f"{'Msteps':>6} {'reward':>8} {'sn_rew':>7} {'succ':>5} {'entropy':>7}  chart")
    for s in steps[:: max(1, len(steps) // rows)] + [steps[-1]]:
        r = rew[s]
        bar = "=" * int(40 * (r - lo) / span)
        print(
            f"{s / 1e6:6.2f} {r:8.1f} {snr.get(s, float('nan')):7.2f} "
            f"{succ.get(s, float('nan')):5.2f} {ent.get(s, float('nan')):7.3f}  |{bar}"
        )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        dump(p)
