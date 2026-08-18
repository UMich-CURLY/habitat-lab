"""How much probability mass does a trained HL policy put on `wait`?

Replays the logged decision states (hl_decisions_*.jsonl) through a saved HL
checkpoint and reports the per-action probabilities. This decides whether an
entropy bonus could ever discover `wait`: if p(wait) is ~1e-4, sampling it for
the several CONSECUTIVE decisions a yield-by-waiting needs is hopeless.

Usage: python probe_wait_prob.py <jsonl> <ckpt.pth> [<ckpt.pth> ...]
"""
import json
import sys

import numpy as np
import torch

from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.utils.common import CategoricalNet

ACTS = ["backoff", "wait", "go_to_goal"]
HIDDEN = 32
IN_DIM = 10


def episodes(path):
    rows = [json.loads(l) for l in open(path)]
    eps, cur = [], []
    for d in rows:
        if float(d["mask"]) == 0.0 and cur:
            eps.append(cur)
            cur = []
        cur.append(d)
    if cur:
        eps.append(cur)
    return eps


def feat_of(d):
    x = list(d["feat"][:6]) + [float(d.get("approach", 0.0))]
    onehot = [0.0] * 3
    pa = int(d.get("prev_action", -1))
    if 0 <= pa < 3:
        onehot[pa] = 1.0
    return x + onehot


def probe(jsonl, ckpt):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    se = build_rnn_state_encoder(IN_DIM, HIDDEN, rnn_type="GRU", num_layers=1)
    head = CategoricalNet(HIDDEN, 3)
    se.load_state_dict(d["state_encoder"])
    head.load_state_dict(d["policy"])
    se.eval()
    head.eval()

    probs = []
    with torch.no_grad():
        for ep in episodes(jsonl):
            x = torch.tensor([feat_of(r) for r in ep], dtype=torch.float32)
            out, _ = se.rnn(x.view(len(ep), 1, IN_DIM), torch.zeros(1, 1, HIDDEN))
            p = torch.softmax(head.linear(out.squeeze(1)), dim=-1)
            probs.append(p.numpy())
    p = np.concatenate(probs, 0)
    print(f"\n=== {ckpt} on {jsonl} ({len(p)} decisions) ===")
    for i, a in enumerate(ACTS):
        col = p[:, i]
        print(
            f"  p({a:11}) mean={col.mean():.6f} median={np.median(col):.6f} "
            f"max={col.max():.6f}"
        )
    pw = p[:, 1]
    print(f"  decisions with p(wait)>0.01: {(pw > 0.01).sum()} / {len(pw)}")
    for k in (1, 3, 5):
        print(
            f"  P(sampling wait {k}x in a row at the median state): "
            f"{np.median(pw) ** k:.3e}"
        )


if __name__ == "__main__":
    for ck in sys.argv[2:]:
        probe(sys.argv[1], ck)
