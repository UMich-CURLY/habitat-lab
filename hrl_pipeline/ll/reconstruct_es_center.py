"""Rebuild the ES center that es_ll_finetune.py computes and then discards.

The driver saves best_vec (the single best-ever candidate) and never saves or
evaluates the final center, even though the population-level statistics show
the center is where the actual learning accumulated. Everything needed to
replay the center trajectory survives in the run's workdir: each candidate's
weight vector (g{g}_c{i}.pth) and its fitness (stats_g{g}c{i}.json).

Usage: python reconstruct_es_center.py <workdir> <base.pth> <out.pth> [topk] [step_cap]
"""
import json
import os
import sys

import numpy as np
import torch

workdir, base_path, out_path = sys.argv[1:4]
topk = int(sys.argv[4]) if len(sys.argv) > 4 else 4
step_cap = float(sys.argv[5]) if len(sys.argv) > 5 else 1200.0
LAST_KEYS = ("net.4.weight", "net.4.bias")

base = torch.load(base_path, map_location="cpu")
sd = base["model"]
keys = [k for k in LAST_KEYS if k in sd]


def flat(d):
    return torch.cat([d[k].flatten() for k in keys])


def unflat(vec):
    out = {k: v.clone() for k, v in sd.items()}
    i = 0
    for k in keys:
        n = sd[k].numel()
        out[k] = vec[i:i + n].view_as(sd[k])
        i += n
    return out


def fitness(stats_path):
    d = json.load(open(stats_path))
    su = np.mean([v["social_nav_to_pos_success"] for v in d.values()])
    co = np.mean([v["did_collide"] for v in d.values()])
    st = np.mean([v["num_steps"] for v in d.values()]) / step_cap
    return float(su - 0.5 * co - 0.1 * st)


center = flat(sd)
gens = sorted({int(f.split("_")[0][1:]) for f in os.listdir(workdir)
               if f.startswith("g") and f.endswith(".pth")})
for g in gens:
    cands = []
    for i in range(64):
        p = os.path.join(workdir, f"g{g}_c{i}.pth")
        s = os.path.join(workdir, f"stats_g{g}c{i}.json")
        if not (os.path.exists(p) and os.path.exists(s)):
            continue
        cands.append((fitness(s), i, flat(torch.load(p, map_location="cpu")["model"])))
    if not cands:
        continue
    cands.sort(key=lambda t: -t[0])
    top = cands[:topk]
    drift = float((torch.stack([c[2] for c in top]).mean(0) - center).norm())
    center = torch.stack([c[2] for c in top]).mean(0)
    print(f"  gen{g:>2}: {len(cands)} cands, top{topk} fitness "
          f"{[round(c[0], 4) for c in top]}, center moved {drift:.4f}")

torch.save({"model": unflat(center), "in_dim": base.get("in_dim", 22)}, out_path)
print(f"reconstructed final center -> {out_path}")
