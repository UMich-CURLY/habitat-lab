"""Filter LL demo rows to clean-success (episode, eval) segments only.

Usage: python verify_ll_demos.py <ll_demos.jsonl> <eval_stats.json> <out.jsonl>

Rows carry the cumulative HL step counter "t"; the stats json (insertion
order = evaluator visit order) gives per-(episode,eval) num_steps, whose
cumsum partitions the t axis. A row belongs to segment k iff
cum[k-1] < t <= cum[k]. Rows from any eval that was not a clean success
(success AND zero collision) are dropped -- BC learns from successes only.
"""
import json
import os
import sys

import numpy as np

jsonl, stats_path, out_path = sys.argv[1:4]

stats = list(json.load(open(stats_path)).items())
lens = [int(v["num_steps"]) for _, v in stats]
clean = [
    v["social_nav_to_pos_success"] > 0.5 and v["did_collide"] < 0.5
    for _, v in stats
]
cum = np.concatenate([[0], np.cumsum(lens)])

rows = [json.loads(l) for l in open(jsonl)]
assert all("t" in r and r["t"] >= 0 for r in rows), "rows lack the t field"
tmax = max(r["t"] for r in rows)
assert tmax <= cum[-1], (tmax, cum[-1])

kept, dropped, per_seg = [], 0, {}
for r in rows:
    k = int(np.searchsorted(cum, r["t"], side="left")) - 1
    k = min(max(k, 0), len(stats) - 1)
    per_seg[k] = per_seg.get(k, 0) + 1
    if clean[k]:
        kept.append(r)
    else:
        dropped += 1

with open(out_path, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")

n_dirty = sum(1 for c in clean if not c)
acts = np.array([r["act"] for r in kept])
zero = float((np.abs(acts).max(axis=1) < 1e-6).mean())
print(
    f"segments: {len(stats)} total, {n_dirty} not-clean -> dropped {dropped} rows; "
    f"kept {len(kept)}/{len(rows)}"
)
print(
    f"actions: lin mean={acts[:,0].mean():+.3f} std={acts[:,0].std():.3f}  "
    f"ang mean={acts[:,1].mean():+.3f} std={acts[:,1].std():.3f}  "
    f"zero-action share={zero:.2f}"
)
segs_with_rows = len(per_seg)
print(f"segments with LL rows: {segs_with_rows} "
      f"(episodes where the HL never yielded contribute none)")
print("wrote", out_path)
