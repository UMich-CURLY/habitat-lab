"""Slice episodes out of a dataset by index, preserving info/pool provenance.

Usage: python make_subset.py <src_stem> <out_stem> <idx> [<idx> ...]
  e.g. python make_subset.py train36_v1 t36_579 5 7 9
"""
import copy
import gzip
import json
import sys

DATA = "/habitat-lab/data"

src, out = sys.argv[1], sys.argv[2]
idxs = [int(x) for x in sys.argv[3:]]

d = json.load(gzip.open(f"{DATA}/social_nav_episode_{src}.json.gz"))
eps = []
for i in idxs:
    e = copy.deepcopy(d["episodes"][i])
    p = e.get("info", {}).get("pool", {})
    scene = e["scene_id"].split("/")[-1].split(".")[0]
    print(
        f"  {src} ep{i} -> subset ep{len(eps)}: label={p.get('label','?')} "
        f"scene={scene} click={p.get('click_id','')}"
    )
    e["episode_id"] = str(len(eps))
    eps.append(e)
d2 = dict(d)
d2["episodes"] = eps
with gzip.open(f"{DATA}/social_nav_episode_{out}.json.gz", "wt") as f:
    json.dump(d2, f)
print(f"wrote {len(eps)} episodes -> social_nav_episode_{out}.json.gz")
