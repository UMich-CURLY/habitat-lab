"""Episode-level provenance / leakage check for the TRAIN and EVAL splits.

Matching by dataset FILENAME is meaningless once splits are re-assembled from
older sets: the same episode can reappear under a new name. This fingerprints
each episode by its geometry (scene + robot start/goal + human start/goal) and
reports, for every episode of TRAIN and EVAL, which other datasets contain the
identical episode -- so "was this ever trained on?" can be answered from
content, not from a filename.

Also reports scene-level overlap between TRAIN and EVAL (a shared scene means
geometry leakage even when the episodes differ).

Usage: python hrl_pipeline/dataset_provenance.py
"""
import glob
import gzip
import json
import os
from collections import defaultdict

DATA = "/habitat-lab/data"


def rnd(v, p=2):
    return tuple(round(float(x), p) for x in v)


def fingerprint(ep):
    info = ep.get("info", {}) or {}
    return (
        ep["scene_id"].split("/")[-1],
        rnd(ep.get("start_position", [])),
        rnd(info.get("robot_goal", [])),
        rnd(info.get("human_start", [])),
        rnd(info.get("human_goal", [])),
    )


def scene(ep):
    return ep["scene_id"].split("/")[-1].split(".")[0]


def load(path):
    try:
        return json.load(gzip.open(path))["episodes"]
    except Exception:
        return []


def main():
    sets = {}
    for p in sorted(glob.glob(f"{DATA}/social_nav_episode_*.json.gz")):
        name = os.path.basename(p)[len("social_nav_episode_"):-len(".json.gz")]
        eps = load(p)
        if eps:
            sets[name] = eps

    fp_index = defaultdict(set)  # fingerprint -> {dataset names}
    for name, eps in sets.items():
        for e in eps:
            fp_index[fingerprint(e)].add(name)

    for split in ("TRAIN", "EVAL"):
        eps = sets.get(split, [])
        if not eps:
            print(f"{split}: MISSING")
            continue
        print(f"\n=== {split}: {len(eps)} episodes ===")
        origin = defaultdict(int)
        for e in eps:
            others = fp_index[fingerprint(e)] - {split}
            key = ",".join(sorted(others)) if others else "(unique to this split)"
            origin[key] += 1
        for k, n in sorted(origin.items(), key=lambda x: -x[1]):
            print(f"  {n:3d} eps also in: {k}")

    tr, ev = sets.get("TRAIN", []), sets.get("EVAL", [])
    if tr and ev:
        s_tr = {scene(e) for e in tr}
        s_ev = {scene(e) for e in ev}
        f_tr = {fingerprint(e) for e in tr}
        f_ev = {fingerprint(e) for e in ev}
        print("\n=== TRAIN vs EVAL ===")
        print(f"  episodes identical in both : {len(f_tr & f_ev)}")
        print(f"  scenes  TRAIN={len(s_tr)}  EVAL={len(s_ev)}  shared={len(s_tr & s_ev)}")
        if s_tr & s_ev:
            print("  shared scenes (geometry leakage across the split):")
            for s in sorted(s_tr & s_ev):
                print(f"    {s:32} TRAIN x{sum(1 for e in tr if scene(e)==s)}"
                      f"  EVAL x{sum(1 for e in ev if scene(e)==s)}")


if __name__ == "__main__":
    main()
