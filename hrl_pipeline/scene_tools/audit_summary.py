"""Summarise an always_go + rule_yield audit pair into a per-episode verdict CSV.

Reads the EVAL_STATS jsons written by hrl_pipeline/scripted_hl_eval.py (keys
"<scene_path>|<episode_id>|<eval_idx>"), averages social_nav_to_pos_success
per episode, and classifies: SWEET = always_go fails & rule_yield succeeds,
trivial = always_go already succeeds, unsolv = both fail. Columns align with
hrl_pipeline/doorcand_v3_audit.csv plus the manual-click provenance.

Usage: python hrl_pipeline/scene_tools/audit_summary.py <dataset.json.gz> <stats_ag.json> <stats_ry.json> <out.csv>
"""
import csv
import gzip
import json
import sys

from collections import defaultdict


def per_ep_success(stats_path):
    acc = defaultdict(list)
    for k, v in json.load(open(stats_path)).items():
        scene, epid, _ = k.rsplit("|", 2)
        acc[(scene.split("/")[-1].split(".")[0], epid)].append(float(v["social_nav_to_pos_success"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def verdict(ag, ry):
    if ag <= 0.34 and ry >= 0.67:
        return "SWEET"
    if ag >= 0.67:
        return "trivial"
    if ag <= 0.34 and ry <= 0.34:
        return "unsolv"
    return "mixed"


def main():
    ds_path, ag_path, ry_path, out = sys.argv[1:5]
    ag = per_ep_success(ag_path)
    ry = per_ep_success(ry_path)
    eps = json.load(gzip.open(ds_path, "rt"))["episodes"]

    rows, counts = [], defaultdict(int)
    for i, ep in enumerate(eps):
        short = ep["scene_id"].split("/")[-1].split(".")[0]
        epid = str(ep["episode_id"])
        a = ag.get((short, epid))
        r = ry.get((short, epid))
        v = verdict(a, r) if a is not None and r is not None else "no_stats"
        counts[v] += 1
        man = ep["info"].get("manual", {})
        rows.append(dict(
            idx=i, scene=short[:14], epid=epid, verdict=v,
            doorW=round(ep["info"].get("door_width", float("nan")), 2),
            AGsu="" if a is None else round(a, 2),
            RYsu="" if r is None else round(r, 2),
            click_id=man.get("click_id", ""), note=man.get("note", ""),
            dt_arrival=man.get("dt_arrival", ""),
        ))

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
