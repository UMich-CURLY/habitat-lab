"""Summarize eval results against the EVAL dataset's own self-describing split.

Usage: python eval_summary.py <label>=<train36_stats.json>[,<holdout_stats.json>] ...

Reads data/social_nav_episode_EVAL.json.gz for the classification (no external
CSV), maps train36 stats positions 0..35 and holdout stats positions 0..1 onto
EVAL positions 0..35 and 36..37, and reports per class / per attribute:
  n, episodes passing EVERY eval, episodes with any human collision, mean steps
  over the passing ones, plus paired step deltas against the first policy.
"""
import gzip
import json
import os
import sys

ROOT = os.environ.get("HAB", "/habitat-lab")
EV = json.load(gzip.open(f"{ROOT}/data/social_nav_episode_EVAL.json.gz"))["episodes"]
SPLIT = [e["info"]["split"] for e in EV]


def load(spec):
    """<main.json>[,<holdout.json>] -> {eval_position: aggregated outcome}"""
    parts = spec.split(",")
    out = {}
    for k, path in enumerate(parts):
        if not path:
            continue
        per = {}
        for key, v in json.load(open(path)).items():
            per.setdefault(int(key.split("|")[1]), []).append(v)
        offset = 0 if k == 0 else 36
        for pos, evs in per.items():
            out[pos + offset] = {
                "pass": all(e["social_nav_to_pos_success"] > 0.5 for e in evs),
                "coll": any(e["did_collide"] > 0.5 for e in evs),
                "steps": sum(e["num_steps"] for e in evs) / len(evs),
                "n": len(evs),
            }
    return out


pols = []
for arg in sys.argv[1:]:
    label, _, spec = arg.partition("=")
    pols.append((label, load(spec)))

GROUPS = [
    ("stable-success", lambda s: s["teacher_class"] == "stable-success"),
    ("  ├ efficient", lambda s: s["teacher_class"] == "stable-success" and s["attr"] == "efficient"),
    ("  ├ inefficient", lambda s: s["teacher_class"] == "stable-success" and s["attr"] == "inefficient"),
    ("  └ near_cap", lambda s: s["teacher_class"] == "stable-success" and "near_cap" in s["attr"]),
    ("stable-failure", lambda s: s["teacher_class"] == "stable-failure"),
    ("flaky", lambda s: s["teacher_class"] == "flaky"),
    ("holdout", lambda s: s["teacher_class"] == "holdout"),
]

hdr = f"{'group':<18}{'n':>4}" + "".join(f"{lab:>22}" for lab, _ in pols)
print(hdr)
print("-" * len(hdr))
for gname, pred in GROUPS:
    ids = [i for i, s in enumerate(SPLIT) if pred(s)]
    row = f"{gname:<18}{len(ids):>4}"
    for _, pol in pols:
        have = [i for i in ids if i in pol]
        if not have:
            row += f"{'-':>22}"
            continue
        p = sum(1 for i in have if pol[i]["pass"])
        c = sum(1 for i in have if pol[i]["coll"])
        st = [pol[i]["steps"] for i in have if pol[i]["pass"]]
        row += f"{f'{p}/{len(have)} coll{c} {int(sum(st)/len(st)) if st else 0}步':>22}"
    print(row)

ref_label, ref = pols[0]
print(f"\n配对步数比较(只在双方都全过的关卡上,基准 = {ref_label}):")
for label, pol in pols[1:]:
    for gname, pred in GROUPS[:1] + GROUPS[4:]:
        ids = [i for i, s in enumerate(SPLIT) if pred(s)
               and i in ref and i in pol and ref[i]["pass"] and pol[i]["pass"]]
        if not ids:
            continue
        d = sum(pol[i]["steps"] - ref[i]["steps"] for i in ids) / len(ids)
        print(f"  {label:<10} {gname:<16} n={len(ids):>2}  Δ{d:+.0f} 步")
    lost = [SPLIT[i]["train36_id"] for i, s in enumerate(SPLIT)
            if s["teacher_class"] == "stable-success" and i in ref and i in pol
            and ref[i]["pass"] and not pol[i]["pass"]]
    won = [SPLIT[i]["train36_id"] for i, s in enumerate(SPLIT)
           if s["teacher_class"] == "stable-failure" and i in pol and pol[i]["pass"]]
    print(f"  {label:<10} 丢掉的稳定成功: {lost or '无'} | 攻克的稳定失败: {won or '无'}")
