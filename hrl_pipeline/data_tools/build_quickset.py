"""Build the mechanism-check subsets from TEACHER_CERT3.csv.

quick<N>     : all stable-failure + 6 sweet + 3 trivial_eff + 2 trivial_zero
               (stable-success only, hardest-first = cert3 mean steps desc,
               fresh scenes first within each label). Flaky excluded.
quick<N>_up3 : same entries + each stable-failure episode duplicated x3
               (arms B/C). Dataset-level duplication; the ACTUAL rollout mix
               is measured by the trainer (metrics/rollout_fail_frac) because
               the per-env scene split can dilute it.
Stamps info.pool.train36_id + teacher_class (read by the trainer tracking and
by_group.py). Writes QUICK<N>.csv. Asserts >= 8 distinct scenes so the
num_environments=8 scene split leaves no env empty.
"""
import copy
import csv
import gzip
import json
import os

ROOT = os.environ.get("HAB", "/habitat-lab")
SD = f"{ROOT}/hrl_pipeline/results"

rows = list(csv.DictReader(open(f"{SD}/TEACHER_CERT3.csv")))
byid = {int(r["train36_id"]): r for r in rows}


def steps3(r):
    return sum(int(x) for x in r["cert3_steps"].split(";")) / 3.0


fail = [r for r in rows if r["final_class"] == "stable-failure"]
ss = [r for r in rows if r["final_class"] == "stable-success"]


def pick(label, n):
    cands = sorted(
        [r for r in ss if r["label"] == label], key=steps3, reverse=True
    )
    got, scenes = [], set()
    for r in cands:  # fresh scenes first, hardest first
        if len(got) >= n:
            break
        if r["scene"] not in scenes:
            got.append(r)
            scenes.add(r["scene"])
    for r in cands:
        if len(got) >= n:
            break
        if r not in got:
            got.append(r)
    return got

sel = fail + pick("sweet", 6) + pick("trivial_eff", 3) + pick("trivial_zero", 2)
ids = [int(r["train36_id"]) for r in sel]
assert len(ids) == len(set(ids))
scenes = {r["scene"] for r in sel}
assert len(scenes) >= 8, f"only {len(scenes)} distinct scenes"

d = json.load(gzip.open(f"{ROOT}/data/social_nav_episode_train36_v1.json.gz"))


def build(stem, id_list):
    eps = []
    for i in id_list:
        e = copy.deepcopy(d["episodes"][i])
        e["info"]["pool"]["train36_id"] = i
        e["info"]["pool"]["teacher_class"] = byid[i]["final_class"]
        e["episode_id"] = str(len(eps))
        eps.append(e)
    d2 = dict(d)
    d2["episodes"] = eps
    with gzip.open(f"{ROOT}/data/social_nav_episode_{stem}.json.gz", "wt") as f:
        json.dump(d2, f)
    print(f"{stem}: {len(eps)} entries")


n = len(sel)
build(f"quick{n}", ids)
up3 = [
    i
    for i in ids
    for _ in range(3 if byid[i]["final_class"] == "stable-failure" else 1)
]
build(f"quick{n}_up3", up3)

with open(f"{SD}/QUICK{n}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train36_id", "label", "scene", "class", "attr", "cert3_steps"])
    for r in sel:
        w.writerow(
            [r["train36_id"], r["label"], r["scene"], r["final_class"],
             r["attr"], r["cert3_steps"]]
        )
print(
    f"quick{n}: {len(fail)} stable-failure + {n - len(fail)} stable-success, "
    f"{len(scenes)} scenes | up3 nominal fail frac "
    f"{3 * len(fail) / len(up3):.2f} (actual measured at train time)"
)
