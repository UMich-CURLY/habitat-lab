"""Build the formal-round training set: ALL non-flaky train36 episodes with
stable-failure upsampled x3 (28 stable-success + 4x3 stable-failure = 40
entries). Flaky episodes stay out of training (contradictory teacher outcomes
= noisy reward targets) and remain eval-only, per the round-1 protocol.
Stamps info.pool.train36_id + teacher_class for the trainer's rollout-mix
tracking. Output: data/social_nav_episode_train32_up3.json.gz
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
ss = sorted(int(r["train36_id"]) for r in rows if r["final_class"] == "stable-success")
sf = sorted(int(r["train36_id"]) for r in rows if r["final_class"] == "stable-failure")
ids = ss + [i for i in sf for _ in range(3)]

d = json.load(gzip.open(f"{ROOT}/data/social_nav_episode_train36_v1.json.gz"))
eps = []
for i in ids:
    e = copy.deepcopy(d["episodes"][i])
    e["info"]["pool"]["train36_id"] = i
    e["info"]["pool"]["teacher_class"] = byid[i]["final_class"]
    e["episode_id"] = str(len(eps))
    eps.append(e)
d2 = dict(d)
d2["episodes"] = eps
with gzip.open(f"{ROOT}/data/social_nav_episode_train32_up3.json.gz", "wt") as f:
    json.dump(d2, f)
scenes = {byid[i]["scene"] for i in set(ids)}
print(
    f"train32_up3: {len(eps)} entries ({len(ss)} ss + {len(sf)}x3 sf), "
    f"{len(scenes)} scenes, nominal fail frac {3 * len(sf) / len(eps):.2f}"
)
