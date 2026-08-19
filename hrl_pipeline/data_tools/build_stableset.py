"""Build the stable-success demo-collection dataset from TEACHER_CERT3.csv.

Output: data/social_nav_episode_stableok.json.gz -- exactly the episodes the
teacher passes cleanly and repeatedly. Positional order = ascending
train36_id, which verify_demo_jsonl.py relies on to map demo segments back.
"""
import copy
import csv
import gzip
import json
import os

ROOT = os.environ.get("HAB", "/habitat-lab")
SD = f"{ROOT}/hrl_pipeline/results"

rows = list(csv.DictReader(open(f"{SD}/TEACHER_CERT3.csv")))
ok = sorted(
    int(r["train36_id"]) for r in rows if r["final_class"] == "stable-success"
)
d = json.load(gzip.open(f"{ROOT}/data/social_nav_episode_train36_v1.json.gz"))
eps = []
for i in ok:
    e = copy.deepcopy(d["episodes"][i])
    e["info"]["pool"]["train36_id"] = i
    e["info"]["pool"]["teacher_class"] = "stable-success"
    e["episode_id"] = str(len(eps))
    eps.append(e)
d2 = dict(d)
d2["episodes"] = eps
with gzip.open(f"{ROOT}/data/social_nav_episode_stableok.json.gz", "wt") as f:
    json.dump(d2, f)
print(f"stableok: {len(eps)} episodes, train36 ids {ok}")
