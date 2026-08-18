"""Classify train36 episodes by TEACHER outcome (3-eval cert + history).

stable-success : every eval on record is a clean success (success AND zero
                 collision). Attributes (do NOT affect eligibility):
                 near_cap (any eval steps >= 1150), inefficient (cert3 mean
                 steps >= 900 or pool label trivial_eff), else efficient.
                 Slow is not dirty -- these are the best material for the
                 "student beats teacher on time" check.
stable-failure : every eval on record fails.
flaky          : anything mixed (success/fail mix, success-with-collision,
                 history contradiction). Excluded from round-1 training and
                 from the mechanism verdicts; kept in the final full-36 eval.

Evidence per episode = stats_teacher_cert3.json (3 evals) union two
historical 1-eval passes of the SAME wait-teacher (stats_teacher_wait_t36 on
train36; stats_teacher_wait_bc on the bcwait subset, mapped back via the
sorted-success-id construction in _new_teacher_chain.sh).

Writes hrl_pipeline/TEACHER_CERT3.csv. The final_class column may later be
downgraded to flaky by verify_demo_jsonl.py (demo-pass failure = new
evidence of instability).
"""
import csv
import gzip
import json
import os
from collections import defaultdict

ROOT = os.environ.get("HAB", "/habitat-lab")
SD = f"{ROOT}/hrl_pipeline/results"


def load_stats(path):
    out = defaultdict(list)
    for k, v in json.load(open(path)).items():
        out[int(k.split("|")[1])].append(
            (
                float(v["social_nav_to_pos_success"]),
                float(v["did_collide"]),
                int(v["num_steps"]),
            )
        )
    return out


cert3 = load_stats(f"{SD}/stats_teacher_cert3.json")
hist_t36 = load_stats(f"{SD}/stats_teacher_wait_t36.json")
hist_bc_local = load_stats(f"{SD}/stats_teacher_wait_bc.json")
ok = sorted(i for i, evs in hist_t36.items() if evs[0][0] > 0.5)
hist_bc = {ok[i]: evs for i, evs in hist_bc_local.items()}

d = json.load(gzip.open(f"{ROOT}/data/social_nav_episode_train36_v1.json.gz"))
assert len(d["episodes"]) == 36 and len(cert3) == 36, (
    len(d["episodes"]),
    len(cert3),
)

rows = []
counts = defaultdict(int)
for i, ep in enumerate(d["episodes"]):
    pool = ep["info"]["pool"]
    scene = ep["scene_id"].split("/")[-1].split(".")[0]
    c3 = cert3[i]
    assert len(c3) == 3, (i, len(c3))
    evs = c3 + hist_t36[i] + hist_bc.get(i, [])
    if all(s > 0.5 and c < 0.5 for s, c, _ in evs):
        cls = "stable-success"
        attr = []
        if sum(st for _, _, st in c3) / 3.0 >= 900 or pool["label"] == "trivial_eff":
            attr.append("inefficient")
        if any(st >= 1150 for _, _, st in evs):
            attr.append("near_cap")
        attr = "+".join(attr) or "efficient"
    elif all(s <= 0.5 for s, _, _ in evs):
        cls, attr = "stable-failure", ""
    else:
        cls, attr = "flaky", ""
    counts[cls] += 1
    hist = f"t36:{'S' if hist_t36[i][0][0] > 0.5 else 'F'}"
    if i in hist_bc:
        hist += f",bc:{'S' if hist_bc[i][0][0] > 0.5 else 'F'}"
    rows.append(
        [
            i,
            pool["label"],
            scene,
            cls,
            attr,
            f"{sum(1 for s, _, _ in c3 if s > 0.5)}/3",
            f"{sum(1 for _, c, _ in c3 if c > 0.5)}/3",
            ";".join(str(st) for _, _, st in c3),
            hist,
        ]
    )

assert sum(counts.values()) == 36
with open(f"{SD}/TEACHER_CERT3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "train36_id",
            "label",
            "scene",
            "final_class",
            "attr",
            "cert3_succ",
            "cert3_coll",
            "cert3_steps",
            "history",
        ]
    )
    w.writerows(rows)
print("classes:", dict(counts))
for r in rows:
    if r[3] != "stable-success":
        print(f"  id{r[0]:>2} {r[3]:<14} label={r[1]:<13} cert3={r[5]} "
              f"coll={r[6]} steps={r[7]} hist={r[8]}")
print(f"wrote {SD}/TEACHER_CERT3.csv")
