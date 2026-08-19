"""Per-group breakdown of deterministic evals against teacher certification.

Usage: python by_group.py <stats.json>:<LABEL> [<stats.json>:<LABEL> ...]

Stats files must come from evals on the FULL train36_v1 (positional ids =
train36 ids), any evals_per_ep. The FIRST stats file is the reference policy
(normally BC) for the paired-efficiency and retention sections.

Definitions (episode level):
  success   = ALL evals succeed (a single flaky pass does not count)
  collided  = ANY eval collides
  steps     = mean num_steps over evals
Teacher reference steps = cert3 mean from TEACHER_CERT3.csv.
"""
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.environ.get("HAB", "/habitat-lab")
SD = f"{ROOT}/hrl_pipeline/results"

cert = {int(r["train36_id"]): r for r in csv.DictReader(open(f"{SD}/TEACHER_CERT3.csv"))}
CLASSES = ["stable-success", "stable-failure", "flaky"]


def load(path):
    per = defaultdict(list)
    for k, v in json.load(open(path)).items():
        per[int(k.split("|")[1])].append(
            (
                float(v["social_nav_to_pos_success"]),
                float(v["did_collide"]),
                float(v["num_steps"]),
            )
        )
    return {
        i: {
            "success": all(s > 0.5 for s, _, _ in evs),
            "collided": any(c > 0.5 for _, c, _ in evs),
            "steps": sum(st for _, _, st in evs) / len(evs),
            "n_evals": len(evs),
        }
        for i, evs in per.items()
    }


policies = []
for arg in sys.argv[1:]:
    path, _, label = arg.rpartition(":")
    if not path:
        path, label = label, os.path.basename(label)
    policies.append((label, load(path)))
ref_label, ref = policies[0]
teacher_steps = {
    i: sum(int(x) for x in r["cert3_steps"].split(";")) / 3.0
    for i, r in cert.items()
}

# ---- group table ----
print("== group table (success = all evals pass; coll = any eval collides) ==")
hdr = f"{'policy':<10}" + "".join(
    f"{c + ' n/succ/coll/steps':>34}" for c in CLASSES
)
print(hdr)
for label, pol in policies:
    cells = []
    for c in CLASSES:
        ids = [i for i in sorted(cert) if cert[i]["final_class"] == c]
        n = len(ids)
        su = sum(1 for i in ids if pol[i]["success"])
        co = sum(1 for i in ids if pol[i]["collided"])
        stl = [pol[i]["steps"] for i in ids if pol[i]["success"]]
        st = f"{sum(stl) / len(stl):.0f}" if stl else "-"
        cells.append(f"{n}/{su}/{co}/{st}".rjust(34))
    print(f"{label:<10}" + "".join(cells))

# ---- paired efficiency on stable-success ----
ss_ids = [i for i in sorted(cert) if cert[i]["final_class"] == "stable-success"]
print("\n== paired steps on stable-success (teacher = cert3 mean; '-' = not all evals passed) ==")
print(
    f"{'id':>3} {'label':<13} {'attr':<22} {'teacher':>8}"
    + "".join(f"{lab:>9}" for lab, _ in policies)
)
for i in ss_ids:
    row = f"{i:>3} {cert[i]['label']:<13} {cert[i]['attr']:<22} {teacher_steps[i]:>8.0f}"
    for _, pol in policies:
        row += (
            f"{pol[i]['steps']:>9.0f}" if pol[i]["success"] else f"{'-':>9}"
        )
    print(row)

print(f"\n== vs reference policy '{ref_label}' ==")
for label, pol in policies[1:]:
    both = [i for i in ss_ids if ref[i]["success"] and pol[i]["success"]]
    lost = [i for i in ss_ids if ref[i]["success"] and not pol[i]["success"]]
    newc = [
        i for i in ss_ids if not ref[i]["collided"] and pol[i]["collided"]
    ]
    if both:
        d_ref = sum(pol[i]["steps"] - ref[i]["steps"] for i in both) / len(both)
        d_tea = sum(pol[i]["steps"] - teacher_steps[i] for i in both) / len(both)
    print(
        f"{label}: paired n={len(both)}  "
        + (
            f"dSteps(vs {ref_label})={d_ref:+.0f}  dSteps(vs teacher)={d_tea:+.0f}  "
            if both
            else "no paired successes  "
        )
        + f"retention_loss={len(lost)} {lost}  new_collisions={len(newc)} {newc}"
    )
    for attr_key in ("inefficient", "near_cap"):
        sub = [i for i in both if attr_key in cert[i]["attr"]]
        if sub:
            d = sum(pol[i]["steps"] - ref[i]["steps"] for i in sub) / len(sub)
            dt = sum(pol[i]["steps"] - teacher_steps[i] for i in sub) / len(sub)
            print(
                f"    [{attr_key} n={len(sub)}] dSteps(vs {ref_label})={d:+.0f} "
                f"dSteps(vs teacher)={dt:+.0f}"
            )

# ---- capability on stable-failure ----
sf_ids = [i for i in sorted(cert) if cert[i]["final_class"] == "stable-failure"]
print(f"\n== capability on stable-failure (teacher 0/{len(sf_ids)}) ==")
for label, pol in policies:
    won = [i for i in sf_ids if pol[i]["success"]]
    scenes = {cert[i]["scene"] for i in won}
    print(f"{label}: new stable successes {len(won)}/{len(sf_ids)} {won}  "
          f"distinct scenes {len(scenes)}")
