"""Verify the stable-success demo pass; downgrade demo-failing episodes.

Usage: python verify_demo_jsonl.py <demos.jsonl> <demo_stats.json> <out.jsonl>

Splits the jsonl into episodes (mask==0 rows), aligns segment order with the
paired EVAL_STATS json (key insertion order = evaluator visit order,
cross-checked via cumulative num_steps deltas), and keeps only segments whose
eval was a clean success. A failure on the demo pass is NEW evidence: that
episode is downgraded to flaky in TEACHER_CERT3.csv, so it leaves BC/DAPG
*and* the stable-success eval group together -- never "still called
stable-success in eval but silently missing from DAPG".
"""
import csv
import json
import os
import sys

ROOT = os.environ.get("HAB", "/habitat-lab")
SD = f"{ROOT}/hrl_pipeline/results"
jsonl, stats_path, out_path = sys.argv[1:4]

rows = list(csv.DictReader(open(f"{SD}/TEACHER_CERT3.csv")))
ok = sorted(
    int(r["train36_id"]) for r in rows if r["final_class"] == "stable-success"
)

segs, cur = [], []
for ln in open(jsonl).read().splitlines():
    d = json.loads(ln)
    if float(d["mask"]) == 0.0 and cur:
        segs.append(cur)
        cur = []
    cur.append((ln, d))
if cur:
    segs.append(cur)

stats = list(json.load(open(stats_path)).items())  # insertion order
assert len(segs) == len(stats) == len(ok), (len(segs), len(stats), len(ok))
for i in range(len(segs) - 1):
    delta = segs[i + 1][0][1]["num_steps"] - segs[i][0][1]["num_steps"]
    assert delta == int(stats[i][1]["num_steps"]), (
        f"segment/stats misalignment at {i}: {delta} != {stats[i][1]['num_steps']}"
    )

bad, keep = [], []
for seg, (k, v) in zip(segs, stats):
    tid = ok[int(k.split("|")[1])]
    if v["social_nav_to_pos_success"] > 0.5 and v["did_collide"] < 0.5:
        keep += [ln for ln, _ in seg]
    else:
        bad.append(tid)

if bad:
    for r in rows:
        if int(r["train36_id"]) in bad:
            r["final_class"] = "flaky"
            r["attr"] = ""
            r["history"] += ";demo_pass:F"
    with open(f"{SD}/TEACHER_CERT3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
with open(out_path, "w") as f:
    f.write("\n".join(keep) + "\n")
print(f"demo pass: {len(segs)} episodes, downgraded to flaky: {bad or 'none'}")
print(f"kept {len(keep)} decisions ({len(segs) - len(bad)} episodes) -> {out_path}")
