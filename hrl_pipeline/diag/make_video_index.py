"""Index + rename the rendered expert videos so the filename says what it is.

The evaluator writes `episode=<id>_<n>-ckpt=0-social_nav_reward=X.mp4`, which
says nothing about the episode's pool label. This renames each file to
    <label>__id<train_id>__<scene-or-click>__succ<0|1>_coll<0|1>_<steps>st.mp4
and writes INDEX.md / index.csv (both names kept for traceability).

Usage: python make_video_index.py <video_dir> <dataset.json.gz> <stats.json> <decisions.jsonl>
"""
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter


def main(vdir, ds_path, stats_path, jsonl_path):
    eps = json.load(gzip.open(ds_path))["episodes"]
    meta = {e["episode_id"]: e for e in eps}

    stats = {}
    if os.path.exists(stats_path):
        for k, v in json.load(open(stats_path)).items():
            stats.setdefault(k.split("|")[1], []).append(v)

    # teacher decisions per episode (segments split on mask==0, in run order)
    decisions = {}
    if os.path.exists(jsonl_path):
        rows = [json.loads(l) for l in open(jsonl_path)]
        segs, cur = [], []
        for r in rows:
            if r["mask"] == 0.0 and cur:
                segs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            segs.append(cur)
        # match segments to episodes by duration against stats num_steps
        used = set()
        for seg in segs:
            dur = seg[-1]["num_steps"] - seg[0]["num_steps"]
            best, bestd = None, 1e9
            for e, vs in stats.items():
                if e in used:
                    continue
                d = abs(int(vs[0]["num_steps"]) - dur)
                if d < bestd:
                    best, bestd = e, d
            if best is not None and bestd <= 60:
                used.add(best)
                acts = Counter(r["action"] for r in seg)
                decisions[best] = {
                    "n_decisions": len(seg),
                    "backoff": acts.get("backoff", 0),
                    "wait": acts.get("wait", 0),
                    "go": acts.get("go_to_goal", 0),
                    "switches": sum(
                        1 for i in range(1, len(seg))
                        if seg[i]["action"] != seg[i - 1]["action"]
                    ),
                    "branch": Counter(r["yield_branch"] for r in seg).most_common(1)[0][0],
                }

    vids = {}
    for f in sorted(os.listdir(vdir)):
        if not f.endswith(".mp4"):
            continue
        m = re.search(r"episode=(\d+)", f)
        if m:
            vids[m.group(1)] = f

    SHORT = {"fail_B_wait": "failB-wait", "fail_C_dither": "failC-dither",
             "trivial_eff": "trivEFF", "trivial_zero": "trivZERO", "sweet": "SWEET"}

    rows_out = []
    for e in sorted(meta, key=int):
        p = meta[e]["info"]["pool"]
        st = stats.get(e, [{}])[0]
        de = decisions.get(e, {})
        src = vids.get(e)
        newname = "MISSING"
        if src:
            tag = p.get("click_id") or (
                meta[e]["scene_id"].split("/")[-1].split(".")[0][:18]
                + f"-src{p['source_idx']}"
            )
            newname = (
                f"{SHORT.get(p['label'], p['label'])}__id{int(e):02d}__{tag}"
                f"__succ{int(st.get('social_nav_to_pos_success', -1))}"
                f"_coll{int(st.get('did_collide', -1))}"
                f"_{int(st.get('num_steps', -1))}st.mp4"
            )
            if newname != src:
                os.replace(os.path.join(vdir, src), os.path.join(vdir, newname))
        rows_out.append({
            "train_id": e,
            "video": newname,
            "orig_video": src or "MISSING",
            "label": p["label"],
            "scene": meta[e]["scene_id"].split("/")[-1].split(".")[0],
            "click_id": p.get("click_id", ""),
            "source": p["source"],
            "success": int(st.get("social_nav_to_pos_success", -1)),
            "collide": int(st.get("did_collide", -1)),
            "steps": int(st.get("num_steps", -1)),
            "hl_decisions": de.get("n_decisions", ""),
            "backoff": de.get("backoff", ""),
            "go": de.get("go", ""),
            "switches": de.get("switches", ""),
            "yield_branch": de.get("branch", ""),
            "note": p.get("note", ""),
        })

    with open(os.path.join(vdir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    lab_order = ["sweet", "trivial_eff", "trivial_zero", "fail_B_wait", "fail_C_dither"]
    with open(os.path.join(vdir, "INDEX.md"), "w") as f:
        f.write("# Expert (rule_markov) rollouts on train36_v1\n\n")
        n_ok = sum(1 for r in rows_out if r["success"] == 1)
        n_col = sum(1 for r in rows_out if r["collide"] == 1)
        f.write(f"{len(rows_out)} episodes | teacher success {n_ok}/{len(rows_out)} "
                f"| collisions {n_col} | videos rendered "
                f"{sum(1 for r in rows_out if r['video'] != 'MISSING')}\n\n")
        f.write("`switches` = how many times the HL changed skill in that episode; "
                "`yield_branch` 0 = a real retreat pocket was found, 1 = hold in place.\n\n")
        for lab in lab_order:
            sel = [r for r in rows_out if r["label"] == lab]
            if not sel:
                continue
            f.write(f"\n## {lab} ({len(sel)})\n\n")
            f.write("| id | video | scene | click | succ | coll | steps | HL dec | backoff | go | switch | branch |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in sel:
                f.write(f"| {r['train_id']} | {r['video']} | {r['scene']} | {r['click_id']} "
                        f"| {r['success']} | {r['collide']} | {r['steps']} | {r['hl_decisions']} "
                        f"| {r['backoff']} | {r['go']} | {r['switches']} | {r['yield_branch']} |\n")
    print(f"wrote {vdir}/INDEX.md and index.csv ({len(rows_out)} rows, "
          f"{sum(1 for r in rows_out if r['video'] != 'MISSING')} videos)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
