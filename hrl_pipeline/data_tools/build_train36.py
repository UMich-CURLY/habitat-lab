"""Select the 36-episode main training set from pool_v1 (sweet : rest = 1:1).

Quotas: sweet 18, trivial_eff 9, trivial_zero 3, fail_B_wait 1, fail_C_dither 5.
(fail_C is 5 not 6: the only C scenes in stock are cd17-family / solid6 / e191;
capping cd17 at 2 leaves 5. The 6th slot goes to trivial_eff.)

Rules: max 2 episodes per physical scene across the WHOLE set; prefer
x3-verified sweets and largest efficiency gaps; maximize distinct scenes.
Outputs data/social_nav_episode_train36_v1.json.gz + hrl_pipeline/TRAIN36.csv,
plus a holdout report: remaining pool episodes whose scenes are UNUSED by the
training set (safe val/test candidates -- scene-level split, no leakage).
"""
import csv
import gzip
import json
import re
from collections import Counter, defaultdict

DATA = "/habitat-lab/data"
SD = "/habitat-lab/hrl_pipeline/results"

d = json.load(gzip.open(f"{DATA}/social_nav_episode_pool_v1.json.gz"))
eps = d["episodes"]


def scene(ep):
    return ep["scene_id"].split("/")[-1].split(".")[0]


def gap(ep):
    m = re.search(r"\+(\d+) steps", ep["info"]["pool"]["note"])
    return int(m.group(1)) if m else 0


def verified(ep):
    p = ep["info"]["pool"]
    return "3/3" in p["note"] or p["source"] in ("manual", "manual0805",
                                                 "overfit_ep70_v2")


scene_count = Counter()
chosen = []


def take(cands, n, cap=2):
    got = 0
    for ep in cands:
        if got >= n:
            break
        s = scene(ep)
        if scene_count[s] >= cap:
            continue
        # within one label, prefer a fresh scene over a 2nd episode of a used
        # scene while fresh ones remain
        chosen.append(ep)
        scene_count[s] += 1
        got += 1
    return got


by = defaultdict(list)
for ep in eps:
    by[ep["info"]["pool"]["label"]].append(ep)

def take_label(cands, n, prefer_fresh=True):
    """Fill n slots: fresh scenes first at cap 2, then relax to cap 3."""
    if prefer_fresh:
        fresh = [e for e in cands if scene_count[scene(e)] == 0]
        rest = [e for e in cands if scene_count[scene(e)] > 0]
        got = take(fresh, n)
        got += take(rest, n - got)
    else:
        got = take(cands, n)
    if got < n:
        left = [e for e in cands if e not in chosen]
        got += take(left, n - got, cap=3)
    return got


# --- fail pool first (scarcest, gets scene priority). NOTE: solid6's two
# fails share the cd17 family scene (103997541) -- the entire C stock spans
# only TWO physical scenes, so the family scene gets the "important scene"
# cap of 3 and fail_C tops out at 4. ---
fc = by["fail_C_dither"]
cd17 = [e for e in fc if e["info"]["pool"]["source"] == "cleandev17_v2"]
solid = [e for e in fc if e["info"]["pool"]["source"] == "bcsolid6_v2"]
e191 = [e for e in fc if e["info"]["pool"]["source"] == "manual0805"]
take(e191, 1)
take(solid, 2, cap=3)
take(cd17, 1, cap=3)
take(by["fail_B_wait"], 1)

# --- sweet 18: verified + fresh-scene first ---
sweets = sorted(by["sweet"], key=lambda e: (not verified(e), scene_count[scene(e)]))
take_label(sweets, 18)

# --- trivial_eff 10 by gap desc ---
take_label(sorted(by["trivial_eff"], key=gap, reverse=True), 10)

# --- trivial_zero 3: prefer the clicked ones + 1 parked ---
tz = by["trivial_zero"]
clicked = [e for e in tz if e["info"]["pool"]["source"] == "manual0805"]
parked = [e for e in tz if e["info"]["pool"]["source"] == "dev20_parked"]
n = take(clicked, 2)
take(parked, 3 - n)

# Final top-up: current stock can't always meet every quota under the scene
# caps (batches re-clicked the same scenes). Fill remaining slots by priority,
# relaxing the cap one notch at a time, and report the true composition.
for cap in (3, 4):
    if len(chosen) >= 36:
        break
    for lab in ("trivial_eff", "trivial_zero", "sweet"):
        left = [e for e in by[lab] if e not in chosen]
        left.sort(key=gap, reverse=True)
        take(left, 36 - len(chosen), cap=cap)
        if len(chosen) >= 36:
            break

assert len(chosen) == 36, f"selected {len(chosen)}"

out = {k: v for k, v in d.items() if k != "episodes"}
out_eps = []
for ep in chosen:
    ep = json.loads(json.dumps(ep))
    ep["info"]["pool"]["pool_v1_id"] = ep["episode_id"]
    ep["episode_id"] = str(len(out_eps))
    out_eps.append(ep)
out["episodes"] = out_eps
with gzip.open(f"{DATA}/social_nav_episode_train36_v1.json.gz", "wt") as f:
    json.dump(out, f)

used_scenes = set(scene(e) for e in chosen)
holdout = [e for e in eps
           if scene(e) not in used_scenes
           and e["info"]["pool"]["label"] != "challenge"]

with open(f"{SD}/TRAIN36.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_id", "label", "scene", "source", "src_idx", "click", "note"])
    for i, ep in enumerate(out_eps):
        p = ep["info"]["pool"]
        w.writerow([i, p["label"], scene(ep), p["source"], p["source_idx"],
                    p["click_id"], p["note"]])
    w.writerow([])
    w.writerow(["-- holdout candidates (scenes unused by train36) --"])
    for ep in holdout:
        p = ep["info"]["pool"]
        w.writerow(["val?", p["label"], scene(ep), p["source"], p["source_idx"],
                    p["click_id"], p["note"]])

print("train36:", Counter(e["info"]["pool"]["label"] for e in chosen))
print("distinct scenes:", len(used_scenes),
      "| per-scene max:", scene_count.most_common(3))
print("holdout candidates (unused scenes):",
      Counter(e["info"]["pool"]["label"] for e in holdout))
