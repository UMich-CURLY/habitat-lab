"""Build a manual episode dataset from a clicks.json (hrl_pipeline/scene_tools/make_click_page.py).

Per entry: snap the 4 clicked XZ points onto the furniture-aware navmesh,
validate the geometry (hard errors kill the entry, warnings are recorded),
deep-copy the source template episode and overwrite start/goals -- same
assembly as gen_v3_datasets.transform. Successful entries get episode_id
0..N-1 in clicks order (file order = audit stats internal id). Output is a
pure overwrite (idempotent); per-entry review top-downs let you eyeball the
placements before running the audit.

Usage: python hrl_pipeline/scene_tools/build_manual_episodes.py <clicks.json> <out.json.gz>
           [--review-dir hrl_pipeline/clicking/review]
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import argparse
import gzip
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gen_v3_datasets as g3
from make_click_page import bake_navmesh_outline
from render_scene_rgb_topdown import make_sim, render_at

PTS = ["robot_start", "robot_goal", "human_start", "human_goal"]
SNAP_MAX = 0.3
HALF = 6.0  # review-render half-extent; match make_click_page so deep human_goal shows


def build_one(entry, template, pf):
    errors, warnings = [], []
    ds, de, mid, _ = g3._door_frame(entry)
    mid_s = np.array(pf.snap_point(mid), float)
    if not np.all(np.isfinite(mid_s)):
        return None, ["door mid not on navmesh"], [], {}
    y = mid_s[1]

    p = {}
    for k in PTS:
        cand = np.array([entry[k][0], y, entry[k][1]])
        s = np.array(pf.snap_point(cand), float)
        if not np.all(np.isfinite(s)) or not pf.is_navigable(s):
            errors.append(f"{k}: not navigable")
            continue
        shift = float(np.linalg.norm((s - cand)[[0, 2]]))
        if shift > SNAP_MAX:
            errors.append(f"{k}: snap moved {shift:.2f}m (>={SNAP_MAX})")
        p[k] = s
    if errors:
        return None, errors, warnings, {}

    side_r = g3._side(ds, de, p["robot_start"])
    if g3._side(ds, de, p["human_start"]) == side_r:
        warnings.append("human_start on robot's side")
    if g3._side(ds, de, p["robot_goal"]) == side_r:
        warnings.append("robot_goal on robot_start side (no crossing?)")
    if g3._side(ds, de, p["human_goal"]) != side_r:
        warnings.append("human_goal not on robot_start side (no cross-through?)")

    g_rr = g3._geo(pf, p["robot_start"], p["robot_goal"])
    via = g3._geo(pf, p["robot_start"], mid_s) + g3._geo(pf, mid_s, p["robot_goal"])
    if not np.isfinite(g_rr):
        errors.append("robot start->goal unreachable")
    elif g_rr > via + 1.0:
        errors.append(f"robot path does not cross door (geo {g_rr:.1f} vs via {via:.1f})")
    g_hh = g3._geo(pf, p["human_start"], p["human_goal"])
    hvia = g3._geo(pf, p["human_start"], mid_s) + g3._geo(pf, mid_s, p["human_goal"])
    if not np.isfinite(g_hh):
        errors.append("human start->goal unreachable")
    elif g_hh > hvia + 0.8:
        warnings.append(f"human path does not cross door (geo {g_hh:.1f} vs via {hvia:.1f})")
    if errors:
        return None, errors, warnings, {}

    for k in PTS:
        clr = pf.distance_to_closest_obstacle(p[k].astype(np.float32), 2.0)
        need = 0.5 if k == "human_goal" else 0.3
        if clr < need:
            warnings.append(f"{k}: clearance {clr:.2f}m (<{need})")

    # Recipe checks (ep70fix template, verified 2026-07-26): straight corridor,
    # shallow robot_goal, DEEP human_goal. These pass navigability but never
    # audit as SWEET otherwise -- deep robot_goal makes the robot grind walls
    # past the door; a shallow human_goal parks the human in the conflict zone
    # so yield never releases (timeout). depth: + = robot-start side of door.
    axis = (de - ds)[[0, 2]]
    axis = axis / (np.linalg.norm(axis) or 1.0)
    nrm = np.array([-axis[1], axis[0]])
    sgn = np.sign((p["robot_start"] - mid_s)[[0, 2]] @ nrm) or 1.0
    depth = lambda pt: float(sgn * ((pt - mid_s)[[0, 2]] @ nrm))
    lat = lambda pt: float(abs((pt - mid_s)[[0, 2]] @ axis))
    if depth(p["robot_goal"]) < -3.0:
        warnings.append(f"robot_goal {-depth(p['robot_goal']):.1f}m past door (deep -> wall-grind; recipe 1-2.5)")
    if depth(p["human_goal"]) < 3.0:
        warnings.append(f"human_goal only {depth(p['human_goal']):.1f}m past door (shallow -> parks in conflict; recipe 4-6)")
    off = [k for k in PTS if lat(p[k]) > 1.2]
    if off:
        warnings.append(f"off-corridor (lateral>1.2m): {','.join(off)} -> weaving/collision risk")

    dt = abs(g3._geo(pf, p["robot_start"], mid_s) / 1.0
             - g3._geo(pf, p["human_start"], mid_s) / 0.8)

    ep = json.loads(json.dumps(template))
    ep["start_position"] = [float(x) for x in p["robot_start"]]
    info = ep["info"]
    info["robot_goal"] = [float(x) for x in p["robot_goal"]]
    info["human_start"] = [float(x) for x in p["human_start"]]
    info["human_goal"] = [float(x) for x in p["human_goal"]]
    info["door_width"] = float(np.linalg.norm((de - ds)[[0, 2]]))
    info.pop("v3", None)
    info.pop("v2_transform", None)
    info["manual"] = dict(
        click_id=entry["id"], note=entry.get("note", ""),
        dt_arrival=round(dt, 2),
        snap_shift=[round(float(np.linalg.norm(
            (p[k] - np.array([entry[k][0], y, entry[k][1]]))[[0, 2]])), 3) for k in PTS],
        warnings=warnings,
    )
    return ep, [], warnings, dict(mid=mid_s, pts=p, dt=dt)


def render_review(rgb, extent, entry, geo, episode_id, out_png):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, extent=extent, origin="upper")
    ds, de = np.array(entry["door_start"]), np.array(entry["door_end"])
    ax.plot([ds[0], de[0]], [ds[2], de[2]], "r-", lw=2.5)
    p = geo["pts"]
    for who, c in (("robot", "#2196f3"), ("human", "#ff9800")):
        s, g = p[f"{who}_start"], p[f"{who}_goal"]
        ax.plot([s[0], g[0]], [s[2], g[2]], "--", color=c, lw=1.2)
        ax.plot(s[0], s[2], "o", color=c, ms=10)
        ax.plot(g[0], g[2], "*", color=c, ms=16)
    warn = "; ".join(geo["warnings"]) if geo["warnings"] else "clean"
    ax.set_title(f"ep{episode_id}  {entry['id']}  dt={geo['dt']:.2f}s\n{warn}", fontsize=10)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    plt.tight_layout(); plt.savefig(out_png, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clicks")
    ap.add_argument("out")
    ap.add_argument("--review-dir", default="hrl_pipeline/clicking/review")
    args = ap.parse_args()
    os.makedirs(args.review_dir, exist_ok=True)

    clicks = json.load(open(args.clicks))
    src = json.load(gzip.open(clicks["source_dataset"], "rt"))
    entries = clicks["entries"]

    by_scene = {}
    for j, e in enumerate(entries):
        by_scene.setdefault((e["scene_id"], e["scene_dataset_config"]), []).append(j)

    results = {}   # clicks index -> (ep|None, errors, warnings, review kwargs)
    for (sid, sds), idxs in sorted(by_scene.items()):
        sim, cam = make_sim(sid, sds)
        pf = sim.pathfinder
        for j in idxs:
            e = entries[j]
            tmpl = src["episodes"][e["src_idx"]]
            if str(tmpl.get("episode_id")) != str(e["src_episode_id"]):
                print(f"WARN {e['id']}: src_idx {e['src_idx']} has episode_id "
                      f"{tmpl.get('episode_id')} != {e['src_episode_id']}")
            ep, errors, warnings, geo = build_one(e, tmpl, pf)
            review = None
            if ep is not None:
                mid = geo["mid"]
                rgb, extent = render_at(sim, cam, float(mid[0]), float(mid[2]), HALF, float(mid[1]))
                rgb = bake_navmesh_outline(rgb, extent, pf, float(mid[1]))
                geo["warnings"] = warnings
                review = (rgb, extent, geo)
            results[j] = (ep, errors, warnings, review)
        sim.close()

    kept, report = [], []
    for j, e in enumerate(entries):
        ep, errors, warnings, review = results[j]
        if ep is None:
            report.append(dict(id=e["id"], status="error", errors=errors, warnings=warnings))
            continue
        eid = len(kept)
        ep["episode_id"] = str(eid)
        kept.append(ep)
        rgb, extent, geo = review
        render_review(rgb, extent, e, geo, eid, f"{args.review_dir}/{eid:02d}_{e['id']}.png")
        report.append(dict(id=e["id"], status="ok", episode_id=eid,
                           dt_arrival=round(geo["dt"], 2), warnings=warnings))

    out = dict(src)
    out["episodes"] = kept
    json.dump(out, gzip.open(args.out, "wt"))
    rp = args.out.replace(".json.gz", "_report.jsonl")
    with open(rp, "w") as f:
        for r in report:
            f.write(json.dumps(r) + "\n")

    n_err = sum(r["status"] == "error" for r in report)
    n_warn = sum(bool(r.get("warnings")) for r in report if r["status"] == "ok")
    print(f"built {len(kept)}/{len(entries)}  (errors={n_err}, with-warnings={n_warn})")
    for r in report:
        if r["status"] == "error":
            print(f"  ERROR {r['id']}: {'; '.join(r['errors'])}")
        elif r["warnings"]:
            print(f"  warn  ep{r['episode_id']} {r['id']}: {'; '.join(r['warnings'])}")
    print(f"-> {args.out}  (report -> {rp}, review -> {args.review_dir}/)")


if __name__ == "__main__":
    main()
