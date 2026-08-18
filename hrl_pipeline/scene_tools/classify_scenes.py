"""G1: classify each social-nav episode as door / open-space / corridor using
pure navmesh geometry (no CV/CNN). Signals (per episode, navmesh recomputed at
the robot physical radius):

- w0  = local passable width at the door midpoint = 2*clearance(door_mid)+2*R
- normal clearance profile c(t)=clearance(door_mid + t*normal), t in [-L,L]:
    door    = sharp local minimum at t~0 that RISES on both sides (a pinch)
    corridor= stays low across the whole range
    open    = high everywhere
- narrow_len = length along the robot geodesic where local width < 1.4 m
    door = short pinch; corridor = long narrow stretch; open = ~0

Usage: python hrl_pipeline/scene_tools/classify_scenes.py <dataset.json.gz> [out.csv]
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import gzip
import json
import sys

import numpy as np
import habitat_sim

R = 0.4  # robot physical footprint radius (match runtime feel)
NARROW_W = 1.4  # width below this = "narrow" (a robot+human can't pass side by side)


def xz(p):
    return np.asarray(p, dtype=np.float64)[[0, 2]]


class SceneCache:
    def __init__(self):
        self.sim = None
        self.key = None

    def pf(self, scene_id, scene_ds):
        if self.key == scene_id:
            return self.sim.pathfinder
        if self.sim is not None:
            self.sim.close()
        bcfg = habitat_sim.SimulatorConfiguration()
        bcfg.scene_id = scene_id
        bcfg.scene_dataset_config_file = scene_ds
        bcfg.enable_physics = False
        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(bcfg, [habitat_sim.agent.AgentConfiguration()])
        )
        nms = habitat_sim.NavMeshSettings()
        nms.set_defaults()
        nms.agent_radius = R
        nms.agent_height = 1.5
        self.sim.recompute_navmesh(self.sim.pathfinder, nms)
        self.key = scene_id
        return self.sim.pathfinder


def clr(pf, p, max_r=3.0):
    return float(
        pf.distance_to_closest_obstacle(np.array(p, dtype=np.float32), max_search_radius=max_r)
    )


def robot_path_narrow_len(pf, rs, rg):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.array(rs, dtype=np.float32)
    sp.requested_end = np.array(rg, dtype=np.float32)
    if not pf.find_path(sp) or not np.isfinite(sp.geodesic_distance):
        return -1.0
    pts = [np.array(p, dtype=np.float64) for p in sp.points]
    # densify every 0.15 m, measure width = 2*clearance+2R, sum narrow arc length
    narrow = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = float(np.linalg.norm(xz(b) - xz(a)))
        n = max(int(np.ceil(seg / 0.15)), 1)
        for k in range(n):
            t = (k + 0.5) / n
            p = a + (b - a) * t
            w = 2 * clr(pf, p) + 2 * R
            if w < NARROW_W:
                narrow += seg / n
    return narrow


def classify_episode(pf, ep):
    info = ep["info"]
    ds = np.array(info["door_start"], dtype=np.float64)
    de = np.array(info["door_end"], dtype=np.float64)
    door_mid = (ds + de) / 2.0
    dvec = xz(de) - xz(ds)
    dvec = dvec / (np.linalg.norm(dvec) + 1e-9)
    nvec = np.array([-dvec[1], dvec[0]])  # door normal (xz)

    c0 = clr(pf, door_mid)
    w0 = 2 * c0 + 2 * R
    # normal clearance profile
    ts = np.arange(-2.0, 2.01, 0.25)
    prof = []
    for t in ts:
        p = door_mid + np.array([nvec[0] * t, 0.0, nvec[1] * t])
        prof.append(clr(pf, p))
    prof = np.array(prof)
    # width on both far sides (|t|>=1.25) vs at the door
    far = prof[np.abs(ts) >= 1.25]
    far_max = float(far.max()) if len(far) else c0
    pinch_ratio = far_max / max(c0, 0.05)  # >1 => door is a local pinch

    rs = np.array(ep["start_position"], dtype=np.float64)
    rg = np.array(info["robot_goal"], dtype=np.float64)
    narrow_len = robot_path_narrow_len(pf, rs, rg)

    # Decision rule (thresholds; tune with topdown cross-check):
    if w0 > 2.4 and pinch_ratio < 1.3:
        stype = "open"       # wide at the door, no pinch
    elif narrow_len > 3.0:
        stype = "corridor"   # long narrow stretch along the path
    elif w0 <= 2.2 and pinch_ratio >= 1.3:
        stype = "door"       # narrow local pinch that widens on both sides
    elif w0 > 2.4:
        stype = "open"
    else:
        stype = "door"       # default narrow-but-ambiguous -> door
    return dict(
        episode_id=ep["episode_id"],
        scene=ep["scene_id"].split("/")[-1][:16],
        type=stype,
        w0=round(w0, 2),
        door_clr=round(c0, 2),
        far_max=round(far_max, 2),
        pinch=round(pinch_ratio, 2),
        narrow_len=round(narrow_len, 2),
    )


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/habitat-lab/data/social_nav_episode_dev20_v2.json.gz"
    out = sys.argv[2] if len(sys.argv) > 2 else "/habitat-lab/hrl_pipeline/scene_types.csv"
    d = json.load(gzip.open(src, "rt"))
    cache = SceneCache()
    rows = []
    # group by scene to reuse one navmesh
    by_scene = {}
    for i, ep in enumerate(d["episodes"]):
        by_scene.setdefault((ep["scene_id"], ep["scene_dataset_config"]), []).append((i, ep))
    for (sid, sds), group in by_scene.items():
        pf = cache.pf(sid, sds)
        for idx, ep in group:
            r = classify_episode(pf, ep)
            r["idx"] = idx
            rows.append(r)
    rows.sort(key=lambda r: r["idx"])
    from collections import Counter
    print("idx  ep    scene            type      w0  door_clr far_max pinch narrow_len")
    for r in rows:
        print("%3d  %-5s %-16s %-8s %.2f  %.2f    %.2f   %.2f  %.2f"
              % (r["idx"], r["episode_id"], r["scene"], r["type"], r["w0"], r["door_clr"], r["far_max"], r["pinch"], r["narrow_len"]))
    print("\n分类汇总:", dict(Counter(r["type"] for r in rows)))
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "episode_id", "scene", "type", "w0", "door_clr", "far_max", "pinch", "narrow_len"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("-> ", out)
    if cache.sim is not None:
        cache.sim.close()


if __name__ == "__main__":
    main()
