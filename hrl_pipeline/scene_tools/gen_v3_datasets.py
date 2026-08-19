"""S4-v3 door-candidate generator. Transforms social-nav episodes into clean
narrow-doorway SWEET candidates, fixing the failure modes this project hit:

  1. FURNITURE-AWARE placement: navmesh recomputed with include_static_objects
     so no start/goal lands inside furniture (v2 used a furniture-free navmesh
     -> wedged starts that ram walls forever).
  2. REAL head-on cross-through: human_goal is on-axis (<= HGOAL_MAX_DEG off the
     door normal) AND the human's path actually crosses through the door to the
     robot's side (v2 sometimes put human_goal in a side corner -> the human
     sidesteps -> "trivial" was a FAKE conflict).
  3. YIELD SPACE near the door: require a navigable pocket within reach of the
     robot's approach that clears the human corridor -- otherwise the robot has
     nowhere to yield and the episode is unsolvable regardless of policy.
  Plus the v2 invariants: narrow door kept, robot start 2.5-3.5 m back
  (observe->decide window), door robot-passable, arrival time-synced.

Originals untouched; output -> new *_v3.json.gz. Rejects -> failed jsonl with a
reason (never dropped silently). Geometry is a PRE-FILTER; the real SWEET test
is the audit (always_go fails & rule_yield succeeds) run afterwards.

Usage: python hrl_pipeline/scene_tools/gen_v3_datasets.py <src.json.gz> <out.json.gz>
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import gzip
import json
import sys

import numpy as np
import habitat_sim

P = dict(
    # User-tuned (2026-07-13, from episode review): tighter geometry all around.
    START_MIN=2.0, START_MAX=2.5,             # robot start 2-2.5 m before the door
    ROBOT_GOAL_DEPTH=1.0,                     # robot goal 1.0-2.5 m past the door
    # human goal DEEP (2.5-4 m past the door): the firm human parks at its goal
    # forever; a shallow goal parks it inside the conflict zone next to the
    # robot's yield pocket -> the head-on predicate never clears -> yield never
    # releases -> timeout (v4 audit: 100/114). Conflict strength is set AT the
    # door, not by goal depth; depth only lets the human EXIT afterwards.
    HGOAL_DEPTH_MIN=2.5, HGOAL_DEPTH_MAX=4.0,
    ARRIVAL_WIN=0.4, ROBOT_SPEED=1.0, HUMAN_SPEED=0.8,
    NAVMESH_RADIUS=0.25, CLEAR=0.3,           # MATCH runtime navmesh
    HGOAL_CLEAR=0.5,                          # human_goal must be in the open
    HGOAL_MIN_TO_RSTART=0.8,
    CORRIDOR_CLEAR=1.1,                       # yield pocket clears the corridor
    YIELD_REACH=2.8,                          # generous: pocket within 2.8 m
    HGOAL_POCKET_CLEAR=1.5,                   # pocket >= this from human_goal
                                              # (mirrors runtime human_goal_clearance)
    # Starts: fan out to +-150 deg but PREFER SMALL angles (angle-major order).
    ROBOT_ANGLES=[0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 105, -105, 120, -120, 135, -135, 150, -150],
    ROBOT_RADII=[2.0, 2.25, 2.5],
    GOAL_ANGLES=[0, 10, -10, 20, -20],        # robot goal within +-20 deg
    GOAL_RADII=[1.2, 1.5, 1.8, 2.0, 1.0],
    HGOAL_ANGLES=[0, 10, -10, 20, -20],
    HGOAL_RADII=[2.5, 3.0, 3.5, 4.0],
    HSTART_ANGLES=[0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 105, -105, 120, -120, 135, -135, 150, -150],
)


def _rotate(v, deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _side(ds, de, p):
    return 1.0 if (p[0]-ds[0])*(de[2]-ds[2]) - (p[2]-ds[2])*(de[0]-ds[0]) > 0 else -1.0


def _geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    return float(sp.geodesic_distance) if pf.find_path(sp) else float("inf")


def _route_pts(pf, a, b, fallback):
    """Navmesh shortest-path polyline a->b (the agent's REAL route). The
    straight [a, door, b] polyline under-covers the swept area, so yield
    pockets validated against it can still sit on the actual route."""
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32); sp.requested_end = np.asarray(b, np.float32)
    if pf.find_path(sp) and len(sp.points) >= 2:
        return [np.array(pt, float) for pt in sp.points]
    return fallback


def _seg_dist_xz(p, a, b):
    p = np.array([p[0], p[2]]); a = np.array([a[0], a[2]]); b = np.array([b[0], b[2]])
    ab = b - a; d = float(np.dot(ab, ab))
    t = 0.0 if d < 1e-9 else float(np.clip(np.dot(p-a, ab)/d, 0, 1))
    return float(np.linalg.norm(p - (a + t*ab)))


def _poly_dist(p, pts):
    return min(_seg_dist_xz(p, pts[i], pts[i+1]) for i in range(len(pts)-1))


def _door_frame(info):
    ds = np.array(info["door_start"], float); de = np.array(info["door_end"], float)
    mid = (ds + de) / 2.0
    d = de - ds; dxz = np.array([d[0], d[2]]); n = np.linalg.norm(dxz)
    dxz = dxz/n if n > 1e-6 else np.array([1.0, 0.0])
    return ds, de, mid, np.array([-dxz[1], dxz[0]])


def _orient(ds, de, mid, n, side):
    probe = mid + np.array([n[0], 0, n[1]]) * 0.5
    return -n if _side(ds, de, probe) != side else n


def _place(pf, ds, de, mid, base, side, angles, radii, checks, clear=None,
           angle_major=False):
    clear = P["CLEAR"] if clear is None else clear
    order = ([(r, a) for a in angles for r in radii] if angle_major
             else [(r, a) for r in radii for a in angles])
    for r, a in order:
        if True:
            dxz = _rotate(base, a)
            cand = mid + np.array([dxz[0], 0, dxz[1]]) * r
            cand[1] = mid[1]
            s = np.array(pf.snap_point(cand), float)
            if not np.all(np.isfinite(s)) or not pf.is_navigable(s):
                continue
            if _side(ds, de, s) != side:
                continue
            if pf.distance_to_closest_obstacle(s, 2.0) < clear:
                continue
            if all(chk(s) for chk in checks):
                return s
    return None


def _yield_space(pf, robot_start, corridor, ds, de, side, human_goal=None):
    """Is there a navigable pocket near the robot that clears the human corridor
    by CORRIDOR_CLEAR, stays >= HGOAL_POCKET_CLEAR from the human goal (mirrors
    the runtime sensor's human_goal_clearance), on the robot's side, within
    YIELD_REACH? Returns detour."""
    rs_xz = np.array([robot_start[0], robot_start[2]])
    best = None
    for r in (0.75, 1.0, 1.5, 2.0, 2.5, 2.8):
        for a in range(0, 360, 30):
            d2 = _rotate(np.array([1.0, 0.0]), a)
            cand = np.array([rs_xz[0]+d2[0]*r, robot_start[1], rs_xz[1]+d2[1]*r])
            s = np.array(pf.snap_point(cand), float)
            if not np.all(np.isfinite(s)) or not pf.is_navigable(s):
                continue
            if _side(ds, de, s) != side:
                continue
            if pf.distance_to_closest_obstacle(s, 2.0) < 0.3:
                continue
            if _poly_dist(s, corridor) < P["CORRIDOR_CLEAR"]:
                continue
            if human_goal is not None and float(
                    np.linalg.norm((s - human_goal)[[0, 2]])) < P["HGOAL_POCKET_CLEAR"]:
                continue
            det = float(np.linalg.norm((s - robot_start)[[0, 2]]))
            if det <= P["YIELD_REACH"] and (best is None or det < best):
                best = det
    return best


def transform(ep, pf):
    info = ep["info"]
    if "door_start" not in info or "door_end" not in info:
        return None, "no_door", {}
    ds, de, mid, n = _door_frame(info)
    rs0 = np.array(ep["start_position"], float)
    sa = _side(ds, de, rs0); sb = -sa
    na = _orient(ds, de, mid, n, sa); nb = -na

    rstart = _place(pf, ds, de, mid, na, sa, P["ROBOT_ANGLES"], P["ROBOT_RADII"],
                    [lambda s: P["START_MIN"] <= _geo(pf, s, mid) <= P["START_MAX"]],
                    angle_major=True)
    if rstart is None:
        return None, "robot_start", {}

    rgoal = _place(pf, ds, de, mid, nb, sb, P["GOAL_ANGLES"], P["GOAL_RADII"],
                   [lambda s: P["ROBOT_GOAL_DEPTH"] <= _geo(pf, s, mid) <= P["ROBOT_GOAL_DEPTH"]+1.5])
    if rgoal is None:
        return None, "robot_goal", {}

    t_r = _geo(pf, rstart, mid) / P["ROBOT_SPEED"]
    # AUTO-MATCHED human start: to arrive at the door WITH the robot, the human
    # (slower, 0.8 m/s) must start at geodesic d_h = HUMAN_SPEED * t_r. Sample
    # radii AROUND d_h (the sync gate below still verifies geodesically).
    d_h = t_r * P["HUMAN_SPEED"]
    hstart_radii = [d_h, d_h - 0.15, d_h + 0.15, d_h - 0.3, d_h + 0.3]
    hstart = _place(pf, ds, de, mid, nb, sb, P["HSTART_ANGLES"], hstart_radii,
                    [lambda s: abs(t_r - _geo(pf, s, mid)/P["HUMAN_SPEED"]) <= P["ARRIVAL_WIN"]],
                    angle_major=True)
    if hstart is None:
        return None, "human_start_sync", {}

    # Robot must physically cross the door (not detour around).
    g_rr = _geo(pf, rstart, rgoal); g_via = _geo(pf, rstart, mid) + _geo(pf, mid, rgoal)
    if not np.isfinite(g_rr) or g_rr > g_via + 1.0:
        return None, "door_impassable_or_detour", {}

    # human_goal: on-axis, open, real cross-through (path via door to robot side).
    def _crossthrough(s):
        g_hg = _geo(pf, hstart, s)
        g_hgvia = _geo(pf, hstart, mid) + _geo(pf, mid, s)
        return np.isfinite(g_hg) and g_hg <= g_hgvia + 0.8
    hgoal = _place(pf, ds, de, mid, na, sa, P["HGOAL_ANGLES"], P["HGOAL_RADII"],
                   [lambda s: P["HGOAL_DEPTH_MIN"] <= _geo(pf, s, mid) <= P["HGOAL_DEPTH_MAX"],
                    lambda s: float(np.linalg.norm((s-rstart)[[0, 2]])) >= P["HGOAL_MIN_TO_RSTART"],
                    _crossthrough],
                   clear=P["HGOAL_CLEAR"])
    if hgoal is None:
        return None, "human_goal_crossthrough", {}

    # Yield space must clear the human's REAL route (navmesh path, not the
    # straight polyline) -- v5 audit showed pockets on the true route cause
    # blocked-gate deadlocks (timeout) or pursuit collisions.
    corridor = _route_pts(pf, hstart, hgoal, [hstart, mid, hgoal])
    detour = _yield_space(pf, rstart, corridor, ds, de, sa, human_goal=hgoal)
    if detour is None:
        return None, "no_yield_space", {}

    out = json.loads(json.dumps(ep))
    out["start_position"] = [float(x) for x in rstart]
    out["info"]["robot_goal"] = [float(x) for x in rgoal]
    out["info"]["human_start"] = [float(x) for x in hstart]
    out["info"]["human_goal"] = [float(x) for x in hgoal]
    out["info"]["door_width"] = float(np.linalg.norm((de-ds)[[0, 2]]))
    meta = dict(
        start_to_door=round(_geo(pf, rstart, mid), 2),
        door_clr=round(pf.distance_to_closest_obstacle(mid.astype(np.float32), 3.0), 2),
        yield_detour=round(detour, 2),
        hgoal_clr=round(pf.distance_to_closest_obstacle(hgoal.astype(np.float32), 3.0), 2),
    )
    out["info"]["v3"] = meta
    return out, "ok", meta


_SIM = {"key": None, "sim": None}


def pf_for(scene_id, sds):
    if _SIM["key"] != scene_id:
        if _SIM["sim"] is not None:
            _SIM["sim"].close()
        bcfg = habitat_sim.SimulatorConfiguration()
        bcfg.scene_id = scene_id; bcfg.scene_dataset_config_file = sds; bcfg.enable_physics = False
        sim = habitat_sim.Simulator(habitat_sim.Configuration(bcfg, [habitat_sim.agent.AgentConfiguration()]))
        nms = habitat_sim.NavMeshSettings(); nms.set_defaults()
        nms.agent_radius = P["NAVMESH_RADIUS"]; nms.agent_height = 1.5
        nms.include_static_objects = True     # FURNITURE-AWARE
        sim.recompute_navmesh(sim.pathfinder, nms)
        _SIM["key"] = scene_id; _SIM["sim"] = sim
    return _SIM["sim"].pathfinder


def main():
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(gzip.open(src, "rt"))
    by_scene = {}
    for ep in d["episodes"]:
        by_scene.setdefault((ep["scene_id"], ep["scene_dataset_config"]), []).append(ep)
    kept, failed = [], []
    from collections import Counter
    reasons = Counter()
    for (sid, sds), group in by_scene.items():
        pf = pf_for(sid, sds)
        for ep in group:
            o, why, meta = transform(ep, pf)
            reasons[why] += 1
            if o is not None:
                kept.append(o)
            else:
                failed.append({"episode_id": ep.get("episode_id"), "scene": sid.split("/")[-1][:16], "reason": why})
    tmpl = dict(d); tmpl["episodes"] = kept
    json.dump(tmpl, gzip.open(out, "wt"))
    fp = out.replace(".json.gz", "_failed.jsonl")
    with open(fp, "w") as f:
        for x in failed:
            f.write(json.dumps(x) + "\n")
    print(f"kept={len(kept)}/{sum(reasons.values())}  reasons={dict(reasons)}")
    print(f"-> {out}  (failed -> {fp})")
    if _SIM["sim"] is not None:
        _SIM["sim"].close()


if __name__ == "__main__":
    main()
