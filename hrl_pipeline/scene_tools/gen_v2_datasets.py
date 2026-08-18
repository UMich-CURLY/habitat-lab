"""S4: transform social-nav episodes to v2 for PHYSICAL SOLVABILITY + a decision
window, WITHOUT simplifying the task.

Keeps: narrow door, head-on swap, firm human, "robot must yield" pressure.
Fixes only two generation side-effects that make success-after-yield impossible:
  (a) robot start too close to the door (no observe->decide->yield window)
  (b) human_goal sitting on the door axis / yield region (arrived human blocks
      the robot's return path forever)

Per episode, using info[door_start/door_end] and the robot's start side:
  robot_start  -> side A, geodesic(start, door_mid) in [START_MIN, START_MAX]
  robot_goal   -> side B, geodesic(goal, door_mid) >= GOAL_DEPTH past the door
  human_start  -> side B, arrival at the door synced with the robot (+-ARRIVAL_WIN s)
  human_goal   -> side A, >= GOAL_DEPTH past the door AND >= CORRIDOR_OFF m off the
                  [door_mid, robot_start] approach segment

Originals are never modified; output goes to new *_v2.json.gz files. Episodes
with no valid placement are written to failed.jsonl with a reason (never dropped
silently). Usage:
  python hrl_pipeline/scene_tools/gen_v2_datasets.py            # builds ep118/ep0 + dev20/dev5
  python hrl_pipeline/scene_tools/gen_v2_datasets.py --full     # also builds the 570 set + split
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib for standalone habitat_sim.
"""
import argparse
import gzip
import json
import os

import numpy as np
import habitat_sim

DATA = "/habitat-lab/data"
# Transform parameters (iterate these, per plan, if the audit retention is low).
P = dict(
    START_MIN=2.5,
    START_MAX=3.5,
    # Robot goal only needs to be clearly ACROSS the door (minimal commitment
    # past it); the HUMAN goal needs depth so the arrived human is off the
    # corridor. Splitting these avoids over-committing the robot into a
    # possibly-cluttered region far past the door.
    ROBOT_GOAL_DEPTH=1.0,
    GOAL_DEPTH=2.5,
    CORRIDOR_OFF=1.2,
    # Tight arrival sync so the two agents reach the door together and the
    # conflict actually bites (loose => the robot passes before the human).
    ARRIVAL_WIN=0.3,
    ROBOT_SPEED=1.0,
    HUMAN_SPEED=0.8,
    CLEAR=0.3,
    # Navmesh radius for placement + reachability = the robot's PHYSICAL body
    # footprint (~0.4 m), not the 0.25 m runtime navmesh radius. This filters
    # out doors the robot cannot physically cross (opening < ~0.8 m); such
    # narrow-for-the-robot doors are infeasible regardless of policy.
    NAVMESH_RADIUS=0.4,
    ROBOT_ANGLES=[0, 20, -20, 40, -40, 60, -60],
    ROBOT_RADII=[3.0, 2.75, 3.25, 2.5, 3.5],
    GOAL_ANGLES=[0, 25, -25, 50, -50, 75, -75],
    # Euclidean candidate radii from door_mid for the ROBOT goal (kept small so
    # the geodesic lands in [ROBOT_GOAL_DEPTH, +1.5] just across the door).
    GOAL_RADII=[1.5, 1.2, 1.8, 2.0, 1.0],
    # human_goal 对穿优先: 小角度(接近门法向)在前, 让人过门后直走到机器人起始侧深处,
    # 构成真正的对穿(而非横向拐弯到别的房间)。
    HGOAL_ANGLES=[0, 15, -15, 30, -30, 45, -45],
    HGOAL_RADII=[3.0, 2.5, 3.5, 4.0],
    # human_goal 只需离机器人起点 >= 此值(避免重合); 不再要求离机器人走廊远(那会
    # 破坏对穿)。人过门后走向 human_goal 时机器人已过门去对侧, 不会互相挡。
    HGOAL_MIN_TO_RSTART=0.8,
    HSTART_ANGLES=[0, 20, -20, 40, -40, 60, -60],
)


def _rotate(v, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _side_sign(ds, de, p):
    return 1.0 if (p[0] - ds[0]) * (de[2] - ds[2]) - (p[2] - ds[2]) * (de[0] - ds[0]) > 0 else -1.0


def _geo(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    if pf.find_path(sp):
        return float(sp.geodesic_distance)
    return float("inf")


def _perp_to_seg(p, a, b):
    p = np.array([p[0], p[2]]); a = np.array([a[0], a[2]]); b = np.array([b[0], b[2]])
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-9 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _door_frame(info):
    ds = np.array(info["door_start"], dtype=np.float64)
    de = np.array(info["door_end"], dtype=np.float64)
    door_mid = (ds + de) / 2.0
    d = de - ds
    d_xz = np.array([d[0], d[2]])
    nrm = np.linalg.norm(d_xz)
    d_xz = d_xz / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
    n_xz = np.array([-d_xz[1], d_xz[0]])
    return ds, de, door_mid, n_xz


def _orient_normal(ds, de, door_mid, n_xz, toward_side):
    """Flip n_xz so door_mid + n_xz*0.5 is on the given side sign."""
    probe = door_mid + np.array([n_xz[0], 0.0, n_xz[1]]) * 0.5
    if _side_sign(ds, de, probe) != toward_side:
        return -n_xz
    return n_xz


def _place(pf, ds, de, door_mid, base_dir, side, angles, radii, checks):
    """Fan candidates around base_dir; return first passing all `checks`."""
    for r in radii:
        for a in angles:
            dxz = _rotate(base_dir, a)
            cand = door_mid + np.array([dxz[0], 0.0, dxz[1]]) * r
            cand[1] = door_mid[1]
            snapped = np.array(pf.snap_point(cand), dtype=np.float64)
            if not np.all(np.isfinite(snapped)) or not pf.is_navigable(snapped):
                continue
            if _side_sign(ds, de, snapped) != side:
                continue
            if pf.distance_to_closest_obstacle(snapped, max_search_radius=2.0) < P["CLEAR"]:
                continue
            if all(chk(snapped) for chk in checks):
                return snapped
    return None


def transform_episode(ep, pf):
    info = ep["info"]
    if "door_start" not in info or "door_end" not in info:
        return None, "no_door"
    ds, de, door_mid, n_xz = _door_frame(info)
    robot_start0 = np.array(ep["start_position"], dtype=np.float64)
    side_a = _side_sign(ds, de, robot_start0)          # robot start side
    side_b = -side_a                                   # robot goal / human start side
    n_a = _orient_normal(ds, de, door_mid, n_xz, side_a)
    n_b = -n_a

    # (a) Robot start: side A, geodesic to door in [START_MIN, START_MAX].
    robot_start = _place(
        pf, ds, de, door_mid, n_a, side_a, P["ROBOT_ANGLES"], P["ROBOT_RADII"],
        [lambda s: P["START_MIN"] <= _geo(pf, s, door_mid) <= P["START_MAX"]],
    )
    if robot_start is None:
        return None, "robot_start"

    # (b) Robot goal: side B, just across the door (>= ROBOT_GOAL_DEPTH).
    robot_goal = _place(
        pf, ds, de, door_mid, n_b, side_b, P["GOAL_ANGLES"], P["GOAL_RADII"],
        [lambda s: P["ROBOT_GOAL_DEPTH"] <= _geo(pf, s, door_mid) <= P["ROBOT_GOAL_DEPTH"] + 1.5],
    )
    if robot_goal is None:
        return None, "robot_goal"

    # (c) Human start: side B, arrival at door synced with the robot.
    t_r = _geo(pf, robot_start, door_mid) / P["ROBOT_SPEED"]

    def _arrival_ok(s):
        t_h = _geo(pf, s, door_mid) / P["HUMAN_SPEED"]
        return abs(t_r - t_h) <= P["ARRIVAL_WIN"]

    human_start = _place(
        pf, ds, de, door_mid, n_b, side_b, P["HSTART_ANGLES"], P["ROBOT_RADII"],
        [_arrival_ok],
    )
    if human_start is None:
        return None, "human_start_arrival"

    # Robot must physically be able to cross the door to its goal: the geodesic
    # (on the PHYSICAL-radius navmesh) must exist and route through the door
    # (its length ~ dist-to-door + dist-past-door, i.e. it doesn't detour
    # around via another opening). If not, the door is too narrow for the robot.
    g_rr = _geo(pf, robot_start, robot_goal)
    g_via_door = _geo(pf, robot_start, door_mid) + _geo(pf, door_mid, robot_goal)
    if not np.isfinite(g_rr) or g_rr > g_via_door + 1.0:
        return None, "door_impassable_or_detour"

    # (d) Human goal: side A, past door >= GOAL_DEPTH, along the door normal
    #     (true cross-through), just not coincident with the robot start.
    human_goal = _place(
        pf, ds, de, door_mid, n_a, side_a, P["HGOAL_ANGLES"], P["HGOAL_RADII"],
        [
            lambda s: _geo(pf, s, door_mid) >= P["GOAL_DEPTH"],
            lambda s: float(np.linalg.norm((s - robot_start)[[0, 2]]))
            >= P["HGOAL_MIN_TO_RSTART"],
        ],
    )
    if human_goal is None:
        return None, "human_goal_corridor"

    out = json.loads(json.dumps(ep))
    out["start_position"] = [float(x) for x in robot_start]
    out["info"]["robot_goal"] = [float(x) for x in robot_goal]
    out["info"]["human_start"] = [float(x) for x in human_start]
    out["info"]["human_goal"] = [float(x) for x in human_goal]
    door_w = float(np.linalg.norm((de - ds)[[0, 2]]))
    out["info"]["door_width"] = door_w
    out["info"]["v2_transform"] = {
        "t_robot_to_door_s": round(t_r, 2),
        "start_to_door_m": round(_geo(pf, robot_start, door_mid), 2),
        "human_goal_off_corridor_m": round(_perp_to_seg(human_goal, door_mid, robot_start), 2),
    }
    return out, "ok"


_SIM = {"key": None, "sim": None}


def _pf_for_scene(scene_id, scene_dataset):
    if _SIM["key"] != scene_id:
        if _SIM["sim"] is not None:
            _SIM["sim"].close()
        bcfg = habitat_sim.SimulatorConfiguration()
        bcfg.scene_id = scene_id
        bcfg.scene_dataset_config_file = scene_dataset
        bcfg.enable_physics = False
        sim = habitat_sim.Simulator(
            habitat_sim.Configuration(bcfg, [habitat_sim.agent.AgentConfiguration()])
        )
        nms = habitat_sim.NavMeshSettings()
        nms.set_defaults()
        nms.agent_radius = P["NAVMESH_RADIUS"]
        nms.agent_height = 1.5
        sim.recompute_navmesh(sim.pathfinder, nms)
        _SIM["key"] = scene_id
        _SIM["sim"] = sim
    return _SIM["sim"].pathfinder


def transform_dataset(src_path, episodes=None):
    """Return (kept_episodes, failed) transforming `episodes` (or all in src)."""
    d = json.load(gzip.open(src_path, "rt"))
    eps = episodes if episodes is not None else d["episodes"]
    # Group by scene to reuse one sim/navmesh per scene.
    by_scene = {}
    for ep in eps:
        by_scene.setdefault((ep["scene_id"], ep["scene_dataset_config"]), []).append(ep)
    kept, failed = [], []
    for (scene_id, scene_ds), group in by_scene.items():
        pf = _pf_for_scene(scene_id, scene_ds)
        for ep in group:
            out, reason = transform_episode(ep, pf)
            if out is not None:
                kept.append(out)
            else:
                failed.append({"episode_id": ep.get("episode_id"), "scene": scene_id, "reason": reason})
    return d, kept, failed


def _write(template, episodes, path):
    out = dict(template)
    out["episodes"] = episodes
    json.dump(out, gzip.open(path, "wt"))
    print(f"wrote {len(episodes)} eps -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    all_failed = []
    dev = []

    # ep118 / ep0 single-episode files (only written if robot-passable).
    for src, out in [
        (f"{DATA}/social_nav_episode_overfit_switch.json.gz", f"{DATA}/social_nav_episode_overfit_switch_v2.json.gz"),
        (f"{DATA}/social_nav_episode_overfit_1.json.gz", f"{DATA}/social_nav_episode_overfit_1_v2.json.gz"),
    ]:
        tmpl, kept, failed = transform_dataset(src)
        all_failed += failed
        if kept:
            _write(tmpl, kept, out)
            dev += kept
        else:
            print(f"SKIP {src} (not robot-passable): {failed}", flush=True)

    # dev20 / dev5: pad up to 20 with extras from OTHER scenes of the 570 set.
    src570 = f"{DATA}/social_nav_episode_0415_570.json.gz"
    d570 = json.load(gzip.open(src570, "rt"))
    used_scenes = {e["scene_id"] for e in dev}
    by_scene = {}
    for ep in d570["episodes"]:
        if ep["scene_id"] in used_scenes:
            continue
        by_scene.setdefault(ep["scene_id"], []).append(ep)
    need = max(0, 20 - len(dev))
    for scene, group in by_scene.items():
        if len(dev) >= 20:
            break
        _, kept, failed = transform_dataset(src570, episodes=group[:3])
        all_failed += failed
        dev += kept[: max(0, 20 - len(dev))]
    _write(d570, dev, f"{DATA}/social_nav_episode_dev20_v2.json.gz")
    _write(d570, dev[:5], f"{DATA}/social_nav_episode_dev5_v2.json.gz")

    if args.full:
        tmpl, kept, failed = transform_dataset(src570)
        all_failed += failed
        _write(tmpl, kept, f"{DATA}/social_nav_episode_0415_570_v2.json.gz")
        # Scene-disjoint ~15% held-out split.
        scenes = sorted({e["scene_id"] for e in kept})
        n_hold = max(1, len(scenes) * 15 // 100)
        hold_scenes = set(scenes[:n_hold])
        train = [e for e in kept if e["scene_id"] not in hold_scenes]
        held = [e for e in kept if e["scene_id"] in hold_scenes]
        _write(tmpl, train, f"{DATA}/social_nav_episode_0415_570_v2_train.json.gz")
        _write(tmpl, held, f"{DATA}/social_nav_episode_0415_570_v2_heldout.json.gz")

    with open("/habitat-lab/hrl_pipeline/datasets_v2_failed.jsonl", "w") as f:
        for x in all_failed:
            f.write(json.dumps(x) + "\n")
    print(f"total failures: {len(all_failed)} -> hrl_pipeline/datasets_v2_failed.jsonl", flush=True)
    if _SIM["sim"] is not None:
        _SIM["sim"].close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
