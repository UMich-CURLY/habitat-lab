"""Textured top-down RGB render per scene (orthographic camera looking straight
down over the loaded scene + furniture), with door lines + world-coord grid
overlaid. This is the "video-like" furniture view, not the flat navmesh map.

Usage: python hrl_pipeline/scene_tools/render_scene_rgb_topdown.py <dataset.json.gz> <out_dir> [scene_substr]
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import gzip
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import habitat_sim
from habitat.utils.visualizations import maps

RES = 1024
CAM_H = 12.0   # metres above the floor (above furniture; scenes are open-top)


def make_sim(scene_id, scene_ds):
    bcfg = habitat_sim.SimulatorConfiguration()
    bcfg.scene_id = scene_id
    bcfg.scene_dataset_config_file = scene_ds
    bcfg.enable_physics = False
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "topdown_rgb"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    cam.resolution = [RES, RES]
    cam.position = [0.0, CAM_H, 0.0]
    cam.orientation = [-np.pi / 2, 0.0, 0.0]   # look straight down (-Y)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bcfg, [agent_cfg]))
    nms = habitat_sim.NavMeshSettings()
    nms.set_defaults(); nms.agent_radius = 0.25; nms.agent_height = 1.5
    nms.include_static_objects = True
    sim.recompute_navmesh(sim.pathfinder, nms)   # so get_bounds() is valid
    return sim, cam


def render_at(sim, cam, cx, cz, half, y_floor):
    """Render an ortho top-down view centred on (cx, cz) covering half metres
    each way. Returns (rgb, extent) with a strictly linear pixel<->world map."""
    # ortho_scale in habitat-sim = world units per pixel from image centre.
    cam.ortho_scale = 1.0 / (2 * half)   # visible width = 1/ortho_scale
    agent = sim.get_agent(0)
    st = agent.get_state()
    st.position = np.array([cx, y_floor, cz], dtype=np.float32)
    st.rotation = np.quaternion(1, 0, 0, 0)
    agent.set_state(st)
    obs = sim.get_sensor_observations()
    rgb = obs["topdown_rgb"][:, :, :3]
    # world<->image mapping: image spans [cx-half, cx+half] x [cz-half, cz+half].
    # ortho cam looking down: +worldX -> +col, +worldZ -> +row (top=cz-half).
    extent = [cx - half, cx + half, cz + half, cz - half]  # L,R,bottom,top for imshow
    return rgb, extent


def render_scene(sim, cam, lo, hi):
    """Center the ortho cam on the scene, scale to cover it (square), render."""
    cx, cz = (lo[0] + hi[0]) / 2.0, (lo[2] + hi[2]) / 2.0
    half = max(hi[0] - lo[0], hi[2] - lo[2]) / 2.0 + 0.5   # square coverage + margin
    return render_at(sim, cam, cx, cz, half, float(lo[1]))


def distinct_doors(eps):
    seen, out = set(), []
    for e in eps:
        info = e.get("info", {})
        if "door_start" not in info or "door_end" not in info:
            continue
        ds = np.array(info["door_start"], float); de = np.array(info["door_end"], float)
        mid = (ds + de) / 2.0
        key = (round(mid[0] * 2) / 2, round(mid[2] * 2) / 2)
        if key in seen:
            continue
        seen.add(key)
        out.append((ds, de, float(np.linalg.norm((de - ds)[[0, 2]]))))
    return out


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    only = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(out_dir, exist_ok=True)
    d = json.load(gzip.open(src, "rt"))
    by_scene = {}
    for e in d["episodes"]:
        by_scene.setdefault((e["scene_id"], e["scene_dataset_config"]), []).append(e)

    for (sid, sds), eps in sorted(by_scene.items(), key=lambda kv: kv[0][0]):
        short = sid.split("/")[-1].split(".")[0]
        if only and only not in short:
            continue
        try:
            sim, cam = make_sim(sid, sds)
            lo, hi = sim.pathfinder.get_bounds()
            rgb, extent = render_scene(sim, cam, lo, hi)
        except Exception as ex:
            print("skip", short, ex, flush=True)
            continue
        doors = distinct_doors(eps)
        # navmesh walkable outline reveals the FULL floor plan (rooms/walls),
        # incl. rooms hidden under a ceiling in the RGB top-down.
        td = maps.get_topdown_map(sim.pathfinder, height=float(lo[1]),
                                  meters_per_pixel=0.05)
        walk = (td == 1).astype(float)
        Xg = np.linspace(lo[0], hi[0], td.shape[1])
        Zg = np.linspace(lo[2], hi[2], td.shape[0])
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(rgb, extent=extent, origin="upper")
        ax.contour(Xg, Zg, walk, levels=[0.5], colors="cyan", linewidths=1.0,
                   zorder=4)
        for i, (ds, de, w) in enumerate(doors):
            ax.plot([ds[0], de[0]], [ds[2], de[2]], "-", color="red",
                    lw=2.5, zorder=5)
            ax.annotate(f"d{i} {w:.2f}", ((ds[0]+de[0])/2, (ds[2]+de[2])/2),
                        color="yellow", fontsize=9, zorder=6, xytext=(3, 3),
                        textcoords="offset points")
        yspan = float(hi[1]) - float(lo[1])
        flag = "  [MULTI-FLOOR: top-down incomplete]" if yspan > 2.5 else ""
        ax.set_title(f"{short}  ({len(eps)} eps)  y-span {yspan:.1f}m{flag}  red=door, cyan=walkable outline", fontsize=11)
        ax.set_xlabel("world X"); ax.set_ylabel("world Z")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(1.0))
            axis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True, which="major", color="yellow", alpha=0.30, lw=0.5)
        ax.grid(True, which="minor", color="yellow", alpha=0.15, lw=0.3)
        ax.tick_params(labelsize=6)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        plt.tight_layout(); plt.savefig(f"{out_dir}/{short}.png", dpi=110); plt.close(fig)
        sim.close()
        print("rendered", short, flush=True)


if __name__ == "__main__":
    main()
