"""TILED textured top-down per scene, stitched by exact world coordinates.

Why tiled + clipped:
  - Each 4x4 m tile is rendered with the ortho camera centred at a KNOWN world
    (x,z), so stitching by coordinates is exact -- no scale/alignment guesswork.
  - The camera's NEAR plane is set to cut at ~2.3 m above the ground floor:
    ceilings / upper floors above the cut are clipped away, so rooms that a
    plain top-down cannot see (roofed) show their interiors, like an
    architectural section cut.

Output: one stitched PNG per scene with cyan navmesh outline (full floor plan),
red door lines, and a true-metre coordinate grid (1 m major / 0.5 m minor).

Usage: python hrl_pipeline/scene_tools/render_scene_tiled.py <dataset.json.gz> <out_dir> [scene_substr]
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

TILE_M = 4.0          # tile edge in metres
TILE_PX = 256         # tile edge in pixels -> 64 px/m
PPM = TILE_PX / TILE_M
CAM_H = 20.0          # camera metres above the ground floor
CUT_ABOVE = 2.3       # section-cut height above ground floor (clips ceilings)


def make_sim(scene_id, scene_ds):
    bcfg = habitat_sim.SimulatorConfiguration()
    bcfg.scene_id = scene_id
    bcfg.scene_dataset_config_file = scene_ds
    bcfg.enable_physics = False
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "tile"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    cam.resolution = [TILE_PX, TILE_PX]
    cam.position = [0.0, 0.0, 0.0]          # camera exactly at agent position
    cam.orientation = [-np.pi / 2, 0.0, 0.0]  # look straight down
    # visible width = 1/ortho_scale  (verified from the projection matrix)
    cam.ortho_scale = 1.0 / TILE_M
    # Section cut: clip everything above (CAM_H - CUT_ABOVE) below the camera.
    cam.near = CAM_H - CUT_ABOVE
    cam.far = CAM_H + 6.0
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bcfg, [agent_cfg]))
    nms = habitat_sim.NavMeshSettings()
    nms.set_defaults()
    nms.agent_radius = 0.25
    nms.agent_height = 1.5
    nms.include_static_objects = True
    sim.recompute_navmesh(sim.pathfinder, nms)
    return sim


def stitch_scene(sim):
    """Render TILE_M tiles over the navmesh bounds; stitch into one canvas.
    Returns (canvas RGB, extent [x0,x1,z1,z0]) with true world coords."""
    lo, hi = sim.pathfinder.get_bounds()
    floor_y = float(lo[1])
    x0, x1 = float(lo[0]) - 0.5, float(hi[0]) + 0.5
    z0, z1 = float(lo[2]) - 0.5, float(hi[2]) + 0.5
    W = int(np.ceil((x1 - x0) * PPM))
    H = int(np.ceil((z1 - z0) * PPM))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    agent = sim.get_agent(0)
    nx = int(np.ceil((x1 - x0) / TILE_M))
    nz = int(np.ceil((z1 - z0) / TILE_M))
    for iz in range(nz):
        for ix in range(nx):
            cx = x0 + TILE_M / 2 + ix * TILE_M
            cz = z0 + TILE_M / 2 + iz * TILE_M
            st = agent.get_state()
            st.position = np.array([cx, floor_y + CAM_H, cz], dtype=np.float32)
            st.rotation = np.quaternion(1, 0, 0, 0)
            agent.set_state(st)
            rgb = sim.get_sensor_observations()["tile"][:, :, :3]
            # paste: image +col = +worldX, +row = +worldZ (verified alignment)
            c0 = int(round((cx - TILE_M / 2 - x0) * PPM))
            r0 = int(round((cz - TILE_M / 2 - z0) * PPM))
            r_lo, r_hi = max(r0, 0), min(r0 + TILE_PX, H)
            c_lo, c_hi = max(c0, 0), min(c0 + TILE_PX, W)
            canvas[r_lo:r_hi, c_lo:c_hi] = rgb[(r_lo - r0):(r_hi - r0),
                                               (c_lo - c0):(c_hi - c0)]
    extent = [x0, x1, z1, z0]
    return canvas, extent, (lo, hi)


def distinct_doors(eps):
    seen, out = set(), []
    for e in eps:
        info = e.get("info", {})
        if "door_start" not in info or "door_end" not in info:
            continue
        ds = np.array(info["door_start"], float)
        de = np.array(info["door_end"], float)
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
            sim = make_sim(sid, sds)
            canvas, extent, (lo, hi) = stitch_scene(sim)
            td = maps.get_topdown_map(sim.pathfinder, height=float(lo[1]),
                                      meters_per_pixel=0.05)
        except Exception as ex:
            print("skip", short, ex, flush=True)
            continue
        walk = (td == 1).astype(float)
        Xg = np.linspace(lo[0], hi[0], td.shape[1])
        Zg = np.linspace(lo[2], hi[2], td.shape[0])

        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(canvas, extent=extent, origin="upper")
        ax.contour(Xg, Zg, walk, levels=[0.5], colors="cyan",
                   linewidths=0.9, zorder=4)
        for i, (ds, de, w) in enumerate(distinct_doors(eps)):
            ax.plot([ds[0], de[0]], [ds[2], de[2]], "-", color="red",
                    lw=2.5, zorder=5)
            ax.annotate(f"d{i} {w:.2f}", ((ds[0]+de[0])/2, (ds[2]+de[2])/2),
                        color="yellow", fontsize=9, zorder=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{short}  ({len(eps)} eps)  section-cut @ +{CUT_ABOVE}m  "
                     f"red=door, cyan=walkable outline", fontsize=11)
        ax.set_xlabel("world X"); ax.set_ylabel("world Z")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(1.0))
            axis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True, which="major", color="yellow", alpha=0.30, lw=0.5)
        ax.grid(True, which="minor", color="yellow", alpha=0.15, lw=0.3)
        ax.tick_params(labelsize=6)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{short}.png", dpi=120)
        plt.close(fig)
        sim.close()
        print("rendered", short, flush=True)


if __name__ == "__main__":
    main()
