"""Render a FULL-scene top-down navmesh with world-coordinate ticks + current
episode points, so we can read off good start/goal positions by hand.

Usage: python hrl_pipeline/scene_tools/scene_layout.py <dataset.json.gz> <episode_id> <out.png>
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import gzip
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classify_scenes import SceneCache
from habitat.utils.visualizations import maps

MPP = 0.05


def main():
    src, eid = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "/habitat-lab/hrl_pipeline/layout.png"
    d = json.load(gzip.open(src, "rt"))
    ep = [e for e in d["episodes"] if e["episode_id"] == eid][0]
    info = ep["info"]
    # Runtime-faithful navmesh: include the scene's static furniture so the map
    # (and any point picked off it) matches what the robot/human actually hit.
    import habitat_sim
    sc = SceneCache()
    pf = sc.pf(ep["scene_id"], ep["scene_dataset_config"])
    nms = habitat_sim.NavMeshSettings()
    nms.set_defaults()
    nms.agent_radius = 0.4
    nms.agent_height = 1.5
    nms.include_static_objects = True
    sc.sim.recompute_navmesh(sc.sim.pathfinder, nms)
    pf = sc.sim.pathfinder
    y = float(ep["start_position"][1])
    td = maps.get_topdown_map(pf, height=y, meters_per_pixel=MPP)
    lower, upper = pf.get_bounds()

    fig, ax = plt.subplots(figsize=(12, 12))
    # extent maps pixel grid to world coords: x-axis = world X, y-axis = world Z
    ax.imshow(td, cmap="gray_r", origin="upper",
              extent=[lower[0], upper[0], upper[2], lower[2]], interpolation="nearest")

    def pt(name, p, m, c):
        p = np.array(p, float)
        ax.plot(p[0], p[2], m, color=c, ms=13, zorder=6)
        ax.annotate(f"{name}\n({p[0]:.1f},{p[2]:.1f})", (p[0], p[2]),
                    color=c, fontsize=9, zorder=7,
                    xytext=(6, 6), textcoords="offset points")

    ds = np.array(info["door_start"]); de = np.array(info["door_end"])
    ax.plot([ds[0], de[0]], [ds[2], de[2]], "-", color="red", lw=3, zorder=5)
    pt("R_start", ep["start_position"], "o", "dodgerblue")
    pt("R_goal", info["robot_goal"], "*", "dodgerblue")
    pt("H_start", info["human_start"], "o", "darkorange")
    pt("H_goal", info["human_goal"], "*", "darkorange")

    ax.set_title(f"ep{eid} {ep['scene_id'].split('/')[-1][:20]}  (red=door, blue=robot, orange=human)")
    ax.set_xlabel("world X"); ax.set_ylabel("world Z")
    ax.grid(True, color="cyan", alpha=0.4, lw=0.5)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print("->", out, "bounds x[%.1f,%.1f] z[%.1f,%.1f]" % (lower[0], upper[0], lower[2], upper[2]))


if __name__ == "__main__":
    main()
