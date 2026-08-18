"""Render one FURNITURE-AWARE top-down per scene (navmesh at the runtime radius
0.25 + include_static_objects) with every distinct door line marked and a world
coordinate grid, so a human can browse scenes and pick which to build episodes
in. No robot/human goals drawn — those come later.

Usage: python hrl_pipeline/scene_tools/render_scene_menu.py <dataset.json.gz> <out_dir>
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

MPP = 0.05
R = 0.25  # runtime agent_0 radius


def furniture_pf(sim, scene_id, scene_ds):
    bcfg = habitat_sim.SimulatorConfiguration()
    bcfg.scene_id = scene_id
    bcfg.scene_dataset_config_file = scene_ds
    bcfg.enable_physics = False
    s = habitat_sim.Simulator(
        habitat_sim.Configuration(bcfg, [habitat_sim.agent.AgentConfiguration()])
    )
    nms = habitat_sim.NavMeshSettings()
    nms.set_defaults()
    nms.agent_radius = R
    nms.agent_height = 1.5
    nms.include_static_objects = True
    s.recompute_navmesh(s.pathfinder, nms)
    return s


def distinct_doors(eps):
    """Dedupe door lines across a scene's episodes (round midpoint to 0.5 m)."""
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
    src = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/habitat-lab/hrl_pipeline/clicking/scene_menu"
    os.makedirs(out_dir, exist_ok=True)
    d = json.load(gzip.open(src, "rt"))
    by_scene = {}
    for e in d["episodes"]:
        by_scene.setdefault((e["scene_id"], e["scene_dataset_config"]), []).append(e)

    names = []
    for (sid, sds), eps in sorted(by_scene.items(), key=lambda kv: kv[0][0]):
        short = sid.split("/")[-1].split(".")[0]
        doors = distinct_doors(eps)
        try:
            s = furniture_pf(None, sid, sds)
        except Exception as e:
            print("skip", short, e)
            continue
        pf = s.pathfinder
        y = float(eps[0]["start_position"][1])
        td = maps.get_topdown_map(pf, height=y, meters_per_pixel=MPP)
        lo, hi = pf.get_bounds()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(td, cmap="gray_r", origin="upper",
                  extent=[lo[0], hi[0], hi[2], lo[2]], interpolation="nearest")
        for i, (ds, de, w) in enumerate(doors):
            ax.plot([ds[0], de[0]], [ds[2], de[2]], "-", color="red", lw=3, zorder=5)
            ax.annotate(f"door{i} {w:.2f}m", ((ds[0]+de[0])/2, (ds[2]+de[2])/2),
                        color="red", fontsize=8, zorder=6, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_title(f"{short}   ({len(eps)} eps, {len(doors)} doors)  red=door(width)")
        ax.set_xlabel("world X"); ax.set_ylabel("world Z")
        ax.grid(True, color="cyan", alpha=0.4, lw=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
        ax.tick_params(labelsize=6)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{short}.png", dpi=100)
        plt.close(fig)
        s.close()
        names.append((short, len(eps), len(doors)))
        print("rendered", short, len(eps), "eps", len(doors), "doors", flush=True)

    # contact sheet
    n = len(names)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    import matplotlib.image as mpimg
    for j, (short, ne, nd) in enumerate(names):
        img = mpimg.imread(f"{out_dir}/{short}.png")
        axes[j].imshow(img)
        axes[j].set_title(f"{short[:14]}\n{ne}ep {nd}d", fontsize=6)
        axes[j].axis("off")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/_grid.png", dpi=110, bbox_inches="tight")
    print(f"\n{n} scenes -> {out_dir}/  (overview: {out_dir}/_grid.png)")


if __name__ == "__main__":
    main()
