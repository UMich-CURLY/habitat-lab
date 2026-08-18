"""G2: render an annotated top-down map per episode so a human can eyeball the
geometric door/open/corridor classification. Reuses the navmesh from
classify_scenes. One PNG grid for the whole dataset.

Overlays: door line (red), robot start (blue o) + goal (blue *), human start
(orange o) + goal (orange *). Title = idx/ep, scene_type, w0.

Usage: python hrl_pipeline/scene_tools/render_topdown.py <dataset.json.gz> <scene_types.csv> <out.png>
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import csv
import gzip
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classify_scenes import SceneCache
from habitat.utils.visualizations import maps

MPP = 0.05  # meters per pixel


def g(pf, td_shape, x, z):
    """world (x,z) -> (col, row) for imshow (x-axis=col, y-axis=row)."""
    row, col = maps.to_grid(z, x, td_shape, pathfinder=pf)
    return col, row


def render(pf, ep, ax, label):
    info = ep["info"]
    y = float(ep["start_position"][1])
    td = maps.get_topdown_map(pf, height=y, meters_per_pixel=MPP)
    ax.imshow(td, cmap="gray_r", origin="upper", interpolation="nearest")

    pts = {
        "door_s": info["door_start"],
        "door_e": info["door_end"],
        "rs": ep["start_position"],
        "rg": info["robot_goal"],
        "hs": info["human_start"],
        "hg": info["human_goal"],
    }
    G = {k: g(pf, td.shape, p[0], p[2]) for k, p in pts.items()}
    # door line
    ax.plot([G["door_s"][0], G["door_e"][0]], [G["door_s"][1], G["door_e"][1]],
            "-", color="red", lw=2, zorder=5)
    # robot
    ax.plot(*G["rs"], "o", color="dodgerblue", ms=7, zorder=6)
    ax.plot(*G["rg"], "*", color="dodgerblue", ms=12, zorder=6)
    # human
    ax.plot(*G["hs"], "o", color="darkorange", ms=7, zorder=6)
    ax.plot(*G["hg"], "*", color="darkorange", ms=12, zorder=6)
    # start->goal thin guide lines
    ax.plot([G["rs"][0], G["rg"][0]], [G["rs"][1], G["rg"][1]], ":", color="dodgerblue", lw=0.8)
    ax.plot([G["hs"][0], G["hg"][0]], [G["hs"][1], G["hg"][1]], ":", color="darkorange", lw=0.8)

    # crop to ROI around all points
    xs = [v[0] for v in G.values()]
    ys = [v[1] for v in G.values()]
    m = 25  # px margin
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(max(ys) + m, min(ys) - m)  # inverted y (image)
    ax.set_title(label, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    src = sys.argv[1]
    csv_path = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "/habitat-lab/hrl_pipeline/topdown.png"
    d = json.load(gzip.open(src, "rt"))
    types = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            types[int(r["idx"])] = r
    eps = d["episodes"]
    n = len(eps)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()
    cache = SceneCache()
    # group by scene to reuse navmesh, but render into the right subplot by idx
    order = sorted(range(n), key=lambda i: eps[i]["scene_id"])
    for i in order:
        ep = eps[i]
        pf = cache.pf(ep["scene_id"], ep["scene_dataset_config"])
        t = types.get(i, {})
        label = "%d ep%s\n%s w0=%s" % (i, ep["episode_id"], t.get("type", "?"), t.get("w0", "?"))
        render(pf, ep, axes[i], label)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("->", out)
    if cache.sim is not None:
        cache.sim.close()


if __name__ == "__main__":
    main()
