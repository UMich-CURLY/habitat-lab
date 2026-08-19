"""One-page click tool for hand-authoring manual door episodes.

Renders a furniture RGB top-down crop centred on each selected door candidate
(walkable-area outline baked into the pixels) and packs everything into ONE
self-contained HTML page. Open it in any browser (no server needed): per door
click robot_start / robot_goal / human_start / human_goal in order, then
Export a clicks.json for hrl_pipeline/scene_tools/build_manual_episodes.py. Import restores a
previous clicks.json for incremental edits.

Usage: python hrl_pipeline/scene_tools/make_click_page.py <doorcand.json.gz> <out.html>
           [--ids 0,3,17] [--half 4.5]
--ids are positions in the source file (= idx column of doorcand_v3_audit.csv);
default = all episodes with a door. One sim per scene.
Needs LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib inside the container.
"""
import argparse
import base64
import csv
import gzip
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import image as mpimg

from habitat.utils.visualizations import maps
from render_scene_rgb_topdown import make_sim, render_at

AUDIT_CSV = "hrl_pipeline/results/doorcand_v3_audit.csv"


def audit_by_idx():
    if not os.path.exists(AUDIT_CSV):
        return {}
    with open(AUDIT_CSV) as f:
        return {int(r["idx"]): r for r in csv.DictReader(f)}


def bake_navmesh_outline(rgb, extent, pf, height):
    """Paint the walkable-area boundary (cyan) directly into the pixels so the
    pixel<->world map stays strictly linear (no matplotlib axes/margins)."""
    td = maps.get_topdown_map(pf, height=height, meters_per_pixel=0.05)
    walk = td == 1
    inner = walk.copy()
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        inner &= np.roll(walk, sh, axis=ax)
    edge = walk & ~inner
    lo, hi = pf.get_bounds()
    H, W = rgb.shape[:2]
    L, R, B, T = extent
    xs = L + (np.arange(W) + 0.5) / W * (R - L)
    zs = T + (np.arange(H) + 0.5) / H * (B - T)
    cx = np.clip(np.round((xs - lo[0]) / (hi[0] - lo[0]) * (td.shape[1] - 1)).astype(int), 0, td.shape[1] - 1)
    cz = np.clip(np.round((zs - lo[2]) / (hi[2] - lo[2]) * (td.shape[0] - 1)).astype(int), 0, td.shape[0] - 1)
    m = edge[cz[:, None], cx[None, :]]
    m &= ((xs >= lo[0]) & (xs <= hi[0]))[None, :] & ((zs >= lo[2]) & (zs <= hi[2]))[:, None]
    out = rgb.copy()
    out[m] = (0, 255, 255)
    return out


def parse_ids(spec, n):
    ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            if "-" in tok:
                a, b = tok.split("-")
                ids.extend(range(int(a), int(b) + 1))
            else:
                ids.append(int(tok))
        except ValueError:
            raise SystemExit(f"--ids: bad token {tok!r} -- want ints/ranges like "
                             f"17,21,65-70 ('...' is a placeholder, not a value)")
    bad = [i for i in ids if not 0 <= i < n]
    if bad:
        raise SystemExit(f"--ids: out of range {bad} (dataset has {n} episodes)")
    return ids


def png_b64(rgb):
    buf = io.BytesIO()
    mpimg.imsave(buf, rgb, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--ids", default="")
    ap.add_argument("--half", type=float, default=6.0)  # ±6 m view: deep human_goal (4-6 m past door) must be clickable
    args = ap.parse_args()

    eps = json.load(gzip.open(args.src, "rt"))["episodes"]
    ids = (parse_ids(args.ids, len(eps)) if args.ids
           else [i for i, e in enumerate(eps) if "door_start" in e.get("info", {})])
    audit = audit_by_idx()

    by_scene = {}
    for i in ids:
        e = eps[i]
        by_scene.setdefault((e["scene_id"], e["scene_dataset_config"]), []).append((i, e))

    payload = []
    for (sid, sds), group in sorted(by_scene.items()):
        short = sid.split("/")[-1].split(".")[0]
        sim, cam = make_sim(sid, sds)
        pf = sim.pathfinder
        for i, e in group:
            info = e["info"]
            ds = np.array(info["door_start"], float)
            de = np.array(info["door_end"], float)
            mid = (ds + de) / 2.0
            y = float(np.array(pf.snap_point(mid), float)[1])
            rgb, extent = render_at(sim, cam, float(mid[0]), float(mid[2]), args.half, y)
            rgb = bake_navmesh_outline(rgb, extent, pf, y)
            a = audit.get(i, {})
            payload.append(dict(
                key=f"{short}-e{e['episode_id']}", idx=i, epid=e["episode_id"],
                scene_id=sid, sds=sds,
                door_start=[float(v) for v in ds], door_end=[float(v) for v in de],
                doorW=round(float(np.linalg.norm((de - ds)[[0, 2]])), 2),
                verdict=a.get("verdict", "?"),
                extent=[float(v) for v in extent], W=rgb.shape[1], H=rgb.shape[0],
                img="data:image/png;base64," + png_b64(rgb),
            ))
            print("rendered", payload[-1]["key"], flush=True)
        sim.close()

    html = HTML.replace("__PAYLOAD__", json.dumps(payload)).replace("__SRC__", args.src)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"-> {args.out}  ({len(payload)} doors)")


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>manual episode clicker</title>
<style>
 body{margin:0;display:flex;font:13px sans-serif;background:#1e1e1e;color:#ddd;height:100vh}
 #side{width:290px;overflow-y:auto;border-right:1px solid #444;padding:8px;flex-shrink:0}
 #main{flex:1;display:flex;flex-direction:column;padding:8px;overflow:auto}
 canvas{max-width:100%;max-height:80vh;cursor:crosshair;background:#000}
 .cand{padding:3px 6px;margin:2px 0;border-radius:4px;cursor:pointer;display:flex;gap:6px;align-items:center}
 .cand:hover{background:#333}.cand.cur{background:#2a4a6a}
 .st{width:26px;text-align:center}
 .done{color:#7c7}.skip{color:#888}
 button{margin:2px;background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:4px 10px;cursor:pointer}
 textarea{width:100%;height:100px;background:#111;color:#9c9;border:1px solid #444;box-sizing:border-box}
 #bar{padding:6px 0;font-size:14px;min-height:20px}
 #note{background:#111;color:#ddd;border:1px solid #444;width:220px}
</style></head><body>
<div id="side">
 <div><button id="exp">Export clicks.json</button><button id="imp">Import from box</button></div>
 <textarea id="io" placeholder="Export writes JSON here; or paste an old clicks.json and hit Import"></textarea>
 <div id="list"></div>
</div>
<div id="main">
 <div id="bar"></div>
 <canvas id="cv"></canvas>
 <div>note: <input id="note" placeholder="sweet-intent / easy / hard">
  <button id="undo">Undo (U)</button><button id="skipb">Skip (S)</button>
  <button id="clr">Reset door</button><button id="dup">+ variant</button></div>
 <div style="margin-top:8px">
  <button id="run" style="background:#2a5a2a">▶ Run expert（跑专家看轨迹，约1-2分钟）</button>
  <span id="runstat" style="margin-left:10px;color:#9c9"></span></div>
 <video id="traj" controls style="display:none;max-width:100%;margin-top:8px;background:#000"></video>
</div>
<script>
const DOORS = __PAYLOAD__;
const SRC = "__SRC__";
const PTS = ["robot_start","robot_goal","human_start","human_goal"];
const COL = {robot_start:"#2196f3",robot_goal:"#2196f3",human_start:"#ff9800",human_goal:"#ff9800"};
const KEY = "manual_clicks_state_v1";
const doorByIdx = {}; DOORS.forEach(d => doorByIdx[d.idx] = d);
const imgs = {}; DOORS.forEach(d => { const im = new Image(); im.src = d.img; imgs[d.idx] = im; });

let entries = [];
try { entries = (JSON.parse(localStorage.getItem(KEY)) || []).filter(e => doorByIdx[e.src_idx]); } catch (e) {}
DOORS.forEach(d => { if (!entries.some(e => e.src_idx === d.idx))
  entries.push({id: d.key, src_idx: d.idx, pts: [], note: "", skipped: false}); });
let cur = Math.max(0, entries.findIndex(e => e.pts.length < 4 && !e.skipped));

const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const bar = document.getElementById("bar"), noteEl = document.getElementById("note");
let hover = null;

function save() { localStorage.setItem(KEY, JSON.stringify(entries)); }
function door() { return doorByIdx[entries[cur].src_idx]; }
function w2p(d, wx, wz) { return [(wx - d.extent[0]) / (d.extent[1] - d.extent[0]) * d.W - 0.5,
                                  (wz - d.extent[3]) / (d.extent[2] - d.extent[3]) * d.H - 0.5]; }
function p2w(d, px, py) { return [d.extent[0] + (px + 0.5) / d.W * (d.extent[1] - d.extent[0]),
                                  d.extent[3] + (py + 0.5) / d.H * (d.extent[2] - d.extent[3])]; }

function star(x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
    ctx[i ? "lineTo" : "moveTo"](x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  ctx.closePath(); ctx.fill();
}

function draw() {
  const e = entries[cur], d = door(), im = imgs[d.idx];
  cv.width = d.W; cv.height = d.H;
  if (!im.complete) { im.onload = draw; return; }
  ctx.drawImage(im, 0, 0);
  const [dx1, dy1] = w2p(d, d.door_start[0], d.door_start[2]);
  const [dx2, dy2] = w2p(d, d.door_end[0], d.door_end[2]);
  ctx.strokeStyle = "red"; ctx.lineWidth = 3; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(dx1, dy1); ctx.lineTo(dx2, dy2); ctx.stroke();
  for (const who of ["robot", "human"]) {
    const s = e.pts[PTS.indexOf(who + "_start")], g = e.pts[PTS.indexOf(who + "_goal")];
    if (s && g) {
      ctx.strokeStyle = COL[who + "_start"]; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
      ctx.beginPath(); ctx.moveTo(...w2p(d, s[0], s[1])); ctx.lineTo(...w2p(d, g[0], g[1])); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  e.pts.forEach((p, i) => {
    const [x, y] = w2p(d, p[0], p[1]);
    ctx.fillStyle = COL[PTS[i]];
    if (PTS[i].endsWith("goal")) star(x, y, 11);
    else { ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.fill(); }
    ctx.fillStyle = "#fff"; ctx.font = "13px sans-serif"; ctx.fillText(PTS[i], x + 12, y + 4);
  });
  const nxt = e.skipped ? "SKIPPED" : e.pts.length < 4 ? `point ${e.pts.length + 1}/4: ${PTS[e.pts.length]}` : "done ✓";
  bar.textContent = `[${e.id}]  doorW=${d.doorW}  was:${d.verdict}  —  ${nxt}` +
    (hover ? `   x=${hover[0].toFixed(2)}, z=${hover[1].toFixed(2)}` : "");
  noteEl.value = e.note || "";
  list();
}

function list() {
  const el = document.getElementById("list");
  el.innerHTML = "";
  entries.forEach((e, i) => {
    const d = doorByIdx[e.src_idx], row = document.createElement("div");
    row.className = "cand" + (i === cur ? " cur" : "");
    const st = e.skipped ? '<span class="st skip">–</span>' :
      e.pts.length === 4 ? '<span class="st done">✓</span>' : `<span class="st">${e.pts.length}/4</span>`;
    row.innerHTML = `${st}<span>${e.id}</span><span style="color:#999">${d.doorW}m ${d.verdict}</span>` +
      (e.note ? `<span style="color:#c9a">${e.note}</span>` : "");
    row.onclick = () => { cur = i; draw(); };
    el.appendChild(row);
  });
}

function advance() { const n = entries.findIndex(e => e.pts.length < 4 && !e.skipped); if (n >= 0) cur = n; }

cv.addEventListener("mousemove", ev => {
  const r = cv.getBoundingClientRect(), d = door();
  hover = p2w(d, (ev.clientX - r.left) * cv.width / r.width, (ev.clientY - r.top) * cv.height / r.height);
  draw();
});
cv.addEventListener("click", ev => {
  const e = entries[cur];
  if (e.pts.length >= 4) return;
  const r = cv.getBoundingClientRect(), d = door();
  const w = p2w(d, (ev.clientX - r.left) * cv.width / r.width, (ev.clientY - r.top) * cv.height / r.height);
  e.skipped = false;
  e.pts.push([+w[0].toFixed(3), +w[1].toFixed(3)]);
  if (e.pts.length === 4) advance();
  save(); draw();
});
document.getElementById("undo").onclick = () => { entries[cur].pts.pop(); save(); draw(); };
document.getElementById("skipb").onclick = () => { entries[cur].skipped = true; advance(); save(); draw(); };
document.getElementById("clr").onclick = () => { const e = entries[cur]; e.pts = []; e.skipped = false; save(); draw(); };
document.getElementById("dup").onclick = () => {
  const e = entries[cur];
  const n = entries.filter(x => x.src_idx === e.src_idx).length;
  entries.splice(cur + 1, 0, {id: doorByIdx[e.src_idx].key + "-" + "bcdefghij"[n - 1],
                              src_idx: e.src_idx, pts: [], note: "", skipped: false});
  cur++; save(); draw();
};
noteEl.oninput = () => { entries[cur].note = noteEl.value; save(); list(); };

// Run expert on the 4 clicked points (needs click_server.py serving this page).
document.getElementById("run").onclick = async () => {
  const e = entries[cur], d = doorByIdx[e.src_idx];
  const stat = document.getElementById("runstat"), btn = document.getElementById("run");
  if (e.pts.length !== 4) { stat.textContent = "先点满 4 个点"; return; }
  btn.disabled = true; stat.textContent = "跑专家中… 约 1-2 分钟，别关页面";
  try {
    const r = await fetch("/run", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source: SRC, src_idx: d.idx, scene_id: d.scene_id, scene_dataset_config: d.sds,
        door_start: d.door_start, door_end: d.door_end, pts: e.pts})});
    const res = await r.json();
    if (!res.ok) { stat.textContent = "失败: " + (res.errors || []).join("; "); }
    else {
      stat.innerHTML = `<b>${res.verdict}</b>  步数 ${res.steps}  撞墙 ${res.scene_col}` +
        (res.warnings && res.warnings.length ? "  ⚠ " + res.warnings.join("; ") : "");
      const v = document.getElementById("traj");
      v.src = "/" + res.video + "?t=" + Date.now(); v.style.display = "block"; v.load();
    }
  } catch (err) { stat.textContent = "请求出错: " + err; }
  btn.disabled = false;
};

document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;
  if (ev.key === "u" || ev.key === "U") document.getElementById("undo").onclick();
  if (ev.key === "s" || ev.key === "S") document.getElementById("skipb").onclick();
});

document.getElementById("exp").onclick = () => {
  const out = {version: 1, source_dataset: SRC, entries: entries
    .filter(e => e.pts.length === 4)
    .map(e => { const d = doorByIdx[e.src_idx]; return {
      id: e.id, src_idx: d.idx, src_episode_id: d.epid,
      scene_id: d.scene_id, scene_dataset_config: d.sds,
      door_start: d.door_start, door_end: d.door_end,
      robot_start: e.pts[0], robot_goal: e.pts[1],
      human_start: e.pts[2], human_goal: e.pts[3], note: e.note || ""}; })};
  const txt = JSON.stringify(out, null, 1);
  document.getElementById("io").value = txt;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], {type: "application/json"}));
  a.download = "clicks.json"; a.click();
};
document.getElementById("imp").onclick = () => {
  let d; try { d = JSON.parse(document.getElementById("io").value); } catch (e) { alert("bad JSON"); return; }
  for (const ent of d.entries || []) {
    if (!doorByIdx[ent.src_idx]) continue;
    let e = entries.find(x => x.id === ent.id);
    if (!e) { e = {id: ent.id, src_idx: ent.src_idx, pts: [], note: "", skipped: false};
              entries.push(e); }
    e.pts = PTS.map(k => ent[k]); e.note = ent.note || ""; e.skipped = false;
  }
  save(); draw();
};

draw();
</script></body></html>
"""


if __name__ == "__main__":
    main()
