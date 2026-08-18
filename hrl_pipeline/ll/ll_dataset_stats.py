"""Dataset statistics for the LL yield BC / DAgger corpora."""
import json
import os
from collections import Counter

import numpy as np

D = "/habitat-lab/hrl_pipeline/demos"
FILES = [
    ("ll_demos_v2_clean.jsonl", os.path.join(D, "ll_demos_v2_clean.jsonl")),
    ("ll_dagger_r1.jsonl", os.path.join(D, "ll_dagger_r1.jsonl")),
    ("ll_train_r1.jsonl", os.path.join(D, "ll_train_r1.jsonl")),
]


def load(path):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows


def q(a, p):
    return float(np.percentile(a, p))


def describe(name, rows):
    lin = np.array([r["act"][0] for r in rows], np.float64)
    ang = np.array([r["act"][1] for r in rows], np.float64)
    nz = (np.abs(lin) < 1e-3) & (np.abs(ang) < 1e-3)
    keys = Counter()
    for r in rows:
        keys[tuple(sorted(r.keys()))] += 1
    print(f"\n===== {name} =====")
    print(f"rows                 {len(rows)}")
    print(f"lin  mean {lin.mean():+.6f}  std {lin.std(ddof=0):.6f}  "
          f"min {lin.min():+.6f}  max {lin.max():+.6f}")
    print(f"lin  p5 {q(lin,5):+.6f}  p50 {q(lin,50):+.6f}  p95 {q(lin,95):+.6f}")
    print(f"ang  mean {ang.mean():+.6f}  std {ang.std(ddof=0):.6f}  "
          f"min {ang.min():+.6f}  max {ang.max():+.6f}")
    print(f"ang  p5 {q(ang,5):+.6f}  p50 {q(ang,50):+.6f}  p95 {q(ang,95):+.6f}")
    print(f"near-zero (|lin|<1e-3 and |ang|<1e-3): {int(nz.sum())} "
          f"= {nz.mean()*100:.3f}%")
    print(f"exact zero act [0,0]:                  "
          f"{int(((lin==0)&(ang==0)).sum())}")
    print(f"|lin|<1e-3 only: {int((np.abs(lin)<1e-3).sum())}   "
          f"|ang|<1e-3 only: {int((np.abs(ang)<1e-3).sum())}")
    print(f"|ang| mean {np.abs(ang).mean():.6f}  |ang|>=0.30: "
          f"{int((np.abs(ang)>=0.30).sum())} = {(np.abs(ang)>=0.30).mean()*100:.2f}%")
    print(f"ang saturated at +/-0.35: {int((np.abs(ang)>=0.3499).sum())} "
          f"= {(np.abs(ang)>=0.3499).mean()*100:.2f}%")
    print(f"lin<0 (reverse): {int((lin<-1e-3).sum())} = {(lin<-1e-3).mean()*100:.2f}%   "
          f"lin>0 (forward): {int((lin>1e-3).sum())} = {(lin>1e-3).mean()*100:.2f}%")
    print("row key-sets:")
    for k, c in keys.most_common():
        print(f"   {c:6d}  {list(k)}")
    nb = sum(1 for r in rows if "branch" in r)
    if nb:
        bc = Counter(r["branch"] for r in rows if "branch" in r)
        print(f"branch histogram ({nb} rows carry it):", dict(sorted(bc.items())))
        for bv in sorted(bc):
            m = np.array([r.get("branch") == bv for r in rows])
            print(f"   branch {bv}: n={int(m.sum())} lin mean {lin[m].mean():+.4f} "
                  f"ang mean {ang[m].mean():+.4f} |ang| mean {np.abs(ang[m]).mean():.4f}")
    # contiguous-t segments = episodes / skill invocations
    ts = [r["t"] for r in rows]
    seg, cur = [], 1
    for i in range(1, len(ts)):
        if ts[i] == ts[i - 1] + 1:
            cur += 1
        else:
            seg.append(cur); cur = 1
    seg.append(cur)
    seg = np.array(seg)
    print(f"contiguous-t segments: {len(seg)}  len mean {seg.mean():.1f} "
          f"p5 {q(seg,5):.0f} p50 {q(seg,50):.0f} p95 {q(seg,95):.0f} max {seg.max()}")
    # obs-side summary
    dist = np.array([r["feat"][0] for r in rows], np.float64)
    bear = np.array([r["feat"][1] for r in rows], np.float64)
    relsp = np.array([r["feat"][3] for r in rows], np.float64)
    minlid = np.array([min(r["lidar"]) for r in rows], np.float64)
    print(f"feat0 human-dist   mean {dist.mean():.4f} p5 {q(dist,5):.4f} "
          f"p50 {q(dist,50):.4f} p95 {q(dist,95):.4f}  min {dist.min():.4f}")
    print(f"feat1 bearing      mean {bear.mean():+.4f} p5 {q(bear,5):+.4f} "
          f"p50 {q(bear,50):+.4f} p95 {q(bear,95):+.4f}")
    print(f"feat3 rel-speed    mean {relsp.mean():+.4f} p5 {q(relsp,5):+.4f} "
          f"p50 {q(relsp,50):+.4f} p95 {q(relsp,95):+.4f}")
    print(f"min-lidar          mean {minlid.mean():.4f} p5 {q(minlid,5):.4f} "
          f"p50 {q(minlid,50):.4f} p95 {q(minlid,95):.4f}  min {minlid.min():.4f}")
    print(f"human closer than 1.0m: {(dist<1.0).mean()*100:.2f}%   "
          f"closer than 1.5m: {(dist<1.5).mean()*100:.2f}%")
    ep = Counter(r.get("env") for r in rows)
    print(f"distinct env ids: {len(ep)} -> {dict(sorted((k,v) for k,v in ep.items()))}")
    t = np.array([r["t"] for r in rows], np.float64)
    print(f"t: min {t.min():.0f} max {t.max():.0f} mean {t.mean():.1f}")
    return lin, ang


def main():
    data = {}
    for name, path in FILES:
        data[name] = load(path)
        describe(name, data[name])

    a = data["ll_demos_v2_clean.jsonl"]
    b = data["ll_dagger_r1.jsonl"]
    u = data["ll_train_r1.jsonl"]

    print("\n===== union check =====")
    print(f"len(A)+len(B) = {len(a)}+{len(b)} = {len(a)+len(b)}  vs len(union) = {len(u)}")
    same_head = all(u[i]["act"] == a[i]["act"] for i in range(len(a)))
    same_tail = all(u[len(a) + i]["act"] == b[i]["act"] for i in range(len(b)))
    print(f"union[:{len(a)}] acts == A acts : {same_head}")
    print(f"union[{len(a)}:] acts == B acts : {same_tail}")

    # how far does the DAgger rollout drift from the demo it shadows?
    print("\n===== A vs B pairwise drift (matched by (env,t)) =====")
    amap = {(r["env"], r["t"]): r for r in a}
    print(f"A t-values unique: {len(set(r['t'] for r in a))}  "
          f"B t-values unique: {len(set(r['t'] for r in b))}  "
          f"shared t: {len(set(r['t'] for r in a) & set(r['t'] for r in b))}")
    matched = 0
    dl, da, dd = [], [], []
    for r in b:
        k = (r["env"], r["t"])
        if k in amap:
            matched += 1
            dl.append(r["act"][0] - amap[k]["act"][0])
            da.append(r["act"][1] - amap[k]["act"][1])
            dd.append(r["feat"][0] - amap[k]["feat"][0])
    dl, da, dd = np.array(dl), np.array(da), np.array(dd)
    print(f"DAgger rows with a same-(env,t) demo row: {matched} / {len(b)}")
    if matched:
        print(f"  |delta lin| mean {np.abs(dl).mean():.6f}  p50 {q(np.abs(dl),50):.6f} "
              f"p95 {q(np.abs(dl),95):.6f}  max {np.abs(dl).max():.6f}")
        print(f"  |delta ang| mean {np.abs(da).mean():.6f}  p50 {q(np.abs(da),50):.6f} "
              f"p95 {q(np.abs(da),95):.6f}  max {np.abs(da).max():.6f}")
        print(f"  |delta human-dist| mean {np.abs(dd).mean():.6f} "
              f"p95 {q(np.abs(dd),95):.6f}  max {np.abs(dd).max():.6f}")
        print(f"  rows where actions differ at all: "
              f"{int(((np.abs(dl)>1e-9)|(np.abs(da)>1e-9)).sum())}")

    # per-env row counts side by side
    print("\n===== per-env row counts A vs B =====")
    ca = Counter(r["env"] for r in a)
    cb = Counter(r["env"] for r in b)
    for k in sorted(set(ca) | set(cb)):
        print(f"  env {k}: A={ca.get(k,0):6d}  B={cb.get(k,0):6d}  diff={cb.get(k,0)-ca.get(k,0):+d}")

    # near-zero fraction split
    print("\n===== freeze-branch accounting =====")
    for nm, rows in (("A demos", a), ("B dagger", b), ("union", u)):
        lin = np.array([r["act"][0] for r in rows])
        ang = np.array([r["act"][1] for r in rows])
        nz = (np.abs(lin) < 1e-3) & (np.abs(ang) < 1e-3)
        print(f"  {nm:9s} near-zero {int(nz.sum()):6d} / {len(rows):6d} = {nz.mean()*100:.3f}%")

    # 2-D action mode histogram
    print("\n===== action modes (rounded to 0.05) =====")
    for nm, rows in (("A demos", a), ("B dagger", b)):
        c = Counter((round(r["act"][0] / 0.05) * 0.05, round(r["act"][1] / 0.05) * 0.05)
                    for r in rows)
        print(f"  -- {nm} top 12 --")
        for (l, g), n in c.most_common(12):
            print(f"     lin={l:+.2f} ang={g:+.2f}  {n:6d}  {n/len(rows)*100:5.2f}%")


if __name__ == "__main__":
    main()
