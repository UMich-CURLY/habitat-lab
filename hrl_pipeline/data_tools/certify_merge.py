"""Merge cert_<ds>_<mode>.json stats into a per-episode solvability certificate.

Usage: python3 certify_merge.py <dataset_stem>
Writes cert_<ds>.csv: episode, per-mode success, verdict, teacher_class.

verdict:
  SWEET      - rule_markov succeeds (expert-success scene)
  SOLVABLE   - expert fails but some mode succeeds -> certified A-class scene;
               the succeeding modes name the behavior PPO must discover
  UNCERT     - no scripted mode succeeds; drop it or hand-verify before
               putting it in the 15 "expert-fail" training scenes
"""
import csv
import json
import sys

MODES = ["always_go", "rule_wait", "rule_waitgo", "smart_wait", "rule_markov", "rule_yield"]


def per_ep(path):
    try:
        s = json.load(open(path))
    except FileNotFoundError:
        return {}
    d = {}
    for k, v in s.items():
        epid = k.split("|")[1]
        d.setdefault(epid, []).append(float(v.get("social_nav_to_pos_success", 0)))
    return {k: sum(v) / len(v) for k, v in d.items()}


def main(ds):
    per_mode = {m: per_ep(f"cert_{ds}_{m}.json") for m in MODES}
    eps = sorted({e for d in per_mode.values() for e in d}, key=lambda x: (len(x), x))
    out = f"cert_{ds}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + MODES + ["verdict", "notes"])
        counts = {"SWEET": 0, "SOLVABLE": 0, "UNCERT": 0}
        for e in eps:
            row = [per_mode[m].get(e, -1) for m in MODES]
            markov = per_mode["rule_markov"].get(e, 0)
            any_ok = any(v >= 0.5 for v in row)
            if markov >= 0.5:
                verdict = "SWEET"
            elif any_ok:
                verdict = "SOLVABLE"
            else:
                verdict = "UNCERT"
            counts[verdict] += 1
            ok_modes = ",".join(m for m, v in zip(MODES, row) if v >= 0.5)
            w.writerow([e] + row + [verdict, ok_modes])
        print(counts, "->", out)


if __name__ == "__main__":
    main(sys.argv[1])
