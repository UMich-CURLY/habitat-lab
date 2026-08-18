"""Summarize the lr/2 probe: matched-budget trend, lr 5e-5 vs 2.5e-5.

Reads results/stats_lrprobe_{base,half}_ck{5,7,9}.json (evals_per_ep=1 on
EVAL: positional ids 0-35 = train36, 36-37 = zero-leak holdout) and
results/TEACHER_CERT3.csv for the class split. 1-eval numbers are a screen --
flaky-boundary episodes wobble +/-1-2.
"""
import csv
import json
import os

ROOT = os.environ.get("HAB", "/habitat-lab")
HP = f"{ROOT}/hrl_pipeline"

cert = {
    int(r["train36_id"]): r
    for r in csv.DictReader(open(f"{HP}/results/TEACHER_CERT3.csv"))
}
ss = [i for i in cert if cert[i]["final_class"] == "stable-success"]
sf = [i for i in cert if cert[i]["final_class"] == "stable-failure"]

print(f"{'point':<12} {'succ36':>7} {'coll36':>7} {'steps':>6} "
      f"{'ss(28)':>7} {'sf':>12} {'holdout':>18}")
for arm, lab in (("base", "lr5e-5"), ("half", "lr2.5e-5")):
    for ck, upd in ((5, "u73"), (7, "u98"), (9, "u122")):
        p = f"{HP}/results/stats_lrprobe_{arm}_ck{ck}.json"
        if not os.path.exists(p):
            print(f"{lab}@{upd}: (pending)")
            continue
        per = {}
        for k, v in json.load(open(p)).items():
            per[int(k.split("|")[1])] = v
        t36 = [per[i] for i in range(36)]
        n_s = sum(1 for v in t36 if v["social_nav_to_pos_success"] > 0.5)
        n_c = sum(1 for v in t36 if v["did_collide"] > 0.5)
        st = sum(v["num_steps"] for v in t36 if v["social_nav_to_pos_success"] > 0.5)
        st = int(st / max(n_s, 1))
        ss_s = sum(1 for i in ss if per[i]["social_nav_to_pos_success"] > 0.5)
        sf_ids = [i for i in sf if per[i]["social_nav_to_pos_success"] > 0.5]
        ho = []
        for i in (36, 37):
            v = per.get(i)
            if v is None:
                ho.append("?")
            else:
                ho.append(
                    f"{'S' if v['social_nav_to_pos_success'] > 0.5 else 'F'}"
                    f"{int(v['num_steps'])}"
                )
        print(
            f"{lab+'@'+upd:<12} {n_s:>5}/36 {n_c:>7} {st:>6} "
            f"{ss_s:>4}/28 {str(sf_ids) or '[]':>12} {'/'.join(ho):>18}"
        )
