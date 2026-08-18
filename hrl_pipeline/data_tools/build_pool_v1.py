"""Consolidate every certified episode into ONE labeled pool dataset.

Reads all source datasets + certification stats, assigns each episode a label:
  sweet        conflict-forcing, expert (rule_markov) passes
  trivial_eff  both always_go and expert pass, expert wastes >=60 steps
               (efficiency-failure material: student can beat expert on time)
  trivial_zero both pass, no meaningful gap (variety filler)
  fail_B_wait  expert fails; wait-family certificate stable  (should-wait type)
  fail_C_dither expert fails; legacy-yield certificate stable (timing/geometry)
  fail_A_go    expert fails; always_go certificate stable    (over-conservative)
  challenge    knife-edge / flaky certificates -> eval-only, not training
Dead episodes (no mode ever succeeds) are EXCLUDED from the dataset but listed
in the CSV with reason.

Outputs:
  data/social_nav_episode_pool_v1.json.gz   (episodes re-id'd 0..N-1, each with
                                             info["pool"] = provenance + label)
  hrl_pipeline/POOL_v1.csv                  (master table incl. excluded rows)

Run inside the container AFTER _recheck_all.sh finished.
"""
import csv
import gzip
import json
import os

SD = "/habitat-lab/hrl_pipeline/results"
DATA = "/habitat-lab/data"


def stats(path, key="social_nav_to_pos_success"):
    """-> {epid: [per-eval values]}"""
    p = os.path.join(SD, path)
    if not os.path.exists(p):
        return {}
    out = {}
    for k, v in json.load(open(p)).items():
        out.setdefault(k.split("|")[1], []).append(float(v.get(key, 0)))
    return out


def steps(path):
    p = os.path.join(SD, path)
    if not os.path.exists(p):
        return {}
    out = {}
    for k, v in json.load(open(p)).items():
        out.setdefault(k.split("|")[1], []).append(int(v.get("num_steps", 0)))
    return out


def ok(vals):  # stable pass
    return vals and all(v >= 0.5 for v in vals)


def ded(vals):  # stable fail
    return vals and all(v < 0.5 for v in vals)


def load_eps(name):
    """Raw click batches live in data/archive_20260817/ now -- only the nine
    built datasets sit at the top of data/ -- so look in both places."""
    for d in (DATA, f"{DATA}/archive_20260817"):
        p = f"{d}/social_nav_episode_{name}.json.gz"
        if os.path.exists(p):
            blob = json.load(gzip.open(p))
            return blob, blob["episodes"]
    raise FileNotFoundError(f"social_nav_episode_{name}.json.gz not in {DATA} or its archive")


def trivial_split(mk_steps, go_steps):
    gap = (mk_steps or 0) - (go_steps or 0)
    return ("trivial_eff", gap) if gap >= 60 else ("trivial_zero", gap)


rows = []          # (label, source_set, source_idx, click_id, note, episode|None)
GAP = {}


def add(label, src, idx, ep, click="", note="", include=True):
    rows.append(dict(label=label, source=src, idx=idx, click=click,
                     note=note, ep=ep if include else None))


# ---------------- manual (first clicked batch, 17) ----------------
_, eps = load_eps("manual")
mk3 = stats("cert3_manual_rule_markov.json")
mk1 = steps("cert_manual_rule_markov.json")
go1 = stats("cert_manual_always_go.json")
go1s = steps("cert_manual_always_go.json")
for i, ep in enumerate(eps):
    e = str(i)
    click = ep["info"]["manual"]["click_id"]
    m = mk3.get(e, [])
    if e in ("0", "4"):
        add("dead", "manual", i, ep, click, "0/3 all modes", include=False)
    elif e == "6":
        add("challenge", "manual", i, ep, click,
            "A-type boundary: go passes 1-eval, markov 1/3 flaky")
    elif e == "9":
        add("challenge", "manual", i, ep, click, "markov 1/3 flaky")
    elif ok(m):
        if ok(go1.get(e, [])):
            lab, gap = trivial_split(mk1[e][0], go1s[e][0])
            add(lab, "manual", i, ep, click, f"expert +{gap} steps")
        else:
            add("sweet", "manual", i, ep, click, "markov 3/3")
    else:
        add("challenge", "manual", i, ep, click, f"markov {sum(m):.0f}/3")

# ---------------- manual0805 (30) ----------------
_, eps = load_eps("manual0805")
mk3 = stats("recheck_m0805_markov.json")
mk1 = steps("cert_manual0805_rule_markov.json")
go1 = stats("cert_manual0805_always_go.json")
go1s = steps("cert_manual0805_always_go.json")
clicks = json.load(open(f"{SD}/0805.json"))["entries"]
SPECIAL = {"10": ("fail_B_wait", "wait/waitgo 3/3, markov 0/3"),
           "15": ("fail_C_dither", "yield 3/3, markov 0/3"),
           "14": ("challenge", "yield 2/3 flaky"),
           "17": ("challenge", "markov 1/3 knife-edge"),
           "21": ("challenge", "wait 2/3 flaky"),
           "1": ("dead", "0/3 all modes + markov collides"),
           "20": ("dead", "0/3 all modes")}
for i, ep in enumerate(eps):
    e = str(i)
    click = clicks[i]["id"]
    if e in SPECIAL:
        lab, note = SPECIAL[e]
        add(lab, "manual0805", i, ep, click, note, include=lab != "dead")
        continue
    m = mk3.get(e, [])
    if ok(m):
        if ok(go1.get(e, [])):
            lab, gap = trivial_split(mk1[e][0], go1s[e][0])
            add(lab, "manual0805", i, ep, click, f"expert +{gap} steps")
        else:
            add("sweet", "manual0805", i, ep, click, "markov 3/3")
    else:
        add("challenge", "manual0805", i, ep, click,
            f"markov {sum(m):.0f}/3 (was SWEET at 1-eval)")

# ---------------- cleandev17 (17) ----------------
_, eps = load_eps("cleandev17_v2")
fam = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "7": 5}  # cd17 ep -> fam subset pos
famy = stats("recheck_cd17fam_yield.json")
mk1 = steps("cert_cd17_rule_markov.json")
mkv = stats("cert_cd17_rule_markov.json")
go1 = stats("cert_cd17_always_go.json")
go1s = steps("cert_cd17_always_go.json")
for i, ep in enumerate(eps):
    e = str(i)
    if e in fam:
        y = famy.get(str(fam[e]), [])
        if ok(y):
            add("fail_C_dither", "cleandev17_v2", i, ep, "",
                "family scene; yield 3/3 stable, markov 0/1")
        else:
            add("challenge", "cleandev17_v2", i, ep, "",
                f"family scene; yield {sum(y):.0f}/3 flaky")
    elif e in ("5", "6", "16"):
        add("dead", "cleandev17_v2", i, ep, "", "no mode passes", include=False)
    elif ok(mkv.get(e, [])):
        if ok(go1.get(e, [])):
            lab, gap = trivial_split(mk1[e][0], go1s[e][0])
            add(lab, "cleandev17_v2", i, ep, "", f"expert +{gap} steps")
        else:
            add("sweet", "cleandev17_v2", i, ep, "", "markov 1-eval")

# ---------------- bcsolid6 (6) ----------------
_, eps = load_eps("bcsolid6_v2")
mkv = stats("stats_rule_markov_solid6.json")
mk1 = steps("stats_rule_markov_solid6.json")
go1 = stats("cert_solid6_always_go.json")
go1s = steps("cert_solid6_always_go.json")
wt = stats("cert_solid6_rule_wait.json")
wg = stats("cert_solid6_rule_waitgo.json")
yl = stats("cert_solid6_rule_yield.json")
for i, ep in enumerate(eps):
    e = str(i)
    if ok(mkv.get(e, [])):
        if ok(go1.get(e, [])):
            lab, gap = trivial_split(mk1[e][0], go1s[e][0])
            add(lab, "bcsolid6_v2", i, ep, "", f"expert +{gap} steps")
        else:
            add("sweet", "bcsolid6_v2", i, ep, "", "markov 1-eval")
    else:
        cert = [n for n, s in (("go", go1), ("wait", wt), ("waitgo", wg), ("yield", yl))
                if ok(s.get(e, []))]
        if cert:
            lab = ("fail_A_go" if "go" in cert
                   else "fail_B_wait" if ("wait" in cert or "waitgo" in cert)
                   else "fail_C_dither")
            add(lab, "bcsolid6_v2", i, ep, "", f"cert(1-eval): {','.join(cert)}")
        else:
            add("dead", "bcsolid6_v2", i, ep, "", "no mode passes (1-eval)",
                include=False)

# ---------------- multiscene4 (4) ----------------
_, eps = load_eps("multiscene4_v2")
mkv = stats("stats_rule_markov_ms4.json")
mk1 = steps("stats_rule_markov_ms4.json")
go1 = stats("cert_ms4_always_go.json")
go1s = steps("cert_ms4_always_go.json")
for i, ep in enumerate(eps):
    e = str(i)
    if ok(go1.get(e, [])):
        lab, gap = trivial_split(mk1[e][0], go1s[e][0])
        add(lab, "multiscene4_v2", i, ep, "", f"expert +{gap} steps")
    else:
        add("sweet", "multiscene4_v2", i, ep, "", "markov 1-eval, go fails")

# ---------------- ep70 golden (1) ----------------
_, eps = load_eps("overfit_ep70_v2")
add("sweet", "overfit_ep70_v2", 0, eps[0], "102816756-e70(fix)",
    "golden phase-1 episode, repeatedly verified")

# ---------------- dev20_parked (20) ----------------
_, eps = load_eps("dev20_parked")
mkv = stats("stats_rule_markov_parked.json")
for i, ep in enumerate(eps):
    e = str(i)
    if ok(mkv.get(e, [])):
        # teacher never triggers on parked humans here (markov==go), no gap
        add("trivial_zero", "dev20_parked", i, ep, "", "parked human, no interaction")
    else:
        add("dead", "dev20_parked", i, ep, "",
            "uncertified stall (parked human blocks gap)", include=False)

# ---------------- write outputs ----------------
template, _ = load_eps("manual0805")
out = {k: v for k, v in template.items() if k != "episodes"}
out_eps = []
for r in rows:
    if r["ep"] is None:
        continue
    ep = json.loads(json.dumps(r["ep"]))  # deep copy
    ep["episode_id"] = str(len(out_eps))
    ep.setdefault("info", {})["pool"] = {
        "label": r["label"], "source": r["source"], "source_idx": r["idx"],
        "click_id": r["click"], "note": r["note"],
    }
    out_eps.append(ep)
out["episodes"] = out_eps
with gzip.open(f"{DATA}/social_nav_episode_pool_v1.json.gz", "wt") as f:
    json.dump(out, f)

with open(f"{SD}/POOL_v1.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pool_id", "label", "source", "source_idx", "click_id", "note"])
    pid = 0
    for r in rows:
        w.writerow([pid if r["ep"] is not None else "-", r["label"], r["source"],
                    r["idx"], r["click"], r["note"]])
        if r["ep"] is not None:
            pid += 1

from collections import Counter
inc = Counter(r["label"] for r in rows if r["ep"] is not None)
exc = Counter(r["label"] for r in rows if r["ep"] is None)
print("pool_v1:", len(out_eps), "episodes ->", dict(inc))
print("excluded:", dict(exc))
print("wrote", f"{DATA}/social_nav_episode_pool_v1.json.gz", "and POOL_v1.csv")
