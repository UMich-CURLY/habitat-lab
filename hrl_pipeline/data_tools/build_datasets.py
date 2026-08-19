"""Single source of truth for every social-nav episode dataset.

Running this produces exactly NINE files in /habitat-lab/data and nothing else:

  socialnav_ALL_labeled.json.gz    every distinct configuration, with its label and
                                   its membership in each suite. Not for training --
                                   it is the catalogue you read to know what exists.
  socialnav_big_{train,val,eval}   the main suite,  ~31 configurations, 70/16/16
  socialnav_small_{train,val,eval} the 1-2 h quick suite, 18 configurations
  socialnav_legacy_hl_{train,eval} the datasets the *existing* HL checkpoints were
                                   trained and evaluated on, kept verbatim so those
                                   results stay reproducible. NOT scene-disjoint.

Two invariants the code asserts rather than assumes:
  1. big_train / big_val / big_eval share no SCENE. Splitting by episode would leak:
     two episodes in one room share its geometry.
  2. each small split's scenes are a subset of the corresponding big split's, so a
     hyper-parameter tuned on the small suite can move to the big one without
     turning a training scene into an evaluation scene.

The pool holds 64 configurations over 26 scenes but the suites want ~31, so
configurations are SELECTED, not taken wholesale: at most PER_SCENE_CAP per scene
(at this size scene diversity beats configuration count), and within a scene the
rarest label wins, because teacher-imperfect episodes are the scarce resource.

Every episode carries info.pool.uid -- a stable "<scene>#<k>" identifier that is the
same in every file it appears in. episode_id stays a plain 0..N-1 integer because
downstream analysis parses stats keys as scene|episode_id|eval_idx and casts to int.

Usage:
  python hrl_pipeline/data_tools/build_datasets.py            # plan only, writes nothing
  python hrl_pipeline/data_tools/build_datasets.py --build    # write the nine files
  python hrl_pipeline/data_tools/build_datasets.py --mode train --build   # see FAILURE_MODE
Must run as root inside the container: /habitat-lab/data is root-owned.
"""
import copy
import gzip
import json
import os
import sys
from collections import defaultdict

DATA = "/habitat-lab/data"
ARCHIVE = f"{DATA}/archive_20260817"
RESULTS = "/habitat-lab/hrl_pipeline/results"
DOC = f"{DATA}/DATASETS.md"

# The five files whose union defines the pool. Order matters only for provenance.
# They are raw click output, not products, so they live in the archive once the
# nine files exist -- hence source() looks in both places.
POOL_SOURCES = ["pool_v1", "manual", "manual0805", "TRAIN", "EVAL"]
LEGACY = {"train": "TRAIN", "eval": "EVAL"}


def source(name):
    for d in (DATA, ARCHIVE):
        p = f"{d}/social_nav_episode_{name}.json.gz"
        if os.path.exists(p):
            return p
    return None

TARGET = {"train": 21, "val": 5, "eval": 5}
SMALL_TARGET = {"train": 12, "val": 3, "eval": 3}
PER_SCENE_CAP = 2
LABEL_PRIORITY = ["fail_C_dither", "fail_B_wait", "fail_A_go", "challenge", "sweet",
                  "trivial_eff", "trivial_zero", "?"]
IMPERFECT = {"fail_C_dither", "fail_B_wait", "fail_A_go", "challenge"}
# Nothing certifies these, so they carry no learnable target. They stay in the
# catalogue -- that is how we avoid clicking them again -- and in no suite.
EXCLUDED = {"dead"}

FAILURE_SCENE = "103997541_171030615"            # 5 cfg, holds 2 of the 3 fail_C
ZERO_LEAK = ["106879080_174887211", "107734479_176000442"]  # never in any training run

VAL_SCENES = ["103997718_171030855",   # holds the only fail_B_wait
              "106366434_174226881",
              "105515160_173104077"]   # challenge
EVAL_BASE = ZERO_LEAK + ["104862534_172226625"]

# Scenes drawn from each big split. TRAIN is 7, not 6, because one of the first
# six lost its second configuration to `dead` and 6 scenes could no longer fill 12.
SMALL_TRAIN_N, SMALL_VAL_N, SMALL_EVAL_N = 7, 2, 2

# A label says WHICH controller family still certifies the episode, not how hard
# it looks. Criteria implemented in classify(); original source build_pool_v1.py.
LABEL_DOC = {
    "sweet": "teacher passes every eval and `always_go` does not -- yielding is genuinely required. The core training signal.",
    "trivial_eff": "teacher and `always_go` both pass, but the teacher is >=60 steps slower: no safety headroom, efficiency headroom",
    "trivial_zero": "teacher and `always_go` both pass and take the same time -- no conflict to learn from; variety filler",
    "challenge": "teacher passes SOME evals but not all: a flaky certificate, not a hard episode. Unstable BC target",
    "fail_A_go": "teacher fails where plain `always_go` succeeds -- the teacher is over-conservative",
    "fail_B_wait": "teacher fails; stopping (`rule_wait`/`rule_waitgo`) is enough. Standing still would have done it",
    "fail_C_dither": "teacher fails; only `rule_yield` certifies -- the robot must physically retreat, not merely stop",
    "dead": "no scripted controller certifies it. Kept in the catalogue only so it is not clicked again; in no suite",
    "?": "no certification on record",
}


# --------------------------------------------------------------------------- pool

def cfg_key(e):
    """What makes two episodes the SAME navigation problem."""
    i = e.get("info", {}) or {}
    g = lambda k: tuple(round(float(x), 3) for x in i.get(k, []))  # noqa: E731
    return (e["scene_id"].split("/")[-1],
            tuple(round(float(x), 3) for x in e.get("start_position", [])),
            g("robot_goal"), g("human_start"), g("human_goal"))


def scene_of(e):
    return e["scene_id"].split("/")[-1].split(".")[0]


def label(e):
    return (e.get("info", {}).get("pool", {}) or {}).get("label", "?")


def load_pool():
    """Union the five sources, de-duplicated on cfg_key, and hand out stable uids."""
    cfgs, sources, header = {}, defaultdict(list), None
    for name in POOL_SOURCES:
        path = source(name)
        if path is None:
            print(f"  WARNING pool source missing: social_nav_episode_{name}.json.gz")
            continue
        blob = json.load(gzip.open(path))
        if header is None:
            header = {k: v for k, v in blob.items() if k != "episodes"}
        for e in blob["episodes"]:
            k = cfg_key(e)
            cfgs.setdefault(k, e)
            sources[k].append(name)
    # uid is deterministic in the cfg_key ordering, so it survives a rebuild
    per_scene = defaultdict(int)
    for k in sorted(cfgs, key=lambda k: (k[0], str(k))):
        s = k[0].split(".")[0]
        cfgs[k] = copy.deepcopy(cfgs[k])
        pool = cfgs[k].setdefault("info", {}).setdefault("pool", {})
        pool["uid"] = f"{s}#{per_scene[s]}"
        pool["sources"] = sorted(set(sources[k]))
        pool.setdefault("label", "?")
        per_scene[s] += 1
    return cfgs, header


def _stats_by_ep(path):
    """results/*.json keys are "<scene>|<episode_id>|<eval_idx>"; bucket by episode."""
    if not os.path.exists(path):
        return {}
    out = defaultdict(list)
    for k, v in json.load(open(path)).items():
        out[k.split("|")[1]].append((v.get("social_nav_to_pos_success", 0) > 0.5,
                                     v.get("did_collide", 0) > 0.5,
                                     int(v.get("num_steps", 0))))
    return out


def classify(mk, cert):
    """The label tree of build_pool_v1.py:52-67,182-188, re-derived from stats.

    mk is the rule_markov runs; cert maps the other scripted modes to theirs.
    A label is a statement about WHICH controller family still certifies the
    episode, not about how hard it looks:
      sweet         teacher passes every eval and always_go does not
      trivial_*     both pass; _eff if the teacher is >=60 steps slower (headroom)
      challenge     teacher passes some evals but not all -- a FLAKY certificate
      fail_A/B/C    teacher never passes, but go / wait-family / yield does
      dead          nothing certifies it; excluded from every suite
    """
    allok = lambda r: bool(r) and all(x[0] for x in r)  # noqa: E731
    go = cert.get("always_go", [])
    if allok(mk):
        if not allok(go):
            return "sweet"
        gap = sum(x[2] for x in mk) / len(mk) - sum(x[2] for x in go) / len(go)
        return "trivial_eff" if gap >= 60 else "trivial_zero"
    if any(x[0] for x in mk):
        return "challenge"
    if allok(go):
        return "fail_A_go"
    if allok(cert.get("rule_wait", [])) or allok(cert.get("rule_waitgo", [])):
        return "fail_B_wait"
    if allok(cert.get("rule_yield", [])):
        return "fail_C_dither"
    return "dead"


def apply_cert(cfgs):
    """Fold the fresh certification of the four unlabelled configurations in.

    They are not new episodes: build_pool_v1.py had already rejected all four as
    `dead` and dropped their info.pool block, which is why they read as "?". The
    teacher has changed since (wait action, ego-invariant release), so the run is
    a re-test, and one of the four does come back alive.
    """
    src = source("UNLABELED4")
    mk_all = _stats_by_ep(f"{RESULTS}/stats_teacher_cert3_UNLABELED4.json")
    if not (mk_all and src):
        print("  note: no UNLABELED4 certification found; the 4 '?' stay '?'")
        return 0
    modes = ["always_go", "rule_wait", "rule_waitgo", "smart_wait", "rule_yield"]
    other = {m: _stats_by_ep(f"{RESULTS}/cert_UNLABELED4_{m}.json") for m in modes}

    for e in json.load(gzip.open(src))["episodes"]:
        eid, k = str(e["episode_id"]), cfg_key(e)
        mk = mk_all.get(eid, [])
        if not mk or k not in cfgs:
            continue
        cert = {m: other[m].get(eid, []) for m in modes}
        lab = classify(mk, cert)
        pool = cfgs[k]["info"]["pool"]
        was = pool.get("label", "?")
        pool["label"] = lab
        pool["cert"] = {"rule_markov": f"{sum(x[0] for x in mk)}/{len(mk)}",
                        **{m: ("pass" if v and all(x[0] for x in v) else "fail")
                           for m, v in cert.items()},
                        "note": "re-tested 2026-08-17; build_pool_v1 had marked it dead"}
        print(f"    {pool['uid']:26} {was} -> {lab}"
              + ("   REVIVED" if lab != "dead" else ""))
    return 1


# ------------------------------------------------------------------------- split

def pick(scenes, by_scene, target, cap=PER_SCENE_CAP):
    """Round-robin over scenes so the budget spreads across geometries.

    Pass k takes the k-th best configuration of every scene, so with cap=2 a scene
    only contributes a second episode after every scene has contributed one.
    """
    rank = {l: i for i, l in enumerate(LABEL_PRIORITY)}
    pools = {s: sorted(by_scene.get(s, []),
                       key=lambda e: (rank.get(label(e), 99), e["info"]["pool"]["uid"]))
             for s in scenes}
    out, k = [], 0
    while len(out) < target and k < cap:
        progressed = False
        for s in scenes:
            if len(out) >= target:
                break
            if len(pools[s]) > k:
                out.append(pools[s][k])
                progressed = True
        if not progressed:
            break
        k += 1
    return out


def plan(cfgs, mode):
    by_scene = defaultdict(list)
    for e in cfgs.values():
        if label(e) not in EXCLUDED:
            by_scene[scene_of(e)].append(e)

    eval_scenes = list(EVAL_BASE)
    if mode == "eval":
        eval_scenes.insert(0, FAILURE_SCENE)
    held = set(VAL_SCENES) | set(eval_scenes)
    train_scenes = [s for s in by_scene if s not in held]
    if mode == "train":
        train_scenes = [FAILURE_SCENE] + [s for s in train_scenes if s != FAILURE_SCENE]

    big = {"train": pick(train_scenes, by_scene, TARGET["train"]),
           "val": pick(VAL_SCENES, by_scene, TARGET["val"]),
           "eval": pick(eval_scenes, by_scene, TARGET["eval"])}

    uniq = lambda eps: list(dict.fromkeys(scene_of(e) for e in eps))  # noqa: E731
    small = {"train": pick(uniq(big["train"])[:SMALL_TRAIN_N], by_scene, SMALL_TARGET["train"]),
             "val": pick(uniq(big["val"])[:SMALL_VAL_N], by_scene, SMALL_TARGET["val"]),
             "eval": pick(uniq(big["eval"])[:SMALL_EVAL_N], by_scene, SMALL_TARGET["eval"])}

    for split, eps in big.items():
        for e in eps:
            e["info"]["pool"]["big_split"] = split
    for split, eps in small.items():
        for e in eps:
            e["info"]["pool"]["small_split"] = split
    for e in cfgs.values():
        e["info"]["pool"].setdefault("big_split", "")
        e["info"]["pool"].setdefault("small_split", "")
    return big, small


def check(big, small):
    """The two invariants. Loud failure beats a silently leaking benchmark."""
    bad = []
    names = list(big)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = {scene_of(e) for e in big[a]} & {scene_of(e) for e in big[b]}
            if shared:
                bad.append(f"big_{a} and big_{b} share scenes {sorted(shared)}")
    for split in small:
        esc = {scene_of(e) for e in small[split]} - {scene_of(e) for e in big[split]}
        if esc:
            bad.append(f"small_{split} uses scenes outside big_{split}: {sorted(esc)}")
    if bad:
        raise SystemExit("SPLIT INVARIANT VIOLATED:\n  " + "\n  ".join(bad))
    return True


# ------------------------------------------------------------------------ output

def compose(header, eps):
    d = dict(header)
    d["episodes"] = [dict(copy.deepcopy(e), episode_id=str(i)) for i, e in enumerate(eps)]
    return d


def write(name, blob):
    path = f"{DATA}/{name}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(blob, f)
    print(f"  wrote {name}.json.gz  ({len(blob['episodes'])} episodes)")


def legacy_blobs(cfgs):
    """The old HL train/eval sets, verbatim, plus a uid so they join with the pool."""
    out = {}
    for split, src in LEGACY.items():
        path = source(src)
        if path is None:
            print(f"  WARNING legacy source missing: social_nav_episode_{src}.json.gz")
            continue
        blob = json.load(gzip.open(path))
        for e in blob["episodes"]:
            k = cfg_key(e)
            pool = e.setdefault("info", {}).setdefault("pool", {})
            if k in cfgs:
                p = cfgs[k]["info"]["pool"]
                pool["uid"] = p["uid"]
                pool["label"] = p["label"]
            pool["legacy_split"] = split
        out[split] = blob
    return out


def summarize(name, eps):
    lab = defaultdict(int)
    for e in eps:
        lab[label(e)] += 1
    hard = sum(v for k, v in lab.items() if k in IMPERFECT)
    comp = " ".join(f"{k} {v}" for k, v in sorted(lab.items(), key=lambda x: -x[1]))
    return dict(n=len(eps), scenes=len({scene_of(e) for e in eps}),
                imperfect=hard, comp=comp)


def report(rows):
    print(f"  {'dataset':32} {'eps':>4} {'scenes':>7} {'imperf':>7}  labels")
    print("  " + "-" * 100)
    for name, s in rows:
        print(f"  {name:32} {s['n']:>4} {s['scenes']:>7} {s['imperfect']:>7}  {s['comp']}")


def write_doc(cfgs, big, small, legacy, mode, rows):
    L = ["# social-nav datasets",
         "",
         "Generated by `hrl_pipeline/data_tools/build_datasets.py` -- do not hand-edit.",
         "Every other episode file has been moved to `archive_20260817/`; nothing in",
         "this directory is a leftover.",
         "",
         "## The nine files", "",
         "| file | episodes | scenes | what it is |", "|---|---|---|---|"]
    blurb = {
        "socialnav_ALL_labeled": "catalogue of every distinct configuration, with labels and suite membership. Not for training.",
        "socialnav_big_train": "main suite, training split",
        "socialnav_big_val": "main suite, validation split -- pick the checkpoint here",
        "socialnav_big_eval": "main suite, evaluation split -- touch once, report this",
        "socialnav_small_train": "quick suite (1-2 h loop), training split",
        "socialnav_small_val": "quick suite, validation split",
        "socialnav_small_eval": "quick suite, evaluation split",
        "socialnav_legacy_hl_train": "what the EXISTING HL checkpoints were trained on. Kept verbatim for reproducibility; NOT scene-disjoint from its eval set.",
        "socialnav_legacy_hl_eval": "what the EXISTING HL checkpoints were evaluated on. Shares scenes with legacy_hl_train, so its numbers are seen-scene numbers.",
    }
    for name, s in rows:
        L.append(f"| `{name}.json.gz` | {s['n']} | {s['scenes']} | {blurb.get(name, '')} |")

    L += ["", "## Splits are scene-disjoint", "",
          "The split unit is the SCENE, never the episode: two episodes in one room",
          "share its geometry, so an episode-level split leaks. `big_train`, `big_val`",
          "and `big_eval` share no scene, and each `small_*` split draws only from the",
          "scenes of the matching `big_*` split -- the build script asserts both.",
          "",
          f"Teacher-failure allocation: `FAILURE_MODE={mode}` "
          f"(the failure-rich scene `{FAILURE_SCENE}` sits in `{mode}`).",
          f"`{ZERO_LEAK[0]}` and `{ZERO_LEAK[1]}` have never appeared in any training run.",
          "", "| split | scenes |", "|---|---|"]
    for k, eps in list(big.items()):
        L.append(f"| big_{k} | {', '.join(sorted({scene_of(e) for e in eps}))} |")
    for k, eps in list(small.items()):
        L.append(f"| small_{k} | {', '.join(sorted({scene_of(e) for e in eps}))} |")

    L += ["", "## Labels", "",
          "`info.pool.label` records what the rule-based teacher does on the episode.",
          "Teacher-imperfect episodes (`challenge`, `fail_*`) are the headroom a learned",
          "policy is supposed to claim, which is why the selector keeps them first.",
          "", "| label | meaning | count |", "|---|---|---|"]
    tally = defaultdict(int)
    for e in cfgs.values():
        tally[label(e)] += 1
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        L.append(f"| `{k}` | {LABEL_DOC.get(k, '')} | {v} |")

    L += ["", "## Per-episode index", "",
          "`info.pool.uid` is stable across every file, so an episode can be traced",
          "between suites. `episode_id` is a plain 0..N-1 index inside each file",
          "(downstream analysis parses `scene|episode_id|eval_idx` stats keys as int).",
          "", "| uid | label | big | small | legacy | sources |", "|---|---|---|---|---|---|"]
    leg = defaultdict(list)
    for split, blob in legacy.items():
        for e in blob["episodes"]:
            u = e.get("info", {}).get("pool", {}).get("uid")
            if u:
                leg[u].append(split)
    for e in sorted(cfgs.values(), key=lambda e: e["info"]["pool"]["uid"]):
        p = e["info"]["pool"]
        u = p["uid"]
        L.append(f"| `{u}` | {p['label']} | {p['big_split'] or '-'} | "
                 f"{p['small_split'] or '-'} | {'+'.join(sorted(set(leg[u]))) or '-'} | "
                 f"{', '.join(p['sources'])} |")

    L += ["", "## Rebuilding", "",
          "```bash", "docker exec -u root wxinyuan bash -lc '. activate habitat && \\",
          "  cd /habitat-lab && python hrl_pipeline/data_tools/build_datasets.py --build'",
          "```", ""]
    open(DOC, "w").write("\n".join(L))
    print(f"  wrote {os.path.basename(DOC)}")


def main():
    mode = "train" if "--mode" in sys.argv and "train" in sys.argv else "eval"
    build = "--build" in sys.argv

    cfgs, header = load_pool()
    print(f"pool: {len(cfgs)} distinct configurations over "
          f"{len({scene_of(e) for e in cfgs.values()})} scenes")
    apply_cert(cfgs)

    big, small = plan(cfgs, mode)
    check(big, small)

    allblob = compose(header, sorted(cfgs.values(), key=lambda e: e["info"]["pool"]["uid"]))
    legacy = legacy_blobs(cfgs)

    rows = [("socialnav_ALL_labeled", summarize("all", allblob["episodes"]))]
    rows += [(f"socialnav_big_{k}", summarize(k, big[k])) for k in ("train", "val", "eval")]
    rows += [(f"socialnav_small_{k}", summarize(k, small[k])) for k in ("train", "val", "eval")]
    rows += [(f"socialnav_legacy_hl_{k}", summarize(k, v["episodes"]))
             for k, v in legacy.items()]

    print(f"\nFAILURE_MODE={mode}  (scene {FAILURE_SCENE} -> {mode})\n")
    report(rows)
    tot = sum(len(v) for v in big.values())
    print(f"\n  big suite total {tot}   ratio "
          + " / ".join(f"{100*len(big[k])/tot:.0f}" for k in ("train", "val", "eval")))
    print("  invariants: scene-disjoint OK, small-inside-big OK")

    if not build:
        print("\n(plan only -- pass --build to write)")
        return

    print()
    write("socialnav_ALL_labeled", allblob)
    for k in ("train", "val", "eval"):
        write(f"socialnav_big_{k}", compose(header, big[k]))
    for k in ("train", "val", "eval"):
        write(f"socialnav_small_{k}", compose(header, small[k]))
    for k, blob in legacy.items():
        write(f"socialnav_legacy_hl_{k}", blob)
    write_doc(cfgs, big, small, legacy, mode, rows)


if __name__ == "__main__":
    main()
