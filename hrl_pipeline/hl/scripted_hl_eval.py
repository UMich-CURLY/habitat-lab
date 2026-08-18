"""Diagnostic: run eval with a SCRIPTED high-level policy through the real HRL machinery.

RULE_MODE env var:
  always_go     - always pick go_to_goal (tests: is the conflict forcing? skill nav quality)
  rule_backoff  - backoff when human close+in-front+facing-us, else go_to_goal
  rule_wait     - same trigger, but wait instead of backoff
  rule_yield    - clear-corridor yield on conflict; RELEASE to go_to_goal once the
                  human has passed the door (or is receding). Legacy M1 teacher.
  rule_markov   - the M2 teacher: PURE function of (7-dim obs incl.
                  human_approach_speed, prev HL action, hysteresis ticks). No
                  door geometry, no world pose, no monotonic recede counter.
                  Trigger tight+immediate, release loose+2-tick. Requires the
                  v2 config (include_approach_speed sensor). Logs additive
                  jsonl keys: approach / prev_action / yield_branch.
  smart_wait    - yield only while the human is approaching
  always_backoff- sanity (robot always yields)
  learned       - pass-through: do NOT override, just log the learned policy's choice

Every HL decision (features + chosen skill) is logged to
/habitat-lab/hrl_pipeline/hl_decisions_<mode>.jsonl for observability analysis.
DEBUG_PREV=1 additionally prints the harness prev_actions vs the teacher's own
tracked previous action at each decision (plumbing-semantics check).
"""
import gzip
import json
import math
import os
import sys

import numpy as np
import torch

MODE = os.environ.get("RULE_MODE", "rule_backoff")
DIST_T = float(os.environ.get("RULE_DIST", "3.0"))
DATASET = os.environ.get(
    "RULE_DATASET", "/habitat-lab/data/social_nav_episode_overfit_switch.json.gz"
)
EVALS = os.environ.get("RULE_EVALS", "8")
VIDEO = os.environ.get("RULE_VIDEO", "0") == "1"
CONFIG = os.environ.get(
    "RULE_CONFIG", "social_nav/social_nav_hierarchical_overfit.yaml"
)
EVAL_STATS = os.environ.get("EVAL_STATS", "")
# Both overridable so concurrent runs never clobber each other's outputs.
LOG_PATH = os.environ.get(
    "RULE_LOG", f"/habitat-lab/hrl_pipeline/hl_decisions_{MODE}.jsonl"
)
VIDEO_DIR = os.environ.get("RULE_VIDEO_DIR", "hrl_pipeline/video_" + MODE)
DEBUG_PREV = os.environ.get("DEBUG_PREV", "0") == "1"
# RULE_TRACE=<path>: per-STEP trajectory jsonl (the decision log above only
# has HL-decision resolution). One row per env step: robot localization, the
# 6/7-dim policy feat, and the currently-active skill index. Joins with the
# decision log via the shared num_steps counter. Human/goal world positions
# are reconstructed OFFLINE from feat polar coords (goal is static per
# episode, which self-calibrates the frame convention).
TRACE_PATH = os.environ.get("RULE_TRACE", "")

# --- rule_markov teacher: every threshold in one place. Deterministic, and
# deliberately RIGID (fixed absolute thresholds, no adaptivity): the teacher's
# structural failures are the learning headroom; only undefined behavior is a
# bug. Trigger is tight and immediate (safety); release is loose (hysteresis:
# a wider still-conflict region) and needs rel_ticks consecutive clear
# decisions (~1 s per tick at control_interval=30). ---
MARKOV_PARAMS = dict(
    trig_dist=DIST_T,          # d < trig_dist               (RULE_DIST env)
    trig_bearing=1.75,         # |human_bearing| <
    trig_rel_heading=math.pi / 2,  # |human_rel_heading| >   (facing us)
    trig_approach=0.15,        # human_approach_speed >      (human closing in)
    trig_ticks=1,
    # Release terms are EGO-INVARIANT only (d, approach). bearing was tried and
    # removed: the yield skill spins the robot while backing off, which swings
    # the human's bearing out of any release cone and fired the release while
    # the human was still mid-transit (ego-rotation pollution -- same family of
    # bug as the old rel_speed gate; collided on cleandev17 ep0-7).
    rel_dist=3.5,              # still-conflict while d <
    rel_approach=0.05,         # still-conflict while approach >
    # Never release with a body this close, moving or not: a firm human frozen
    # by its 0.8 m block gate reads approach=0 but stands ON the resume path
    # (releasing at 0.77 m turned-and-drove straight into it). Holding keeps
    # the episode a timeout instead of a collision; the human unfreezes and
    # finishes its transit once it can.
    rel_min_dist=1.0,
    rel_ticks=2,
    # Split the yield into two phases by HOW LONG we have been retreating:
    # after this many consecutive backoff decisions the robot has reached (or is
    # near) its yield pocket, the useful part of retreating is done, and holding
    # is the honest description of what it does next. Counted from the
    # prev-action sequence alone, so the student can reconstruct it -- no
    # privileged pocket/branch info, which would re-introduce the label aliasing
    # this teacher was rebuilt to remove.
    #
    # A distance split was tried first (`wait` whenever d < 1.5) and DEADLOCKED:
    # stopping merely because the human is near can leave the robot standing IN
    # the human's way; the firm human then freezes against it, so d never grows
    # and the release predicate (which needs d to grow) never fires. 2 of 3
    # sweet episodes timed out. "Human is close" != "I have cleared the way".
    wait_after_ticks=6,
)

_markov_state = {}  # env -> {"trig": n, "rel": n}
_prev_hl = {}       # env -> previous HL action (-1 at episode start)


def _markov_action(feat, i):
    """PURE function of (current obs, prev HL action, tick counters). feat must
    be 7-dim: feat[6] = human_approach_speed (the human's absolute velocity
    toward the robot; ego-motion invariant, so a parked human reads ~0 no
    matter how the robot moves -- this is what kills the go<->backoff limit
    cycle the old rel_speed gate produced)."""
    if len(feat) < 7:
        raise RuntimeError(
            "rule_markov needs the 7-dim policy state (set "
            "include_approach_speed: true in the benchmark; use the v2 config)"
        )
    p = MARKOV_PARAMS
    d, b, rh, approach = feat[0], feat[1], feat[2], feat[6]
    st = _markov_state.setdefault(i, {"trig": 0, "rel": 0, "boff": 0})
    yielding = _prev_hl.get(i, -1) in (0, 1)
    if not yielding:
        trig = (
            d < p["trig_dist"]
            and abs(b) < p["trig_bearing"]
            and abs(rh) > p["trig_rel_heading"]
            and approach > p["trig_approach"]
        )
        st["trig"] = st["trig"] + 1 if trig else 0
        st["rel"] = 0
        st["boff"] = 0
        return 0 if st["trig"] >= p["trig_ticks"] else 2
    # rel_min_dist only applies while still RETREATING. It exists to stop the
    # release from firing mid-reverse with a body right behind us -- but it
    # assumes the robot keeps opening the gap. Once we switch to `wait` nothing
    # moves: the human is blocked by us (its 0.8 m firm gate) and we are held,
    # so d freezes (measured: stuck at 0.99 for 900+ steps) and the gate can
    # never clear -- the only action that could satisfy it is the one it forbids.
    # In the wait phase the ego-invariant terms alone decide.
    still = (d < p["rel_dist"] and approach > p["rel_approach"]) or (
        d < p["rel_min_dist"] and st["boff"] < p["wait_after_ticks"]
    )
    st["rel"] = st["rel"] + 1 if not still else 0
    st["trig"] = 0
    if st["rel"] >= p["rel_ticks"]:
        return 2
    if st["boff"] >= p["wait_after_ticks"]:
        return 1  # retreat done -> hold in place (facing the goal)
    st["boff"] += 1
    return 0

from habitat_baselines.rl.hrl.hl.social_nav_neural_policy import (
    SocialNavNeuralHighLevelPolicy,
)

_orig = SocialNavNeuralHighLevelPolicy.get_next_skill

# --- Door geometry for the rule_yield release condition (human passed door) ---
# PER-EPISODE: a multi-scene dataset has a different door per episode, so the
# release predicate must use the CURRENT episode's door, not episode 0's (using
# ep0's door for every episode made rule_yield fire/misfire at random -> both
# false SWEETs and false unsolvs). num_environments=1 runs episodes
# sequentially; on each episode boundary we pick the door of the episode whose
# stored start_position is nearest the robot's spawn. Single-episode datasets
# reduce to the old behaviour (the only door is ep0's).
_DOOR = None            # fallback = first episode's door
_ALL_DOORS = []         # [(start_xz, (door_start, door_end, robot_start)), ...]
_cur_door = {}          # env index -> current episode's (ds, de, rstart)


def _load_door_geometry(path):
    global _DOOR, _ALL_DOORS
    try:
        with gzip.open(path) as f:
            d = json.load(f)
        _ALL_DOORS = []
        for ep in d["episodes"]:
            info = ep["info"]
            if "door_start" not in info or "door_end" not in info:
                continue
            door = (
                np.array(info["door_start"], dtype=np.float64),
                np.array(info["door_end"], dtype=np.float64),
                np.array(ep["start_position"], dtype=np.float64),
            )
            start_xz = np.array([ep["start_position"][0], ep["start_position"][2]], dtype=np.float64)
            _ALL_DOORS.append((start_xz, door))
        _DOOR = _ALL_DOORS[0][1] if _ALL_DOORS else None
    except Exception as e:
        print(f"[harness] could not load door geometry: {e}", flush=True)
        _DOOR = None


def _side_sign(ds, de, p):
    return 1.0 if (p[0] - ds[0]) * (de[2] - ds[2]) - (p[2] - ds[2]) * (de[0] - ds[0]) > 0 else -1.0


def _human_world_xz(feat, robot_xz, yaw):
    """Invert the SocialNavPolicyState polar (human_dist, human_bearing) in the
    robot frame back to a world (x, z). Frame: forward=(cos yaw, -sin yaw),
    lateral=(forward[1], -forward[0]) (see SocialNavPolicyStateSensor)."""
    hd, hb = feat[0], feat[1]
    forward = np.array([math.cos(yaw), -math.sin(yaw)])
    lateral = np.array([forward[1], -forward[0]])
    d = hd * math.cos(hb) * forward + hd * math.sin(hb) * lateral
    return robot_xz + d


# Per-env rule_yield state (latched human-passed-door + receding counter).
_yield_state = {}


def _rule_action(feat, i, robot_xz, yaw):
    human_dist, human_bearing, human_rel_heading, human_rel_speed = feat[:4]
    if MODE == "always_go":
        return 2
    if MODE == "always_backoff":
        return 0
    conflict = (
        human_dist < DIST_T
        and abs(human_bearing) < 1.75
        and abs(human_rel_heading) > math.pi / 2
        # Motion condition: only an APPROACHING human is a conflict. A human
        # PARKED at its goal 1.4 m away used to re-trigger yield forever (the
        # robot could never walk past it); stationary/receding humans are just
        # static obstacles to route by.
        and human_rel_speed > 0.1
    )
    if MODE == "smart_wait":
        if conflict and human_rel_speed > 0.15:
            return 1
        return 2
    if MODE in ("rule_yield", "rule_waitgo"):
        st = _yield_state.setdefault(i, {"passed": False, "recede": 0, "prev_d": None})
        # RELEASE: human has passed the door, or is receding for 3 decisions.
        passed = st["passed"]
        door = _cur_door.get(i, _DOOR)
        if not passed and door is not None and robot_xz is not None:
            ds, de, rstart = door
            hxz = _human_world_xz(feat, robot_xz, yaw)
            hp3 = np.array([hxz[0], 0.0, hxz[1]])
            robot_side = _side_sign(ds, de, rstart)
            # perp dist to door line
            a = np.array([ds[0], ds[2]]); b = np.array([de[0], de[2]])
            ab = b - a; nrm = np.linalg.norm(ab)
            perp = abs(np.cross(ab, np.array([hxz[0], hxz[1]]) - a)) / nrm if nrm > 1e-9 else 0.0
            if _side_sign(ds, de, hp3) == robot_side and perp > 0.3:
                st["passed"] = passed = True
        if st["prev_d"] is not None and human_dist > st["prev_d"] + 1e-3:
            st["recede"] += 1
        else:
            st["recede"] = 0
        st["prev_d"] = human_dist
        # RESUME only once the human is no longer a head-on conflict. In a
        # cross-through the human crosses the door onto the robot's side while
        # still directly IN FRONT of the robot -- resuming then drives straight
        # into it. Keep yielding until the human is past/receding (not conflict).
        if (passed and not conflict) or st["recede"] >= 3:
            return 2  # corridor clear -> go
        if conflict:
            # rule_yield: side-clear the corridor (backoff=0);
            # rule_waitgo: just stop and let the human pass first (wait=1).
            return 0 if MODE == "rule_yield" else 1
        return 2
    if conflict:
        return 0 if MODE == "rule_backoff" else 1
    return 2


def patched(
    self,
    observations,
    rnn_hidden_states,
    prev_actions,
    masks,
    should_choose_new_skill,
    deterministic=False,
    log_info=None,
):
    new_skills, new_skill_args, term, pad = _orig(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        should_choose_new_skill,
        deterministic,
        log_info,
    )
    feats = None
    for k in ("agent_0_social_nav_policy_state", "social_nav_policy_state"):
        if k in observations:
            v = observations[k]
            feats = v.detach().cpu().numpy() if torch.is_tensor(v) else v
            break
    loc = None
    for k in ("agent_0_localization_sensor", "localization_sensor"):
        if k in observations:
            v = observations[k]
            loc = v.detach().cpu().numpy() if torch.is_tensor(v) else v
            break
    num_steps = None
    if "num_steps" in observations:
        v = observations["num_steps"]
        num_steps = int(v.flatten()[0]) if torch.is_tensor(v) else int(np.array(v).flatten()[0])
    # Yield-cascade branch (last dim of the backoff sensor when include_branch
    # is on: widths 4 or 6). -1 = branch not available.
    branches = None
    for k in ("agent_0_backoff_waypoint_delta", "backoff_waypoint_delta"):
        if k in observations:
            v = observations[k]
            w = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            if w.ndim == 2 and w.shape[1] in (4, 6):
                branches = w[:, -1]
            break
    with open(LOG_PATH, "a") as f:
        for i in range(masks.shape[0]):
            # Reset per-env teacher state on episode boundary, and latch the
            # current episode's door by nearest stored start_position.
            if float(masks[i].item()) == 0.0:
                _yield_state.pop(i, None)
                _markov_state.pop(i, None)
                _prev_hl[i] = -1
                if loc is not None and _ALL_DOORS:
                    rxz = np.array([loc[i][0], loc[i][2]])
                    _cur_door[i] = min(
                        _ALL_DOORS, key=lambda t: float(np.sum((t[0] - rxz) ** 2))
                    )[1]
            if not bool(should_choose_new_skill[i]):
                continue
            fv = [float(x) for x in feats[i]]
            prev_a = int(_prev_hl.get(i, -1))
            branch = int(branches[i]) if branches is not None else -1
            if DEBUG_PREV:
                print(
                    f"[prev-check env={i}] harness_prev_actions="
                    f"{float(prev_actions[i, 0].item()):.1f} teacher_prev={prev_a}"
                    f" mask={float(masks[i].item()):.0f}",
                    flush=True,
                )
            robot_xz = np.array([loc[i][0], loc[i][2]]) if loc is not None else None
            yaw = float(loc[i][3]) if loc is not None else 0.0
            if MODE == "learned":
                a = int(round(float(pad.actions[i, 0].item())))
            else:
                if MODE == "rule_markov":
                    a = _markov_action(fv, i)
                else:
                    a = _rule_action(fv, i, robot_xz, yaw)
                skill_name = self._action_to_skill[a]
                new_skills[i] = self._skill_name_to_idx.get(skill_name, 0)
                new_skill_args[i] = {}
                pad.actions[i, 0] = float(a)
            _prev_hl[i] = a
            skill_name = self._action_to_skill.get(a, str(a))
            row = {
                # "feat" stays 6-dim forever (jsonl back-compat); the 7th dim
                # goes into its own additive key.
                "feat": fv[:6],
                "action": skill_name,
                "env": i,
                "mask": float(masks[i].item()),
                "num_steps": num_steps,
                "prev_action": prev_a,
                "yield_branch": branch,
            }
            if len(fv) > 6:
                row["approach"] = fv[6]
            f.write(json.dumps(row) + "\n")
    return new_skills, new_skill_args, term, pad


SocialNavNeuralHighLevelPolicy.get_next_skill = patched
_load_door_geometry(DATASET)

if TRACE_PATH:
    from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy

    _orig_act = HierarchicalPolicy.act
    _trace_meta_done = [False]

    def _traced_act(self, observations, rnn_hidden_states, prev_actions,
                    masks, deterministic=False):
        out = _orig_act(self, observations, rnn_hidden_states, prev_actions,
                        masks, deterministic)
        # Both agents' hierarchical policies share this class-level patch;
        # only trace the robot (its HL is the SocialNavNeural policy).
        if not isinstance(
            self._high_level_policy, SocialNavNeuralHighLevelPolicy
        ):
            return out
        feats = loc = None
        for k in ("agent_0_social_nav_policy_state", "social_nav_policy_state"):
            if k in observations:
                v = observations[k]
                feats = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                break
        for k in ("agent_0_localization_sensor", "localization_sensor"):
            if k in observations:
                v = observations[k]
                loc = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                break
        with open(TRACE_PATH, "a") as f:
            if not _trace_meta_done[0]:
                _trace_meta_done[0] = True
                f.write(json.dumps({
                    "meta": True,
                    "skill_names": getattr(self, "_skill_name_to_idx", None)
                    and {v: k for k, v in self._skill_name_to_idx.items()},
                    "obs_keys": sorted(observations.keys()),
                }) + "\n")
            for i in range(masks.shape[0]):
                sk = self._cur_skills[i]
                f.write(json.dumps({
                    "t": int(self._step_counter),
                    "env": i,
                    "mask": float(masks[i].item()),
                    "loc": [float(x) for x in loc[i]] if loc is not None else None,
                    "feat": [float(x) for x in feats[i]] if feats is not None else None,
                    "skill": int(sk.item() if torch.is_tensor(sk) else sk),
                }) + "\n")
        return out

    HierarchicalPolicy.act = _traced_act

if __name__ == "__main__":
    from habitat_baselines.run import main

    sys.argv = [
        "run.py",
        f"--config-name={CONFIG}",
        "habitat_baselines.evaluate=True",
        "habitat_baselines.eval.should_load_ckpt=False",
        "habitat_baselines.eval_ckpt_path_dir=checkpoints_social_nav_hrl/latest.pth",
        f"habitat_baselines.eval.video_option={['disk'] if VIDEO else []}",
        f"habitat_baselines.video_dir={VIDEO_DIR}",
        "habitat_baselines.num_environments=1",
        f"habitat_baselines.eval.evals_per_ep={EVALS}",
        f"habitat.dataset.data_path={DATASET}",
        "habitat.seed=7",
    ]
    if EVAL_STATS:
        sys.argv.append(
            f"habitat_baselines.eval.episode_stats_path={EVAL_STATS}"
        )
    _step_cap = os.environ.get("STEP_CAP", "")
    if _step_cap:
        # Cap via the TASK's natural episode length (not the evaluator's forced
        # reset, which slices long episodes into multiple re-evaluated segments
        # and skips episodes). This keeps one eval per episode, covering all.
        sys.argv.append(f"habitat.environment.max_episode_steps={_step_cap}")
    # Extra hydra overrides, space-separated. Mainly for RULE_MODE=learned:
    # RULE_EXTRA="+habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.pretrained_hl_weights=/path.pth"
    _extra = os.environ.get("RULE_EXTRA", "")
    if _extra:
        sys.argv.extend(_extra.split())
    main()
