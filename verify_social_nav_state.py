"""
Quick verification of the robot-frame SocialNavPolicyStateSensor inputs.

Run inside the habitat container:
    source activate habitat && cd /habitat-lab
    python verify_social_nav_state.py

Checks (no training / no checkpoint needed):
  1. human_dist / goal_dist returned by the sensor equal the true Euclidean
     XZ distances (frame-invariant magnitude).
  2. A target placed straight ahead of the robot has bearing ~= 0.
  3. Robot-frame invariance: rotating the robot in place leaves the distances
     unchanged and rotates the bearings by exactly minus the robot rotation.
  4. human_rel_speed is ~0 when nothing moves and becomes > 0 after the human
     moves between two sensor reads.

All angles printed in degrees.
"""
import math

import magnum as mn
import numpy as np

import habitat
from habitat_baselines.config.default import get_config

CFG = "social_nav/social_nav_hierarchical.yaml"
KEY = "agent_0_social_nav_policy_state"


def _read(sensor, env):
    """Re-read the sensor against the current sim state."""
    return np.asarray(
        sensor.get_observation(task=env.task, episode=env.current_episode),
        dtype=np.float64,
    )


def _ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    cfg = get_config(CFG)
    passed = True
    with habitat.Env(config=cfg.habitat) as env:
        env.reset()
        sensor = env.task.sensor_suite.sensors[KEY]
        robot = env.sim.get_agent_data(0).articulated_agent
        human = env.sim.get_agent_data(1).articulated_agent

        # ---- 1. distances match euclidean ----
        s = _read(sensor, env)
        r = np.array(robot.base_pos)
        h = np.array(human.base_pos)
        g = np.array(env.task.my_nav_to_info.robot_info.nav_goal_pos)
        eucl_h = math.hypot(r[0] - h[0], r[2] - h[2])
        eucl_g = math.hypot(r[0] - g[0], r[2] - g[2])
        print(f"\nsensor = {np.round(s, 3)}")
        print(f"human_dist={s[0]:.3f} (eucl {eucl_h:.3f}) | "
              f"goal_dist={s[4]:.3f} (eucl {eucl_g:.3f})")
        passed &= _ok("human_dist == euclidean", abs(s[0] - eucl_h) < 1e-2)
        passed &= _ok("goal_dist == euclidean", abs(s[4] - eucl_g) < 1e-2)

        # ---- 2. target straight ahead -> bearing ~ 0 ----
        # Place the human 2 m straight in front of the robot (along its forward
        # axis) and check the bearing is ~0.
        fwd = robot.base_transformation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )
        fxz = np.array([fwd[0], fwd[2]])
        fxz = fxz / (np.linalg.norm(fxz) + 1e-9)
        ahead = np.array([r[0] + 2.0 * fxz[0], r[1], r[2] + 2.0 * fxz[1]])
        human.base_pos = mn.Vector3(*[float(c) for c in ahead])
        s2 = _read(sensor, env)
        print(f"\nhuman placed straight ahead -> bearing={math.degrees(s2[1]):.2f} deg, "
              f"dist={s2[0]:.3f}")
        passed &= _ok("ahead bearing ~ 0 deg", abs(math.degrees(s2[1])) < 2.0)
        passed &= _ok("ahead dist ~ 2 m", abs(s2[0] - 2.0) < 0.05)

        # ---- 3. rotate robot in place: dist invariant, bearing shifts by -delta ----
        base_before = _read(sensor, env)
        delta = math.radians(40.0)
        robot.base_rot = robot.base_rot + delta
        base_after = _read(sensor, env)
        d_dist = abs(base_after[0] - base_before[0])
        # bearing should decrease by delta (target appears rotated the other way)
        d_bear = math.degrees(
            math.atan2(
                math.sin(base_after[1] - base_before[1]),
                math.cos(base_after[1] - base_before[1]),
            )
        )
        print(f"\nrotate robot +40 deg: d_human_dist={d_dist:.4f}, "
              f"d_bearing={d_bear:.2f} deg (expect ~ -40)")
        passed &= _ok("dist invariant under robot rotation", d_dist < 1e-2)
        passed &= _ok("bearing shifts by -rotation", abs(d_bear + 40.0) < 3.0)
        robot.base_rot = robot.base_rot - delta  # restore

        # ---- 4. relative speed ----
        _read(sensor, env)  # prime prev positions
        s_still = _read(sensor, env)
        hp = np.array(human.base_pos)
        human.base_pos = mn.Vector3(
            float(hp[0] + 0.3), float(hp[1]), float(hp[2])
        )
        s_move = _read(sensor, env)
        print(f"\nrel_speed still={s_still[3]:.3f}  after human moves={s_move[3]:.3f}")
        passed &= _ok("rel_speed ~0 when still", s_still[3] < 1e-3)
        passed &= _ok("rel_speed > 0 after human moves", s_move[3] > 1e-3)

    print(f"\n==== {'ALL PASS' if passed else 'SOME FAILED'} ====")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
