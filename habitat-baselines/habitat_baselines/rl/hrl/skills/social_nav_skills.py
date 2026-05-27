"""
Low-level skills for social navigation with Spot robot.

Three discrete skills:
  1. BackOffSkill:  Navigate toward goal via oracle nav to yield to the human;
                    terminates when distance to human >= 3.0 m or HL requests.
  2. WaitSkill:     Stay in place while rotating to face the robot goal;
                    terminates only on HL request.
  3. GoToGoalSkill: Navigate toward goal via oracle nav at full speed; terminates
                    when dist < 0.5 m.
"""

import logging
import math
from typing import Any, List

import numpy as np
import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat_baselines.rl.hrl.skills import SkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions

logger = logging.getLogger(__name__)


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


class SocialNavSkillBase(nn.Module, SkillPolicy):
    """
    Base for social nav skills that output oracle nav or velocity actions.
    Bypasses SkillPolicy.__init__ to avoid PDDL requirements.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        nn.Module.__init__(self)

        self._config = config
        self._batch_size = batch_size

        self.should_ignore_grip = getattr(config, "ignore_grip", True)
        self._apply_postconds = getattr(config, "apply_postconds", False)
        self._force_end_on_timeout = getattr(config, "force_end_on_timeout", False)
        self._max_skill_steps = getattr(config, "max_skill_steps", -1)
        self._pddl_ac_start = None
        self._cur_skill_step = 0
        self._cur_skill_args = [None] * batch_size
        self._raw_skill_args = [None] * batch_size
        self._pddl_problem = None
        self._last_num_steps = -1  # Track last num_steps to avoid duplicate termination

        self._full_ac_size = get_num_actions(action_space)

        # Action order for agent_0: base_velocity(2) + oracle_nav_action(1) + rearrange_stop(1)
        # oracle_nav_action is at index 2
        self._oracle_nav_ac_idx = 2

    @property
    def required_obs_keys(self) -> List[str]:
        return []

    def on_enter(self, skill_args, batch_indices, observations,
                 rnn_hidden_states, prev_actions, skill_name=None):
        return rnn_hidden_states, prev_actions

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False, **kwargs):
        """Call _internal_act and return PolicyActionData."""
        return self._internal_act(observations, rnn_hidden_states, prev_actions, masks,
                                  cur_batch_idx, deterministic)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()

        self._cur_skill_step += 1

        # Rule-based skill switching: terminate if human is too close or approaching
        for i in range(batch_size):
            if not call_hl[i]:  # Only check if HL hasn't already requested termination
                dist_to_human = None
                vel_towards_human = None

                # Calculate distance to human
                for key in ["human_relative_position", "human_pos_relative"]:
                    if key in observations:
                        hpos = observations[key]
                        if isinstance(hpos, np.ndarray):
                            hpos = torch.from_numpy(hpos).float()
                        # Get distance from relative position [hpx, hpy]
                        if hpos.shape[0] > i and hpos.shape[1] >= 2:
                            dist = torch.sqrt(hpos[i, 0]**2 + hpos[i, 1]**2).item()
                            dist_to_human = dist
                        break

                # Calculate closing speed
                for key in ["human_relative_velocity", "human_vel_relative"]:
                    if key in observations:
                        hvel = observations[key]
                        if isinstance(hvel, np.ndarray):
                            hvel = torch.from_numpy(hvel).float()
                        # Get velocity components [velx, vely]
                        if hvel.shape[0] > i and hvel.shape[1] >= 2:
                            velx = hvel[i, 0].item()
                            vely = hvel[i, 1].item()
                            # Get relative position to compute dot product
                            for pos_key in ["human_relative_position", "human_pos_relative"]:
                                if pos_key in observations:
                                    hpos = observations[pos_key]
                                    if isinstance(hpos, np.ndarray):
                                        hpos = torch.from_numpy(hpos).float()
                                    if hpos.shape[0] > i and hpos.shape[1] >= 2:
                                        hpx = hpos[i, 0].item()
                                        hpy = hpos[i, 1].item()
                                        dist_sq = hpx**2 + hpy**2
                                        if dist_sq > 0:
                                            vel_towards_human = (velx * hpx + vely * hpy) / (dist_sq**0.5)
                                    break
                        break

                # Apply rule-based switching
                should_switch = False
                if dist_to_human is not None and dist_to_human < 0.5:
                    should_switch = True
                elif vel_towards_human is not None and vel_towards_human > 0.5:
                    should_switch = True

                if should_switch:
                    call_hl[i] = True
                    logger.info(f"[SocialNavSkillBase] Rule-based switch at step {self._cur_skill_step}: dist={dist_to_human:.2f}m, closing_speed={vel_towards_human:.2f}m/s")

        # Fallback: also terminate after max steps if set
        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            logger.warning(f"[SocialNavSkillBase] Terminating after max steps {self._cur_skill_step}")
            self._cur_skill_step = 0
        elif call_hl.any():
            # Reset step counter when switching
            self._cur_skill_step = 0

        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions

    @classmethod
    def from_config(cls, config, observation_space, action_space, num_envs, full_config):
        """Create skill from config."""
        return cls(config, action_space, num_envs)


class BackOffSkill(SocialNavSkillBase):
    """
    Back away from humans by retracing the trajectory backward.
    Moves toward increasingly earlier waypoints in the collected trajectory,
    one waypoint per timestep. Terminates when reaching trajectory start or HL requests.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        super().__init__(config, action_space, batch_size, **kwargs)
        # Waypoint tracking: index into reversed trajectory
        self._backoff_waypoint_idx = [0] * batch_size
        self._skill_entered = [False] * batch_size  # Track if skill has been entered before
        # Distance threshold to consider waypoint "reached"
        self._waypoint_threshold = 0.03  # meters (much smaller since waypoints are ~0.06-0.1m apart)
        # Backoff velocity magnitude
        self._backoff_velocity = 0.5  # m/s

    def on_enter(self, skill_args, batch_indices, observations,
                 rnn_hidden_states, prev_actions, skill_name=None):
        """Reset waypoint index when skill starts (only once)."""
        for batch_idx in batch_indices:
            if batch_idx < len(self._backoff_waypoint_idx):
                # Only reset if this is the first time entering the skill
                if not self._skill_entered[batch_idx]:
                    self._backoff_waypoint_idx[batch_idx] = 1  # Start at 1 (one step back), not 0 (current position)
                    self._skill_entered[batch_idx] = True
        return rnn_hidden_states, prev_actions

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size)

        # Get trajectory buffer from observations
        trajectory = observations.get("trajectory_buffer", []) if isinstance(observations, dict) else []

        # Get num_steps for logging
        num_steps = 0
        if isinstance(observations, dict) and "num_steps" in observations:
            num_steps_obs = observations["num_steps"]
            if isinstance(num_steps_obs, torch.Tensor):
                num_steps = int(num_steps_obs[0].item()) if num_steps_obs.dim() > 0 else int(num_steps_obs.item())
            else:
                num_steps = int(num_steps_obs[0]) if hasattr(num_steps_obs, '__getitem__') else int(num_steps_obs)


        # If we have trajectory, follow it backward; otherwise back up slowly
        if trajectory and len(trajectory) > 1:
            for i in range(batch_size):
                waypoint_idx = self._backoff_waypoint_idx[i]

                # Get target position from reversed trajectory
                # waypoint_idx=1 is one step back, waypoint_idx=2 is two steps back, etc.
                # trajectory[-1] is current position, trajectory[-2] is one step back
                if waypoint_idx < len(trajectory):
                    target_idx = len(trajectory) - 1 - waypoint_idx  # waypoint_idx=1 → target_idx=len-1-1
                    if target_idx >= 0 and target_idx < len(trajectory):
                        target_pos = np.array(trajectory[target_idx][:2])
                        # Estimate current position as last trajectory position
                        curr_pos = np.array(trajectory[-1][:2]) if trajectory else np.array([0.0, 0.0])

                        # Compute direction to target
                        diff = target_pos - curr_pos
                        dist = np.linalg.norm(diff)

                        if dist > self._waypoint_threshold:
                            # Move toward waypoint
                            direction = diff / (dist + 1e-6)
                            action[i, 0] = direction[0] * self._backoff_velocity
                            action[i, 1] = direction[1] * self._backoff_velocity
                        else:
                            # Reached waypoint, move to next one (further back)
                            self._backoff_waypoint_idx[i] += 1
                            action[i, 0] = 0.0
                            action[i, 1] = 0.0
                    else:
                        # Past start of trajectory, stop
                        action[i, 0] = 0.0
                        action[i, 1] = 0.0
                else:
                    # Finished retracing all waypoints
                    action[i, 0] = 0.0
                    action[i, 1] = 0.0
        else:
            # No trajectory yet, back up slowly
            for i in range(batch_size):
                action[i, 0] = -self._backoff_velocity
                action[i, 1] = 0.0

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()

        # Terminate on HL request or if skill exceeded max steps
        if self._max_skill_steps > 0:
            self._cur_skill_step += 1
            if self._cur_skill_step >= self._max_skill_steps:
                call_hl[:] = True
                self._cur_skill_step = 0
                # Reset waypoint tracking for next backoff
                self._backoff_waypoint_idx = [0] * batch_size

        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions


class WaitSkill(SocialNavSkillBase):
    """
    Stay in place while rotating to face the robot goal.
    Terminates only when the high-level policy requests it.
    """

    @property
    def required_obs_keys(self) -> List[str]:
        return ["agent_0_base_pos", "robot_goal_pos", "agent_0_orientation"]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size)

        # Get num_steps for logging
        num_steps = 0
        if isinstance(observations, dict) and "num_steps" in observations:
            num_steps_obs = observations["num_steps"]
            if isinstance(num_steps_obs, torch.Tensor):
                num_steps = int(num_steps_obs[0].item()) if num_steps_obs.dim() > 0 else int(num_steps_obs.item())
            else:
                num_steps = int(num_steps_obs[0]) if hasattr(num_steps_obs, '__getitem__') else int(num_steps_obs)

        print(f"[WaitSkill] EXECUTING at num_steps={num_steps}")

        if ("agent_0_base_pos" in observations
                and "robot_goal_pos" in observations
                and "agent_0_orientation" in observations):
            robot_pos = _to_tensor(observations["agent_0_base_pos"])   # [B, 3]
            goal_pos  = _to_tensor(observations["robot_goal_pos"])     # [B, 2 or 3]
            orient    = _to_tensor(observations["agent_0_orientation"]) # [B, ≥3]
            robot_yaw = orient[:, 2]                                    # [B]

            dx = goal_pos[:, 0] - robot_pos[:, 0]
            dy = goal_pos[:, 1] - robot_pos[:, 1]
            desired_yaw = torch.atan2(dy, dx)
            yaw_err = desired_yaw - robot_yaw
            # Wrap to [-π, π]
            yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
            # Proportional angular control, clipped to [-1, 1]
            ang_vel = torch.clamp(yaw_err / math.pi, -1.0, 1.0).cpu()
            action[:, 1] = ang_vel  # action[:, 1] = angular velocity

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)


class GoToGoalSkill(SocialNavSkillBase):
    """
    Navigate toward the robot's goal using oracle nav.
    """

    @property
    def required_obs_keys(self) -> List[str]:
        return []

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size)

        # Get num_steps for logging
        num_steps = 0
        if isinstance(observations, dict) and "num_steps" in observations:
            num_steps_obs = observations["num_steps"]
            if isinstance(num_steps_obs, torch.Tensor):
                num_steps = int(num_steps_obs[0].item()) if num_steps_obs.dim() > 0 else int(num_steps_obs.item())
            else:
                num_steps = int(num_steps_obs[0]) if hasattr(num_steps_obs, '__getitem__') else int(num_steps_obs)

        if num_steps == 1 and isinstance(observations, dict):
            print(f"[GoToGoalSkill] obs_keys={sorted(observations.keys())}")


        print(f"[GoToGoalSkill] EXECUTING at num_steps={num_steps}")

        # Navigate using oracle_nav_action to TARGET_robot_0_goal (alphabetical idx 0 → 1-indexed 1.0)
        action[:, self._oracle_nav_ac_idx] = 1.0

        if isinstance(observations, dict):
            for key in observations:
                if "goal" in key.lower() or "compass" in key.lower():
                    rho = _to_tensor(observations[key]).view(-1)[0].item()
                    finished = observations.get("has_finished_oracle_nav")
                    print(f"[GoToGoalSkill] {key}[0]={rho:.3f}  has_finished_oracle_nav={finished}")
                    break

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()

        self._cur_skill_step += 1

        # Request termination if human is detected (will trigger HL policy to decide next skill)
        for i in range(batch_size):
            if not call_hl[i]:
                if isinstance(observations, dict) and "humanoid_detector_sensor" in observations:
                    detector = _to_tensor(observations["humanoid_detector_sensor"])
                    if detector.shape[0] > i and detector[i, 0].item() > 0.5:
                        # Human detected - request termination to trigger HL policy
                        call_hl[i] = True

        # Also terminate on HL request or if max steps reached
        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            self._cur_skill_step = 0
        elif call_hl.any():
            self._cur_skill_step = 0

        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions
