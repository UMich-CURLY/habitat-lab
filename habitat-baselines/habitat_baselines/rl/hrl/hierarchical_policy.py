import os.path as osp
from typing import Dict

import gym.spaces as spaces
import numpy as np
import torch

from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl import  HighLevelPolicy
from habitat_baselines.rl.hrl.hl.cycling_policy import (  # noqa: F401.
    CyclingHighLevelPolicy,
)
from habitat_baselines.rl.hrl.hl.fixed_policy import (  # noqa: F401.
    FixedHighLevelPolicy,
)
from habitat_baselines.rl.hrl.hl.social_nav_neural_policy import (  # noqa: F401.
    SocialNavNeuralHighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from habitat_baselines.rl.hrl.skills.social_nav_skills import (  # noqa: F401.
    BackOffSkill,
    WaitSkill,
    GoToGoalSkill,
)
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions


def _to_tensor(x):
    """Convert numpy array or other types to tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int,
        orig_action_space: spaces.Space = None,
        agent_name: str = None,
        pddl_problem = None,
    ):
        super().__init__(action_space)

        self._action_space = action_space
        self._num_envs: int = num_envs
        self._envs = None  # Will be set by trainer after initialization

        # Extract agent_name from config if not explicitly provided
        if agent_name is None and hasattr(config, 'agent_name'):
            agent_name = config.agent_name
        self._agent_name = agent_name

        # Store pddl_problem for fixed policies
        self._pddl_problem = pddl_problem

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}

        # Create PDDL problem so skills can look up entities
        self._pddl = None
        try:
            self._pddl = self._create_pddl(full_config)
        except Exception as e:
            baselines_logger.warning(f"Could not create PDDL problem: {e}")

        # Load skills from config.defined_skills
        for i, (skill_id, skill_config) in enumerate(
            config.defined_skills.items()
        ):
            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config, observation_space, action_space, self._num_envs, full_config,
            )
            self._skills[i] = skill_policy
            self._name_to_idx[skill_id] = i

            # Set PDDL problem for skills that need it (e.g., OracleNavPolicy)
            if self._pddl is not None:
                try:
                    skill_policy.set_pddl_problem(self._pddl)
                except Exception as e:
                    baselines_logger.warning(f"Could not set PDDL problem for skill {skill_id}: {e}")

        self._call_high_level: torch.Tensor = torch.ones(
            self._num_envs, dtype=torch.bool
        )
        self._cur_skills: torch.Tensor = torch.zeros(self._num_envs)

        self._step_counter = 0  # Track environment steps for skill cycling

        # Trajectory tracking for BackOffSkill - estimate positions from goal distance
        self._trajectory_buffer_size = 120
        self._trajectory_buffer = {i: [] for i in range(self._num_envs)}
        self._last_goal_dist = {i: 0.0 for i in range(self._num_envs)}
        self._last_pos_estimate = {i: np.array([0.0, 0.0]) for i in range(self._num_envs)}
        self._backoff_active = {i: False for i in range(self._num_envs)}  # Track if backoff skill is active; freeze trajectory when backoff starts

        # Try to get PDDL problem from full_config for fixed policies
        pddl_problem_for_hl = None
        if hasattr(full_config, 'task') and hasattr(full_config.task, 'pddl_problem'):
            pddl_problem_for_hl = full_config.task.pddl_problem

        high_level_cls = eval(config.high_level_policy.name)

        # Build PDDL action name to skill name mapping for FixedHighLevelPolicy
        pddl_action_name_to_skill_name = {}
        if hasattr(config, 'defined_skills'):
            for skill_name, skill_config in config.defined_skills.items():
                if hasattr(skill_config, 'pddl_action_names') and skill_config.pddl_action_names:
                    for pddl_action_name in skill_config.pddl_action_names:
                        pddl_action_name_to_skill_name[pddl_action_name] = skill_name

        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            pddl_problem_for_hl,  # pddl_problem positional arg
            num_envs,
            self._name_to_idx,
            observation_space,
            action_space,
            aux_loss_config=None,
            agent_name=agent_name,
            pddl_action_name_to_skill_name=pddl_action_name_to_skill_name,
        )
        
        # Find STOP action index if action_space is Dict (manipulation tasks)
        # For navigation-only tasks (like social nav), action_space is Box, so skip this
        ### TRIBHI CHECK ACTION SPACE!!!! 
        self._stop_action_idx = 0
        if isinstance(action_space, spaces.Dict):
            found = False
            for k in action_space:
                if k == "REARRANGE_STOP":
                    found = True
                    break
                self._stop_action_idx += get_num_actions(action_space[k])
            if not found:
                raise ValueError(f"Could not find STOP action in {action_space}")
        # For Box action spaces (social nav), we don't have a STOP action

    def _create_pddl(self, full_config):
        """Creates PDDL problem from the task config YAML files."""
        task_spec_file = osp.join(
            full_config.habitat.task.task_spec_base_path,
            full_config.habitat.task.task_spec + ".yaml",
        )
        return PddlProblem(
            full_config.habitat.task.pddl_domain_def,
            task_spec_file,
            read_config=False,
        )

    def eval(self):
        pass

    def set_envs(self, envs):
        """Set the environment reference so skills can access task.actions"""
        self._envs = envs
        # Also propagate to skills that need it
        for skill in self._skills.values():
            if hasattr(skill, 'set_envs'):
                skill.set_envs(envs)

    @property
    def num_recurrent_layers(self):
        # Use high-level policy's RNN layers
        if hasattr(self._high_level_policy, 'num_recurrent_layers'):
            return self._high_level_policy.num_recurrent_layers
        # Fallback to first skill
        if len(self._skills) > 0:
            return self._skills[0].num_recurrent_layers
        return 0

    @property
    def hidden_state_shape(self):
        """Return the hidden state shape from the high-level policy."""
        if hasattr(self._high_level_policy, 'hidden_state_shape'):
            return self._high_level_policy.hidden_state_shape
        # Fallback for skills without RNN
        return (self.num_recurrent_layers, self.recurrent_hidden_size)

    @property
    def hidden_state_shape_lens(self):
        """Return the hidden state shape lens from the high-level policy."""
        if hasattr(self._high_level_policy, 'hidden_state_shape_lens'):
            return self._high_level_policy.hidden_state_shape_lens
        return [self.recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self):
        """Return the recurrent hidden size from the high-level policy."""
        if hasattr(self._high_level_policy, 'recurrent_hidden_size'):
            return self._high_level_policy.recurrent_hidden_size
        return 0

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self._skills[0].parameters()

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)
        self._high_level_policy.to(device)
        self._call_high_level = self._call_high_level.to(device)
        self._cur_skills = self._cur_skills.to(device)
        return self

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        # Increment step counter each environment step
        self._step_counter += 1

        # Pass step counter to high-level policy for testing/debugging
        if hasattr(self._high_level_policy, '_steps_since_start'):
            self._high_level_policy._steps_since_start = self._step_counter

        self._high_level_policy.apply_mask(masks)
        use_device = prev_actions.device

        # Use actual batch size from masks, not self._num_envs which may differ
        # during evaluation (eval may use fewer envs than training config).
        actual_batch_size = masks.shape[0]


        batched_observations = [
            {k: v[batch_idx].unsqueeze(0) for k, v in observations.items()}
            for batch_idx in range(actual_batch_size)
        ]

        # Track trajectory by recording position estimated from compass bearing + distance
        # goal_to_agent_gps_compass = [bearing_to_goal, distance_to_goal]
        # We estimate agent position relative to goal and record it as waypoint
        for batch_idx in range(actual_batch_size):
            # Detect when backoff skill is active (skill_idx == 0) and freeze trajectory
            current_skill_idx = int(self._cur_skills[batch_idx].item()) if isinstance(self._cur_skills[batch_idx], torch.Tensor) else int(self._cur_skills[batch_idx])
            if current_skill_idx == 0:  # BackOffSkill - freeze trajectory
                self._backoff_active[batch_idx] = True
            elif current_skill_idx == 2:  # GoToGoalSkill - enable trajectory recording
                self._backoff_active[batch_idx] = False

            # Only record trajectory if backoff is NOT active (we're in GoToGoalSkill)
            if not self._backoff_active[batch_idx] and "goal_to_agent_gps_compass" in batched_observations[batch_idx]:
                compass_obs = _to_tensor(batched_observations[batch_idx]["goal_to_agent_gps_compass"])

                # compass_obs should be [bearing_angle, distance]
                if compass_obs.numel() >= 2:
                    bearing = compass_obs.view(-1)[0].item()  # Angle to goal
                    distance = compass_obs.view(-1)[1].item()  # Distance to goal

                    # Estimate agent position: agent is at distance D away from goal in direction opposite to bearing
                    # Goal is at bearing θ, distance d → agent is at [-d*cos(θ), -d*sin(θ)]
                    import math
                    agent_x = -distance * math.cos(bearing)
                    agent_y = -distance * math.sin(bearing)

                    # Record this as a waypoint every 3 steps to avoid noise
                    if self._step_counter % 3 == 0:
                        if len(self._trajectory_buffer[batch_idx]) >= self._trajectory_buffer_size:
                            self._trajectory_buffer[batch_idx].pop(0)
                        waypoint = np.array([agent_x, agent_y])
                        self._trajectory_buffer[batch_idx].append(waypoint.copy())

            # Add trajectory buffer to observations for BackOffSkill
            batched_observations[batch_idx]["trajectory_buffer"] = self._trajectory_buffer[batch_idx].copy()

        # Add num_steps to observations for skill cycling
        for batch_idx in range(actual_batch_size):
            batched_observations[batch_idx]["num_steps"] = torch.tensor([self._step_counter], dtype=torch.long)


        # Ensure rnn_hidden_states has correct shape [num_layers, batch, hidden]
        # MultiPolicy.split() creates [batch, num_layers, hidden], we need [num_layers, batch, hidden_per_agent]
        num_layers = self.num_recurrent_layers
        hidden_size = self.recurrent_hidden_size
        
        # Track original input shape to transpose back before returning
        original_rnn_hidden_states = rnn_hidden_states
        did_transpose = False
        
        # Special handling for 0-layer policies (non-RNN)
        if num_layers == 0:
            # For 0-layer policies, reshape from [batch, max_layers, 0] to [batch, 0, 0]
            # MultiPolicy gives us [batch, shared_max_layers, 0] but we need [batch, 0, 0]
            batched_rnn_hidden_states = rnn_hidden_states.new_zeros((actual_batch_size, 0, 0))
        else:
            if rnn_hidden_states.dim() == 3:
                # Check if it's [batch, num_layers, hidden] format (from MultiPolicy.split)
                if rnn_hidden_states.shape[1] == num_layers and rnn_hidden_states.shape[0] == actual_batch_size:
                    # Transpose to [num_layers, batch, hidden]
                    rnn_hidden_states = rnn_hidden_states.transpose(0, 1)
                    did_transpose = True
                elif rnn_hidden_states.shape[0] == num_layers and rnn_hidden_states.shape[1] == actual_batch_size:
                    # Already in [num_layers, batch, hidden] format
                    pass
                else:
                    # Try to infer - if first dim is small (num_layers), keep as is
                    if rnn_hidden_states.shape[0] <= 4:
                        # Likely already [num_layers, batch, hidden]
                        pass
                    else:
                        # Likely [batch, num_layers, hidden]
                        rnn_hidden_states = rnn_hidden_states.transpose(0, 1)
                        did_transpose = True
            elif rnn_hidden_states.dim() == 2:
                # Shape is [batch, hidden], shouldn't happen but handle it
                batch_size, hidden_dim = rnn_hidden_states.shape
                if hidden_dim == hidden_size * num_layers:
                    # Reshape to [batch, num_layers, hidden_per_layer]
                    hidden_per_layer = hidden_dim // num_layers
                    rnn_hidden_states = rnn_hidden_states.view(batch_size, num_layers, hidden_per_layer)
                    rnn_hidden_states = rnn_hidden_states.transpose(0, 1)
                    did_transpose = True
            
            batched_rnn_hidden_states = rnn_hidden_states
        
        batched_prev_actions = prev_actions.unsqueeze(1)
        batched_masks = masks.unsqueeze(1)

        batched_bad_should_terminate = torch.zeros(
            actual_batch_size, device=use_device, dtype=torch.bool
        )

        # Validate tensor shapes before processing
        assert batched_rnn_hidden_states.dim() == 3, f"rnn_hidden_states must be 3D, got shape {batched_rnn_hidden_states.shape}"
        if num_layers > 0:
            assert batched_rnn_hidden_states.shape[0] == num_layers, \
                f"First dim should be num_layers={num_layers}, got {batched_rnn_hidden_states.shape[0]}"
            assert batched_rnn_hidden_states.shape[1] == actual_batch_size, \
                f"Second dim should be batch={actual_batch_size}, got {batched_rnn_hidden_states.shape[1]}"
            assert batched_rnn_hidden_states.shape[2] == hidden_size, \
                f"Third dim should be hidden_size={hidden_size}, got {batched_rnn_hidden_states.shape[2]}"
        else:
            # For 0-layer policies: [batch, 0, 0]
            assert batched_rnn_hidden_states.shape == (actual_batch_size, 0, 0), \
                f"For 0-layer policy, expected shape ({actual_batch_size}, 0, 0), got {batched_rnn_hidden_states.shape}"

        # Resize per-env state tensors if actual_batch_size differs (e.g., eval vs train).
        if self._call_high_level.shape[0] != actual_batch_size:
            self._call_high_level = torch.ones(actual_batch_size, dtype=torch.bool, device=use_device)
            self._cur_skills = torch.zeros(actual_batch_size, dtype=torch.long, device=use_device)
        # Ensure tensors are on the correct device (may have been moved by .to()).
        self._call_high_level = self._call_high_level.to(use_device)
        self._cur_skills = self._cur_skills.to(use_device)

        # Check if skills should terminate.
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            if masks[batch_idx] == 0.0:
                # Don't check if the skill is done if the episode ended.
                continue

            skill = self._skills.get(int(skill_idx.item()))
            if skill is None:
                continue

            # Prepare hidden states for should_terminate check - handle 0-layer policies
            if num_layers > 0:
                skill_hs = batched_rnn_hidden_states[:, batch_idx:batch_idx+1, :]
            else:
                skill_hs = batched_rnn_hidden_states.new_zeros((0, 1, 0))

            try:
                call_hl_i, bad_term_i, _ = skill.should_terminate(
                    observations=batched_observations[batch_idx],
                    rnn_hidden_states=skill_hs,
                    prev_actions=batched_prev_actions[batch_idx:batch_idx+1],
                    masks=batched_masks[batch_idx:batch_idx+1],
                    actions=batched_prev_actions[batch_idx:batch_idx+1],
                    hl_wants_skill_term=torch.zeros(1, dtype=torch.bool),
                    batch_idx=[batch_idx],
                    skill_name=[None],
                    log_info=[{}],
                )
                batched_bad_should_terminate[batch_idx] = bad_term_i.any()
                self._call_high_level[batch_idx] = call_hl_i.any()
            except Exception as e:
                import traceback
                print(f"[HRL] ERROR in skill {skill_idx} should_terminate() for batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

        # Always call high-level if the episode is over.
        self._call_high_level = self._call_high_level | (~masks).view(-1)

        # Always call high-level at step 20 to verify skill switching works

        # Rule-based: call high-level to trigger backoff if human is detected
        if isinstance(observations, dict):
            # Look for humanoid_detector_sensor with potential agent prefix
            detector_key = None
            for key in observations.keys():
                if "humanoid_detector_sensor" in key:
                    detector_key = key
                    break

            if detector_key is not None:
                detector = observations[detector_key]
                if isinstance(detector, torch.Tensor):
                    detector = detector.cpu()
                elif isinstance(detector, np.ndarray):
                    detector = torch.from_numpy(detector).float()

                for i in range(min(detector.shape[0], self._call_high_level.shape[0])):
                    if detector.shape[0] > i and detector[i, 0].item() > 0.5:  # Human detected
                        self._call_high_level[i] = True

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(
            actual_batch_size, device=use_device, dtype=torch.bool
        )
        if self._call_high_level.sum() > 0:
            # Prepare observations for HL policy with num_steps included
            hl_observations = dict(observations)
            hl_observations["num_steps"] = torch.tensor([self._step_counter], dtype=torch.long)

            (
                new_skills,
                new_skill_args,
                hl_terminate,
                _,  # PolicyActionData from high-level policy
            ) = self._high_level_policy.get_next_skill(
                hl_observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
                deterministic,
                [],  # log_info
            )

            # Get indices where we need to call high-level policy
            call_hl_mask = self._call_high_level.cpu() if isinstance(self._call_high_level, torch.Tensor) else self._call_high_level
            call_hl_indices = torch.nonzero(call_hl_mask, as_tuple=False).squeeze(-1)
            
            if len(call_hl_indices) > 0:
                # Ensure call_hl_indices is 1D
                if call_hl_indices.dim() == 0:
                    call_hl_indices = call_hl_indices.unsqueeze(0)
                
                for batch_idx_tensor in call_hl_indices:
                    batch_idx = int(batch_idx_tensor.item())
                    
                    # Ensure batch_idx is valid
                    if batch_idx < 0 or batch_idx >= actual_batch_size:
                        continue
                    
                    skill_idx_val = new_skills[batch_idx]
                    skill_idx = int(skill_idx_val.item()) if isinstance(skill_idx_val, torch.Tensor) else int(skill_idx_val)

                    if skill_idx not in self._skills:
                        continue

                    skill = self._skills[skill_idx]

                    try:
                        # Prepare hidden states for skill - handle 0-layer policies
                        if num_layers > 0:
                            skill_hidden_states = batched_rnn_hidden_states[:, batch_idx:batch_idx+1, :]
                        else:
                            skill_hidden_states = batched_rnn_hidden_states.new_zeros((0, 1, 0))
                        
                        returned_hidden, returned_prev_action = skill.on_enter(
                            [new_skill_args[batch_idx]],
                            [batch_idx],
                            batched_observations[batch_idx],
                            skill_hidden_states,
                            batched_prev_actions[batch_idx:batch_idx+1],
                            skill_name=None,
                        )
                        # on_enter returns hidden state, update it
                        if num_layers > 0 and returned_hidden is not None and isinstance(returned_hidden, torch.Tensor):
                            if returned_hidden.shape == skill_hidden_states.shape:
                                batched_rnn_hidden_states[:, batch_idx:batch_idx+1, :] = returned_hidden
                    except Exception as e:
                        baselines_logger.warning(f"Error in skill {skill_idx} on_enter() for batch {batch_idx}: {e}")
                        continue

            self._cur_skills = (
                (~self._call_high_level) * self._cur_skills
            ) + (self._call_high_level * new_skills.to(self._call_high_level.device))
            # Ensure cur_skills contains integers for dictionary indexing
            self._cur_skills = self._cur_skills.long()


            # Reset _call_high_level to False now that we've selected new skills
            self._call_high_level = torch.zeros_like(self._call_high_level)

        # Compute the actions from the current skills
        actions = torch.zeros(
            actual_batch_size, get_num_actions(self._action_space), device=use_device
        )

        for batch_idx in range(actual_batch_size):
            skill_idx = int(self._cur_skills[batch_idx].item())
            if skill_idx not in self._skills:
                # Skip if skill index is invalid
                baselines_logger.warning(f"skill_idx={skill_idx} not in self._skills (available: {list(self._skills.keys())})")
                continue

            try:
                # Prepare hidden states for skill - handle 0-layer policies
                if num_layers > 0:
                    skill_hidden_states = batched_rnn_hidden_states[:, batch_idx:batch_idx+1, :]
                else:
                    skill_hidden_states = batched_rnn_hidden_states.new_zeros((0, 1, 0))

                skill = self._skills[skill_idx]
                action_data = skill.act(
                    batched_observations[batch_idx],
                    skill_hidden_states,
                    batched_prev_actions[batch_idx],
                    batched_masks[batch_idx],
                    [batch_idx],
                )

                # Extract action and hidden state from PolicyActionData
                action = action_data.actions if hasattr(action_data, 'actions') else action_data[0]
                returned_hidden = action_data.rnn_hidden_states if hasattr(action_data, 'rnn_hidden_states') else action_data[1]

                # Validate returned shapes before assignment
                if num_layers > 0 and returned_hidden is not None and returned_hidden.shape == skill_hidden_states.shape:
                    batched_rnn_hidden_states[:, batch_idx:batch_idx+1, :] = returned_hidden
                elif returned_hidden is not None and num_layers > 0:
                    # If shape mismatch, log warning and skip assignment
                    baselines_logger.warning(
                        f"Skill {skill_idx} returned hidden state with shape {returned_hidden.shape}, "
                        f"expected {skill_hidden_states.shape}. Skipping assignment."
                    )

                # Ensure action is properly shaped for assignment
                if isinstance(action, torch.Tensor):
                    action = action.squeeze()

                # Assign action directly to actions array
                # In per-agent scenarios, actions shape is [batch, agent_ac_size]
                if action.shape[0] < actions.shape[1]:
                    actions[batch_idx, :action.shape[0]] = action
                else:
                    actions[batch_idx] = action

            except Exception as e:
                baselines_logger.warning(f"Error in skill {skill_idx} act() for batch {batch_idx}: {e}")
                # Continue with zero action
                continue

        should_terminate = batched_bad_should_terminate | hl_terminate.to(use_device)
        if should_terminate.sum() > 0:
            # End the episode where requested.
            try:
                for batch_idx in range(actual_batch_size):
                    if should_terminate[batch_idx].item():
                        baselines_logger.info(
                            f"Calling stop action for batch {batch_idx}"
                        )
                        if self._stop_action_idx is not None:
                            actions[batch_idx, self._stop_action_idx] = 1.0
            except Exception as e:
                baselines_logger.warning(f"Error processing terminate signals: {e}")

        # Transpose back to original format if we transposed on input
        # Return format should match what MultiPolicy expects for concatenation
        if num_layers == 0:
            # For 0-layer policies, ensure we return [batch, 0, 0] for proper concatenation
            return_hidden_states = batched_rnn_hidden_states.view(actual_batch_size, 0, 0)
        elif did_transpose:
            # Transpose from [num_layers, batch, hidden] back to [batch, num_layers, hidden]
            return_hidden_states = batched_rnn_hidden_states.transpose(0, 1)
        else:
            # Already in correct format
            return_hidden_states = batched_rnn_hidden_states

        # Build policy_info with current skill for each env
        policy_info = []
        skill_id_to_name = {v: k for k, v in self._name_to_idx.items()}
        for batch_idx in range(actual_batch_size):
            skill_idx = int(self._cur_skills[batch_idx].item())
            skill_name = skill_id_to_name.get(skill_idx, f'skill_{skill_idx}')
            policy_info.append({
                'cur_skill': skill_name,  # Skill name for video visualization
                'cur_skill_idx': float(skill_idx),  # Skill index as metric
            })

        return PolicyActionData(
            actions=actions,
            rnn_hidden_states=return_hidden_states,
            policy_info=policy_info,
            values=None,
            action_log_probs=None,
            take_actions=None,
            should_inserts=None,
        )
    def parameters(self):
        # Return parameters from high-level policy which has neural layers
        return self._high_level_policy.parameters()

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        """
        Create HierarchicalPolicy from config.
        Expects config to be the full config object (config.habitat_baselines.rl...)
        and agent_name to be passed as a kwarg.
        """
        agent_name = kwargs.get("agent_name", "agent_0")
        orig_action_space = kwargs.get("orig_action_space", None)
        # Get hierarchical policy config for this agent
        hl_policy_config = config.habitat_baselines.rl.policy[agent_name].hierarchical_policy

        return cls(
            hl_policy_config,
            config,
            observation_space,
            action_space,
            config.habitat_baselines.num_environments,
            orig_action_space=orig_action_space,
            agent_name=agent_name,
        )