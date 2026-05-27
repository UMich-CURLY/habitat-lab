"""Unit tests for social navigation skills (BackOffSkill, WaitSkill, GoToGoalSkill)."""

import math
from types import SimpleNamespace

import gym.spaces as spaces
import pytest
import torch

from habitat_baselines.rl.hrl.skills.social_nav_skills import (
    BackOffSkill,
    GoToGoalSkill,
    WaitSkill,
)


@pytest.fixture
def action_space():
    """Action space: [lin_vel, ang_vel, stop, oracle_nav] (4D Box)."""
    return spaces.Box(low=-1, high=1, shape=(4,), dtype="float32")


@pytest.fixture
def skill_config():
    """Mock skill config with required attributes."""
    return SimpleNamespace(
        ignore_grip=True,
        apply_postconds=False,
        force_end_on_timeout=False,
        max_skill_steps=-1,
    )


def make_masks(batch_size=1):
    """Create a batch of active masks (all 1.0)."""
    return torch.ones(batch_size, 1)


def make_hidden_states(batch_size=1, num_layers=1, hidden_size=1):
    """Create a batch of RNN hidden states."""
    return torch.zeros(num_layers, batch_size, hidden_size)


# ──────────────────────────────────────────────────────────────────────────────
# BackOffSkill Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_backoff_oracle_nav_targets_goal(action_space, skill_config):
    """BackOffSkill should output 0.5 (half speed) for oracle navigation toward goal."""
    skill = BackOffSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.zeros(1, 3),
        "agent_1_base_pos": torch.ones(1, 3) * 5.0,  # Far away, so human is behind
        "agent_0_orientation": torch.zeros(1, 3),
    }
    action, _ = skill.act(obs, make_hidden_states(), None, make_masks())
    assert action[skill._oracle_nav_ac_idx].item() == pytest.approx(0.5)


def test_backoff_terminates_when_close_to_human(action_space, skill_config):
    """BackOffSkill should terminate (call_hl=True) when dist < 2.0m."""
    skill = BackOffSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.tensor([[0.0, 0.0, 0.0]]),
        "agent_1_base_pos": torch.tensor([[1.0, 0.0, 0.0]]),  # dist=1.0 < 2.0
    }
    call_hl, bad_term, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.zeros(1, dtype=torch.bool),
        actions=torch.zeros(1, 4),
    )
    assert call_hl[0].item() is True


def test_backoff_does_not_terminate_when_far_from_human(action_space, skill_config):
    """BackOffSkill should not terminate (call_hl=False) when dist >= 2.0m."""
    skill = BackOffSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.tensor([[0.0, 0.0, 0.0]]),
        "agent_1_base_pos": torch.tensor([[5.0, 0.0, 0.0]]),  # dist=5.0 > 2.0
    }
    call_hl, bad_term, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.zeros(1, dtype=torch.bool),
        actions=torch.zeros(1, 4),
    )
    assert call_hl[0].item() is False


def test_backoff_terminates_on_hl_request(action_space, skill_config):
    """BackOffSkill should always terminate if hl_wants_skill_term=True."""
    skill = BackOffSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.tensor([[0.0, 0.0, 0.0]]),
        "agent_1_base_pos": torch.tensor([[10.0, 0.0, 0.0]]),  # Far away
    }
    call_hl, _, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.ones(1, dtype=torch.bool),  # Request termination
        actions=torch.zeros(1, 4),
    )
    assert call_hl[0].item() is True


# ──────────────────────────────────────────────────────────────────────────────
# WaitSkill Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_wait_zero_linear_velocity(action_space, skill_config):
    """WaitSkill should output zero linear velocity (action[:, 0] = 0)."""
    skill = WaitSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.zeros(1, 3),
        "robot_goal_pos": torch.tensor([[1.0, 0.0, 0.0]]),
        "agent_0_orientation": torch.tensor([[0.0, 0.0, 0.0]]),
    }
    action, _ = skill.act(obs, make_hidden_states(), None, make_masks())
    assert action[0].item() == pytest.approx(0.0)


def test_wait_rotates_toward_goal(action_space, skill_config):
    """WaitSkill should output nonzero angular velocity when facing away from goal."""
    skill = WaitSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.zeros(1, 3),
        "robot_goal_pos": torch.tensor([[0.0, 1.0, 0.0]]),  # Goal at +Y
        "agent_0_orientation": torch.tensor(
            [[0.0, 0.0, -math.pi / 2]]
        ),  # Facing -Y
    }
    action, _ = skill.act(obs, make_hidden_states(), None, make_masks())
    # Angular velocity (action[:, 1]) should be nonzero
    assert abs(action[1].item()) > 0.0


def test_wait_only_terminates_on_hl_request(action_space, skill_config):
    """WaitSkill should only terminate when hl_wants_skill_term=True."""
    skill = WaitSkill(skill_config, action_space, batch_size=1)
    obs = {}

    # No HL request → should not terminate
    call_hl_no, _, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.zeros(1, dtype=torch.bool),
        actions=torch.zeros(1, 4),
    )
    assert call_hl_no[0].item() is False

    # HL requests → should terminate
    call_hl_yes, _, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.ones(1, dtype=torch.bool),
        actions=torch.zeros(1, 4),
    )
    assert call_hl_yes[0].item() is True


# ──────────────────────────────────────────────────────────────────────────────
# GoToGoalSkill Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_gotogoal_oracle_nav_is_full_speed(action_space, skill_config):
    """GoToGoalSkill should output 1.0 (full speed) for oracle navigation."""
    skill = GoToGoalSkill(skill_config, action_space, batch_size=1)
    obs = {}
    action, _ = skill.act(obs, make_hidden_states(), None, make_masks())
    assert action[skill._oracle_nav_ac_idx].item() == pytest.approx(1.0)


def test_gotogoal_terminates_at_goal(action_space, skill_config):
    """GoToGoalSkill should terminate (call_hl=True) when dist < 0.5m."""
    skill = GoToGoalSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.tensor([[0.0, 0.0, 0.0]]),
        "robot_goal_pos": torch.tensor([[0.3, 0.0, 0.0]]),  # dist=0.3 < 0.5
    }
    call_hl, _, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.zeros(1, dtype=torch.bool),
        actions=torch.zeros(1, 4),
    )
    assert call_hl[0].item() is True


def test_gotogoal_does_not_terminate_far_from_goal(action_space, skill_config):
    """GoToGoalSkill should not terminate (call_hl=False) when dist >= 0.5m."""
    skill = GoToGoalSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.tensor([[0.0, 0.0, 0.0]]),
        "robot_goal_pos": torch.tensor([[3.0, 0.0, 0.0]]),  # dist=3.0 > 0.5
    }
    call_hl, _, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.zeros(1, dtype=torch.bool),
        actions=torch.zeros(1, 4),
    )
    assert call_hl[0].item() is False


def test_gotogoal_terminates_on_hl_request(action_space, skill_config):
    """GoToGoalSkill should always terminate if hl_wants_skill_term=True."""
    skill = GoToGoalSkill(skill_config, action_space, batch_size=1)
    obs = {
        "agent_0_base_pos": torch.tensor([[0.0, 0.0, 0.0]]),
        "robot_goal_pos": torch.tensor([[10.0, 0.0, 0.0]]),  # Far away
    }
    call_hl, _, _ = skill.should_terminate(
        obs,
        make_hidden_states(),
        None,
        make_masks(),
        hl_wants_skill_term=torch.ones(1, dtype=torch.bool),  # Request termination
        actions=torch.zeros(1, 4),
    )
    assert call_hl[0].item() is True
