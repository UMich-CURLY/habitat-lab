import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from IPython import embed

class HabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        observations = envs.reset()
        observations = envs.post_step(observations)
        initial_episodes = envs.current_episodes()
        logger.info(f"Initial episodes after reset: {[(e.scene_id, e.episode_id) for e in initial_episodes]}")
        
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        # construct_envs may have reduced the env count below the configured
        # num_environments (eval caps it at #scenes) — size per-env state by
        # the actual env count.
        test_recurrent_hidden_states = torch.zeros(
            (
                envs.num_envs,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            envs.num_envs,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            envs.num_envs,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                # construct_envs may have reduced the env count below the
                # configured num_environments (eval caps it at #scenes).
                for env_idx in range(envs.num_envs)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        logger.info(f"Agent type: {type(agent)}")
        logger.info(f"Agent has actor_critic: {hasattr(agent, 'actor_critic')}")
        if hasattr(agent, '_agents'):
            logger.info(f"Agent has _agents: {agent._agents}")
            logger.info(f"Number of agents: {len(agent._agents) if agent._agents else 0}")
            if agent._agents:
                for i, ag in enumerate(agent._agents):
                    logger.info(f"Agent {i}: type={type(ag)}, has actor_critic={hasattr(ag, 'actor_critic')}")
                    if hasattr(ag, 'actor_critic'):
                        logger.info(f"  Actor critic type: {type(ag.actor_critic)}")
        
            
        step_count = 0
        episode_step_counts = [0] * envs.num_envs  # Track steps per environment
        # Force reset after this many steps; -1 disables the forced reset.
        max_episode_steps_override = int(
            getattr(
                config.habitat_baselines.eval,
                "max_episode_steps_override",
                500,
            )
        )
        
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            step_count += 1
            # Increment step counts for each environment
            for i in range(envs.num_envs):
                episode_step_counts[i] += 1
            
            if step_count % 100 == 0:
                logger.info(f"###### episodes evaluation ### ({len(stats_episodes)}/{number_of_eval_episodes * evals_per_ep}) - Step {step_count}")
            
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=bool(
                        getattr(
                            config.habitat_baselines.eval,
                            "deterministic",
                            False,
                        )
                    ),
                    **space_lengths,
                )
                
                if step_count == 1:
                    logger.info(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'NOT A DICT'}")
                    logger.info(f"ALL BATCH KEYS: {list(batch.keys())}")
                    has_oracle = "agent_0_oracle_nav_action" in batch
                    logger.info(f"HAS agent_0_oracle_nav_action: {has_oracle}")
                    if has_oracle:
                        logger.info(f"agent_0_oracle_nav_action shape: {batch['agent_0_oracle_nav_action'].shape}")
                    sample_items = [(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in list(batch.items())[:3]]
                    logger.info(f"Batch sample values: {sample_items}")
                    logger.info(f"First action_data: actions shape={action_data.actions.shape}, min={action_data.actions.min()}, max={action_data.actions.max()}, std={action_data.actions.std()}")
                    logger.info(f"env_actions shape={action_data.env_actions.shape}, min={action_data.env_actions.min()}, max={action_data.env_actions.max()}, std={action_data.env_actions.std()}")
                    logger.info(f"Agent actor_critic type: {type(agent.actor_critic)}")
                    
                    # For MultiPolicy, check active policies and their outputs
                    if hasattr(agent.actor_critic, '_active_policies'):
                        active_policies = agent.actor_critic._active_policies
                        logger.info(f"MultiPolicy active policies type: {type(active_policies)}")
                        if isinstance(active_policies, list):
                            logger.info(f"Number of active policies: {len(active_policies)}")
                            for i, policy in enumerate(active_policies):
                                logger.info(f"  Policy {i}: type={type(policy).__name__}")
                                # For HierarchicalPolicy, check skills
                                if hasattr(policy, '_skills'):
                                    logger.info(f"    Skills: {policy._skills}")
                                    logger.info(f"    Name to idx: {policy._name_to_idx}")
                                    logger.info(f"    Current skills: {policy._cur_skills}")
                                    logger.info(f"    Call HL: {policy._call_high_level}")
                        elif isinstance(active_policies, dict):
                            logger.info(f"Active policy keys: {list(active_policies.keys())}")
                            for name, policy in active_policies.items():
                                logger.info(f"  Policy '{name}': type={type(policy).__name__}")
                    
                    # Check if this is a hierarchical policy
                    if hasattr(agent.actor_critic, 'agent_0_policy'):
                        logger.info(f"Has agent_0_policy: {agent.actor_critic.agent_0_policy}")
                    if hasattr(agent.actor_critic, 'hierarchical_policy'):
                        logger.info(f"Has hierarchical_policy: {agent.actor_critic.hierarchical_policy}")
                
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]
            outputs = envs.step(step_data)
            
            if step_count == 1:
                logger.info(f"First step_data sent to env: {step_data}")
                logger.info(f"action_data.env_actions type: {type(action_data.env_actions)}, shape: {action_data.env_actions.shape}")
            
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)

            # DEBUG: Print agent positions to track movement
            if step_count % 10 == 0:
                if "agent_0_base_pos" in observations[0]:
                    pos = observations[0]["agent_0_base_pos"]
                    print(f"[POSITION] Step {step_count}: agent_0_base_pos={pos}")
                if "agent_1_base_pos" in observations[0]:
                    pos = observations[0]["agent_1_base_pos"]
                    print(f"[POSITION] Step {step_count}: agent_1_base_pos={pos}")
            # print("Testing agent_sensors ")
            # print("Position of human gps ", observations[0]["agent_0_other_agent_gps"])
            # print("Agent goal sensor ", observations[0]["agent_0_goal_to_agent_gps_compass"])
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            # Force episode end if it's been running too long (step counter safety) - DO THIS BEFORE CREATING MASKS
            for i in range(len(dones)):
                if (
                    max_episode_steps_override > 0
                    and episode_step_counts[i] >= max_episode_steps_override
                ):
                    dones[i] = True
                    episode_step_counts[i] = 0
                    if step_count % 100 == 0:
                        logger.info(f"Forced episode end for env {i} due to step limit ({max_episode_steps_override})")

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            if step_count % 100 == 0:
                logger.info(f"Dones: {dones}, any_done: {any(dones)}, current_episode_info: {[(e.scene_id, e.episode_id) for e in current_episodes_info[:2]]}")
                logger.info(f"First env info keys: {list(infos[0].keys()) if infos else 'NO INFOS'}")
                if infos:
                    logger.info(f"First env info sample: {dict(list(infos[0].items())[:3])}")

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # Overlay the social-nav HL policy inputs (robot-frame) onto
                    # the video so they can be visually verified. Angles shown in
                    # degrees, distances in m, speed in m/s. Prepended so all six
                    # stay visible above the measurement list. No-op for other
                    # tasks (guarded by key).
                    for _sk in (
                        "agent_0_social_nav_policy_state",
                        "social_nav_policy_state",
                    ):
                        if _sk in batch:
                            _s = batch[_sk][i]
                            _s = (
                                _s.detach().cpu().numpy()
                                if hasattr(_s, "detach")
                                else np.asarray(_s)
                            ).reshape(-1)
                            if _s.shape[0] >= 6:
                                disp_info = {
                                    "hl_human_dist_m": float(_s[0]),
                                    "hl_human_bearing_deg": float(np.degrees(_s[1])),
                                    "hl_human_relheading_deg": float(np.degrees(_s[2])),
                                    "hl_human_relspeed_mps": float(_s[3]),
                                    "hl_goal_dist_m": float(_s[4]),
                                    "hl_goal_bearing_deg": float(np.degrees(_s[5])),
                                    **disp_info,
                                }
                            break
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    logger.info(f"✓ Episode {k} COMPLETED (eval #{ep_eval_count[k]}/{evals_per_ep}), reward: {episode_stats['reward']:.4f}, stats: {episode_stats}")
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )
                    
                    # The env has auto-reset to its NEXT dataset episode. Pause it
                    # only when that next episode has already been evaluated
                    # evals_per_ep times (dataset cycled); otherwise let a single
                    # env sequentially cover many episodes instead of stopping
                    # after the first. Use .get() so we don't insert phantom keys
                    # into the defaultdict (len(ep_eval_count) gates the assertion).
                    next_k = (
                        next_episodes_info[i].scene_id,
                        next_episodes_info[i].episode_id,
                    )
                    # Only pause once we already have enough episode-evals; the
                    # episode iterator cycles, so otherwise a single env would be
                    # paused the moment it revisits an episode (e.g. after one
                    # scene's worth) and the run would stop short of the target.
                    if (
                        ep_eval_count.get(next_k, 0) >= evals_per_ep
                        and len(stats_episodes) >= number_of_eval_episodes * evals_per_ep
                    ):
                        envs_to_pause.append(i)
                        logger.info(f"Marking environment {i} for pause (next episode already fully evaluated)")
                else:
                    # Episode not done, check if the CURRENT episode (which is running) has already been fully evaluated
                    current_ep_key = (
                        next_episodes_info[i].scene_id,
                        next_episodes_info[i].episode_id,
                    )
                    if (
                        current_ep_key in ep_eval_count
                        and ep_eval_count[current_ep_key] >= evals_per_ep
                        and len(stats_episodes) >= number_of_eval_episodes * evals_per_ep
                    ):
                        envs_to_pause.append(i)
                        logger.info(f"Marking environment {i} for pause (current episode already fully evaluated)")
                    elif step_count % 300 == 0:
                        logger.info(f"Env {i} still running episode {current_ep_key}, cumulative reward: {current_episode_reward[i].item():.4f}")

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        # Count episode-EVALS (instances), not unique episodes: the iterator
        # cycles, so with test_episode_count > #unique-available we intentionally
        # re-run episodes to reach the requested count.
        _target_evals = number_of_eval_episodes * evals_per_ep
        assert (
            len(stats_episodes) >= _target_evals
        ), f"Expected {_target_evals} episode-evals, got {len(stats_episodes)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        episode_stats_path = str(
            getattr(config.habitat_baselines.eval, "episode_stats_path", "")
        )
        if episode_stats_path:
            import json as _json

            with open(episode_stats_path, "w") as f:
                _json.dump(
                    {
                        f"{scene_ep[0]}|{scene_ep[1]}|{count}": stats
                        for (scene_ep, count), stats in stats_episodes.items()
                    },
                    f,
                    indent=1,
                    default=float,
                )
            logger.info(f"Wrote per-episode eval stats to {episode_stats_path}")

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
