#!/usr/bin/env python3
"""Simple runner to drive both agents using oracle navigation actions.

This script loads a two-agent social-nav config and repeatedly issues
oracle nav actions for agent_0 and agent_1 (if available) until both
have finished navigating or a max number of steps is reached.

It purposely avoids habitat-baselines so no RL policy/model gets built.
"""

import argparse
import time
import sys
import os
import numpy as np
from datetime import datetime

# TensorBoard SummaryWriter import with fallbacks (torch, tensorboardX, tensorboard)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except Exception:
        try:
            from tensorboard.summary.writer.event_file_writer import SummaryWriter  # type: ignore
        except Exception:
            SummaryWriter = None  # type: ignore

try:
    import habitat
except Exception:
    print("Could not import habitat. Make sure your PYTHONPATH includes the repo and habitat is installed.")
    raise


def _log_to_tensorboard(writer, obs, env, global_step, log_images=True):
    """Helper: log scalar metrics and a few image-like observations to TB.

    This function is defensive — it swallows exceptions so logging never
    interrupts the run.
    """
    if writer is None:
        return

    # Scalars / metrics
    try:
        metrics = env.get_metrics()
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        writer.add_scalar(f"metrics/{k}", float(v), global_step)
                except Exception:
                    # skip non-scalar metric
                    pass
    except Exception:
        # env.get_metrics may not exist or fail early; ignore
        pass

    if not log_images or not isinstance(obs, dict):
        return

    # Images
    for oname, oval in obs.items():
        try:
            arr = np.array(oval)
        except Exception:
            continue

        try:
            # image-like arrays: (H,W,3), (3,H,W), (H,W)
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                img = arr
                if img.dtype != np.uint8:
                    mn = np.nanmin(img)
                    mx = np.nanmax(img)
                    if mx > mn:
                        img = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
                    else:
                        img = (img * 255.0).astype(np.uint8)
                writer.add_image(f"obs/{oname}", img, global_step, dataformats="HWC")
            elif arr.ndim == 3 and arr.shape[0] in (3, 4):
                img = np.transpose(arr, (1, 2, 0))
                if img.dtype != np.uint8:
                    mn = np.nanmin(img)
                    mx = np.nanmax(img)
                    if mx > mn:
                        img = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
                    else:
                        img = (img * 255.0).astype(np.uint8)
                writer.add_image(f"obs/{oname}", img, global_step, dataformats="HWC")
            elif arr.ndim == 2:
                a = arr.astype(np.float32)
                mn = np.nanmin(a)
                mx = np.nanmax(a)
                if mx > mn:
                    a = (a - mn) / (mx - mn)
                a = (a * 255.0).astype(np.uint8)
                a = np.expand_dims(a, axis=2)
                writer.add_image(f"obs/{oname}", a, global_step, dataformats="HWC")
        except Exception:
            # Keep logging robust
            continue


def run_episode(
    env,
    max_steps: int = 1000,
    sleep: float = 0.0,
    writer=None,
    tb_log_interval: int = 1,
    log_images: bool = True,
    ep_index: int = 0,
):
    obs = env.reset()
    finished0 = bool(obs.get("agent_0_has_finished_oracle_nav", 0))
    finished1 = bool(obs.get("agent_1_has_finished_oracle_nav", 0))
    
    # Step repeatedly until episode ends or max_steps reached. We use a
    # conservative empty-action dict for stepping; callers can replace this
    # with their own action-selection logic if needed.
    # Log initial observation (step 0)
    if writer is not None:
        global_step = ep_index * max_steps + getattr(env, "_elapsed_steps", 0)
        _log_to_tensorboard(writer, obs, env, global_step, log_images=log_images)

    for _ in range(max_steps):
        if getattr(env, "episode_over", False):
            break
        import pdb; pdb.set_trace()
        obs = env.step({"agent_0_oracle_nav_action"})
        # Observations may include the has_finished_* flags
        finished0 = finished0 or bool(obs.get("agent_0_has_finished_oracle_nav", 0))
        finished1 = finished1 or bool(obs.get("agent_1_has_finished_oracle_nav", 0))

        # Log at interval if requested
        if writer is not None:
            global_step = ep_index * max_steps + getattr(env, "_elapsed_steps", 0)
            if global_step % max(1, tb_log_interval) == 0:
                _log_to_tensorboard(writer, obs, env, global_step, log_images=log_images)

        if sleep:
            time.sleep(sleep)

    # Observations may include the has_finished_* flags
    finished0 = finished0 or bool(obs.get("agent_0_has_finished_oracle_nav", 0))
    finished1 = finished1 or bool(obs.get("agent_1_has_finished_oracle_nav", 0))
    # TensorBoard logging (metrics + observations)
    if writer is not None and not env.episode_over:
        global_step = ep_index * max_steps + env._elapsed_steps
        # Log metrics if available
        try:
            metrics = env.get_metrics()
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    try:
                        if isinstance(v, (int, float, np.floating, np.integer)):
                            writer.add_scalar(f"metrics/{k}", float(v), global_step)
                    except Exception:
                        # skip non-scalar metric
                        pass
        except Exception:
            # env.get_metrics may not exist or fail early; ignore
            pass
        # Log a few observation images (RGB / depth) if requested
        if log_images and isinstance(obs, dict):
            for oname, oval in obs.items():
                try:
                    arr = np.array(oval)
                except Exception:
                    continue
                # image-like arrays: (H,W,3), (3,H,W), (H,W)
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    # HWC -> use as-is
                    img = arr
                    # Convert uint8 if necessary
                    if img.dtype != np.uint8:
                        # scale to 0-255
                        mn = np.nanmin(img)
                        mx = np.nanmax(img)
                        if mx > mn:
                            img = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
                        else:
                            img = (img * 255.0).astype(np.uint8)
                    writer.add_image(f"obs/{oname}", img, global_step, dataformats="HWC")
                elif arr.ndim == 3 and arr.shape[0] in (3, 4):
                    # CHW -> convert to HWC
                    img = np.transpose(arr, (1, 2, 0))
                    if img.dtype != np.uint8:
                        mn = np.nanmin(img)
                        mx = np.nanmax(img)
                        if mx > mn:
                            img = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
                        else:
                            img = (img * 255.0).astype(np.uint8)
                    writer.add_image(f"obs/{oname}", img, global_step, dataformats="HWC")
                elif arr.ndim == 2:
                    # single-channel image (depth/gray) -> expand to CHW
                    # normalize to 0-1
                    a = arr.copy()
                    a = a.astype(np.float32)
                    mn = np.nanmin(a)
                    mx = np.nanmax(a)
                    if mx > mn:
                        a = (a - mn) / (mx - mn)
                    a = (a * 255.0).astype(np.uint8)
                    a = np.expand_dims(a, axis=2)
                    writer.add_image(f"obs/{oname}", a, global_step, dataformats="HWC")

    if sleep:
        time.sleep(sleep)

    print(f"Episode finished after {env._elapsed_steps} steps. finished0={finished0} finished1={finished1}")
    # print any metrics if available
    try:
        metrics = env.get_metrics()
        print("Metrics:", metrics)
    except Exception:
        pass




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="benchmark/multi_agent/hssd_spot_human_social_nav_agent0_oracle.yaml",
                        help="Path to the Habitat config relative to the repo (same format used by examples)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between step calls (useful for watching render)")
    parser.add_argument("--tb-logdir", type=str, default=None, help="TensorBoard logdir (if omitted, TB logging disabled)")
    parser.add_argument("--tb-log-interval", type=int, default=10, help="Log to TB every N steps")
    parser.add_argument("--tb-log-images", action="store_true", help="Also log image-like observations to TensorBoard (may be large)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional config overrides")

    args = parser.parse_args()

    # Load config and enable debug render for visualization
    config = habitat.get_config(args.cfg, args.opts or [])
    with habitat.config.read_write(config):
        # Turn on simulator debug rendering when running locally
        try:
            config.habitat.simulator.debug_render = True
        except Exception:
            pass

    print("Creating environment from config:", args.cfg)
    env = habitat.Env(config=config)

    # Setup TensorBoard writer if requested and available
    writer = None
    if args.tb_logdir is not None and SummaryWriter is not None:
        # If user provided a directory, use it; otherwise create a timestamped run dir
        tb_dir = args.tb_logdir
        if tb_dir == "":
            tb_dir = os.path.join(
                "runs",
                f"run_oracle_both_agents_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
            )
        os.makedirs(tb_dir, exist_ok=True)
        try:
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard writer created at {tb_dir}")
        except Exception as e:
            print(f"Could not create TensorBoard SummaryWriter: {e}")
            writer = None
    elif args.tb_logdir is not None and SummaryWriter is None:
        print("TensorBoard logging requested but SummaryWriter not available (install torch/tensorboardX/tensorboard)")

    for ep in range(args.episodes):
        print(f"=== Episode {ep+1}/{args.episodes} ===")
        run_episode(
            env,
            max_steps=args.max_steps,
            sleep=args.sleep,
            writer=writer,
            tb_log_interval=args.tb_log_interval,
            log_images=args.tb_log_images,
            ep_index=ep,
        )

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


    env.close()


if __name__ == "__main__":
    main()
