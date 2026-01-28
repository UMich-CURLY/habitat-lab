#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Change to habitat-lab directory for correct relative paths
# os.chdir("/home/tribhi/research/habicrowd/habitat-lab")

def setup_tensorboard_logging():
    """Setup comprehensive tensorboard logging."""
    tb_dir = "./tb_logs/oracle_social_nav"
    os.makedirs(tb_dir, exist_ok=True)
    print(f"Tensorboard logs will be saved to: {tb_dir}")
    print(f"Start tensorboard with: tensorboard --logdir {tb_dir}")

def main():
    """Run oracle social navigation with full visualization."""
    
    print("=== Oracle Social Navigation Evaluation ===")
    print("This script runs two agents with oracle navigation")
    print("and logs all sensors/measurements to tensorboard.")
    print()
    
    # Setup tensorboard
    setup_tensorboard_logging()
    
    # Set environment variables for better debugging
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["MAGNUM_LOG"] = "quiet"

    # Compute repository root and the absolute config path so Hydra loads
    # the workspace configs (avoids falling back to installed package configs).
    repo_root = Path(__file__).resolve().parents[2]
    config_path = str(repo_root / "habitat-baselines" / "habitat_baselines" / "config" / "social_nav")
    # Build command to run habitat-baselines directly
    # Use the correct config structure from your social_nav_twoagent.yaml
    cmd_args = [
        "python", "-u", "-m", "habitat_baselines.run",
        f"--config-path={config_path}",
        "--config-name=social_nav_twoagent",
        
        # Use the correct paths (no habitat_baselines prefix)
    # Use fully-qualified Hydra overrides so keys exist in the composed
    # config schema (avoid Hydra "Key 'evaluate' is not in struct" errors)
    # Use + to append overrides when the composed structured config
    # doesn't expose the `habitat_baselines` node at top-level.
    "habitat_baselines.evaluate=True",
    "habitat_baselines.num_environments=1",
    "habitat_baselines.num_updates=0",
        
        # Evaluation settings
    "habitat_baselines.eval.should_load_ckpt=False",
    "habitat_baselines.eval.video_option=[tensorboard,disk]",
    "habitat_baselines.video_fps=10",
        
        # Logging
    "habitat_baselines.tensorboard_dir=./tb_logs/oracle_social_nav",
    "habitat_baselines.log_interval=1",
        
        # Run settings  
        "+test_episode_count=5",
        
    # Debug rendering: use plain override (no +) because this key
    # already exists in the composed config and cannot be appended.
    "habitat.simulator.debug_render=True",
    ]
    
    print("Running command:")
    print(" ".join(cmd_args))
    print()
    
    try:
        import subprocess
        # Run from the repository root so Hydra composes using the local
        # workspace config path instead of pkg:// installed configs.
        result = subprocess.run(cmd_args, check=True, cwd=str(repo_root))
        
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=== Evaluation Complete ===")
    print("Check tensorboard for visualizations:")
    print("  tensorboard --logdir ./tb_logs/oracle_social_nav")
    print("Videos saved in tb_logs directory")
    
    return 0

if __name__ == "__main__":
    exit(main())