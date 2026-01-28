#!/usr/bin/env bash
# Wrapper to run DD-PPO for the two-agent social-nav config with a safe
# LD_LIBRARY_PATH that prefers the active conda env's lib directory. This
# avoids libcudart / libtorch CUDA undefined symbol problems caused by the
# system loader picking the wrong CUDA runtime.

set -euo pipefail

# Resolve conda env lib path if available, otherwise fallback to a common path
if [ -n "${CONDA_PREFIX-}" ]; then
  CONDA_LIB_DIR="$CONDA_PREFIX/lib"
else
  CONDA_LIB_DIR="/opt/conda/envs/habitat/lib"
fi

export LD_LIBRARY_PATH="$CONDA_LIB_DIR:${LD_LIBRARY_PATH-}"

echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Default args
CONFIG_NAME=${1:-social_nav/social_nav_twoagent}
shift || true

# Run the habitat-baselines runner with the provided hydra overrides/args.
# Example usage:
# ./scripts/run_ddppo_twoagent.sh social_nav/social_nav_twoagent habitat.dataset.data_path=/abs/path/to/dataset.json.gz

python -u -m habitat_baselines.run --config-name "$CONFIG_NAME" "$@"
