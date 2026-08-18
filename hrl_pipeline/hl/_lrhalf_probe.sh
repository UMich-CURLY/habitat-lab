#!/bin/bash
# lr/2 trend probe (~2.5h end-to-end): actor lr 5e-5 -> 2.5e-5, everything
# else identical to the NR1 recipe (3-term reward, TRAIN set, no DAPG,
# warmup 120 calls). Budget 1.0e6 (u122, ckpt every 100k).
#
# Streaming readout: matched-budget points are quick-evaluated (evals_per_ep=1
# on EVAL) as they appear -- ckpt.5/7/9 = ~u73/u98/u122 -- against the SAME
# checkpoints of the lr=5e-5 baseline (checkpoints/nr1, same data content and
# recipe). 1-eval is a screen, not a verdict: flaky-boundary episodes wobble.
set -x
cd /habitat-lab && . activate habitat
HP=/habitat-lab/hrl_pipeline
V2=social_nav/social_nav_hierarchical_overfit_v2.yaml
HL=habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy
R=habitat.task.measurements.social_nav_reward
export SPLIT_SCENES=1 DISABLE_CUDNN=1

qeval () {  # tag hl.pth
  python -u -m habitat_baselines.run --config-name=$V2 \
    habitat_baselines.evaluate=True habitat_baselines.eval.should_load_ckpt=False \
    +$HL.pretrained_hl_weights=$2 \
    habitat_baselines.num_environments=1 habitat_baselines.eval.evals_per_ep=1 \
    habitat.dataset.data_path=/habitat-lab/data/social_nav_episode_EVAL.json.gz \
    habitat.seed=7 habitat.environment.max_episode_steps=1200 \
    "habitat_baselines.eval.video_option=[]" \
    habitat_baselines.eval.episode_stats_path=$HP/results/stats_lrprobe_$1.json \
    > $HP/runs/_qeval_lrprobe_$1.log 2>&1
  echo "QEVAL_${1}_EXIT=$?"
}

rm -rf $HP/checkpoints/lrhalf $HP/tb/lrhalf
python -u -m habitat_baselines.run --config-name=$V2 \
  habitat.dataset.data_path=/habitat-lab/data/social_nav_episode_TRAIN.json.gz \
  +$HL.pretrained_hl_weights=$HP/weights/bc_stable_hl.pth \
  +$HL.critic_warmup_calls=120 \
  habitat_baselines.rl.ppo.critic_lr=3e-2 \
  habitat_baselines.rl.ppo.gamma=0.99 \
  habitat_baselines.rl.ppo.num_steps=1024 habitat_baselines.rl.ppo.num_mini_batch=1 \
  habitat_baselines.rl.ppo.entropy_coef=0.0 habitat_baselines.rl.ppo.value_loss_coef=0.05 \
  habitat_baselines.rl.ppo.lr=2.5e-5 habitat_baselines.rl.ppo.max_grad_norm=2.0 \
  habitat.task.success_reward=50.0 \
  $R.collide_penalty=30.0 $R.safe_dis_min=0.0 \
  $R.eff_success_reward=20.0 $R.corridor_potential_coef=3.0 $R.release_bonus=3.0 \
  habitat_baselines.total_num_steps=1.0e6 habitat_baselines.num_checkpoints=10 \
  habitat_baselines.checkpoint_folder=hrl_pipeline/checkpoints/lrhalf \
  habitat_baselines.tensorboard_dir=hrl_pipeline/tb/lrhalf \
  habitat_baselines.video_dir=hrl_pipeline/videos/lrhalf \
  > $HP/runs/_ppo_lrhalf.log 2>&1 &
TRAIN_PID=$!

# lr=5e-5 baseline points while the probe warms up (2nd GPU proc, ~4 GB).
for ck in 5 7 9; do
  python $HP/hl/_extract_hl_map.py $HP/checkpoints/nr1/ckpt.$ck.pth \
    $HP/weights/nr1_ck${ck}_hl.pth
  qeval base_ck$ck $HP/weights/nr1_ck${ck}_hl.pth
done

for ck in 5 7 9; do
  until [ -f $HP/checkpoints/lrhalf/ckpt.$ck.pth ]; do
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
      echo TRAIN_DIED_EARLY
      tail -40 $HP/runs/_ppo_lrhalf.log | grep -E "Error|OutOfMemory" | head -3
      exit 1
    fi
    sleep 60
  done
  sleep 10
  python $HP/hl/_extract_hl_map.py $HP/checkpoints/lrhalf/ckpt.$ck.pth \
    $HP/weights/lrhalf_ck${ck}_hl.pth
  qeval half_ck$ck $HP/weights/lrhalf_ck${ck}_hl.pth
done

wait $TRAIN_PID
echo "LRHALF_TRAIN_EXIT=$?"
python $HP/hl/quick_trend.py
echo LRHALF_PROBE_DONE
