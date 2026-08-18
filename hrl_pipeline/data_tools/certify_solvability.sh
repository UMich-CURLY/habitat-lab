#!/bin/bash
# Solvability certification (Step 5.5): run every scripted mode on a candidate
# dataset; an episode is CERTIFIED solvable if ANY mode succeeds, and the set
# of succeeding modes tells you which behavior class PPO must discover.
#
# Usage (inside container):  bash certify_solvability.sh <dataset_stem>
#   e.g. bash certify_solvability.sh manual30      (data/social_nav_episode_manual30.json.gz)
# Then on host:              python3 certify_merge.py <dataset_stem>
set -x
cd /habitat-lab && . activate habitat
DS=$1
SD=/habitat-lab/hrl_pipeline
V2CFG=social_nav/social_nav_hierarchical_overfit_v2.yaml
for mode in always_go rule_wait rule_waitgo smart_wait rule_markov rule_yield; do
  rm -f $SD/hl_decisions_$mode.jsonl
  RULE_MODE=$mode RULE_EVALS=1 STEP_CAP=1200 \
    EVAL_STATS=$SD/cert_${DS}_${mode}.json RULE_CONFIG=$V2CFG \
    RULE_DATASET=/habitat-lab/data/social_nav_episode_${DS}.json.gz \
    python hrl_pipeline/scripted_hl_eval.py > $SD/_cert_${DS}_${mode}.log 2>&1
  mv -f $SD/hl_decisions_$mode.jsonl $SD/hl_decisions_${mode}_cert_${DS}.jsonl 2>/dev/null
done
echo CERT_RUNS_DONE
