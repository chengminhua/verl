#!/usr/bin/env bash
# Demo: GRPO with optional per-request seed toggle (release/v0.8.0).
#
# This script is standalone; it does not modify any existing example scripts.
#
# Usage:
#   bash examples/per_request_seed/run_grpo_qwen3_8b_fsdp.sh
#   ROLLOUT_PER_REQUEST_SEED=true bash examples/per_request_seed/run_grpo_qwen3_8b_fsdp.sh
#   ROLLOUT_PER_REQUEST_SEED=true ROLLOUT_SEED=42 bash examples/per_request_seed/run_grpo_qwen3_8b_fsdp.sh
#
# See docs/features/per_request_seed/README.md for details.

set -xeuo pipefail

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
rollout_n=${ROLLOUT_N:-5}

# Per-request seed toggle (default: off, legacy behavior)
rollout_per_request_seed=${ROLLOUT_PER_REQUEST_SEED:-false}
rollout_seed=${ROLLOUT_SEED:-}

gsm8k_train=${GSM8K_TRAIN:-$HOME/data/gsm8k/train.parquet}
gsm8k_test=${GSM8K_TEST:-$HOME/data/gsm8k/test.parquet}

HYDRA_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="['${gsm8k_train}']"
    data.val_files="['${gsm8k_test}']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.per_request_seed=${rollout_per_request_seed}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME:-verl_grpo_per_request_seed_demo}
    trainer.experiment_name=${EXPERIMENT_NAME:-qwen3_8b_per_request_seed_${rollout_per_request_seed}}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.total_epochs=1
    trainer.test_freq=-1
    trainer.save_freq=-1
)

if [[ -n "${rollout_seed}" ]]; then
    HYDRA_ARGS+=(actor_rollout_ref.rollout.seed=${rollout_seed})
fi

python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" "$@"
