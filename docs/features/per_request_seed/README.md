# Per-Request Seed（每条 Prompt 独立采样 Seed）

> 本说明适用于 **release/v0.8.0** 分支上的 `feat/per-request-seed-v0.8.0`。

## 功能说明

默认情况下，VERL rollout **不会**为每条 prompt 设置 `SamplingParams.seed`，vLLM 使用引擎内全局 RNG，行为与改前一致。

开启 `per_request_seed` 后，每条生成请求会获得独立的、可复现的 `SamplingParams.seed`。

## 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `actor_rollout_ref.rollout.per_request_seed` | `false` | **主开关**：`false` 关闭（默认），`true` 开启 |
| `actor_rollout_ref.rollout.seed` | `null` | 可选基础 seed（默认 0），参与 per-request seed 计算及 vLLM 引擎启动 |

Per-request seed 计算公式（`temperature > 0` 时生效）：

```
seed = (base_seed + sample_index * 1000003 + rollout_n * 1009 + global_step * 9176) & 0x7FFFFFFF
```

## 脚本启动方式

### 方式 1：环境变量（推荐，见 demo 脚本）

```bash
# 默认关闭，与改前行为一致
bash examples/per_request_seed/run_grpo_qwen3_8b_fsdp.sh

# 开启 per-prompt seed
ROLLOUT_PER_REQUEST_SEED=true bash examples/per_request_seed/run_grpo_qwen3_8b_fsdp.sh

# 开启并指定基础 seed
ROLLOUT_PER_REQUEST_SEED=true ROLLOUT_SEED=42 bash examples/per_request_seed/run_grpo_qwen3_8b_fsdp.sh
```

### 方式 2：Hydra 命令行

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.per_request_seed=true \
    actor_rollout_ref.rollout.seed=42 \
    ...
```

## 注意事项

1. **`temperature=0` 时 seed 无效**（greedy 采样），验证集默认不受影响
2. **引擎 seed 仍会设置**（`replica_rank + seed`），用于 TP worker 一致性
3. **多轮 tool agent**：同一 `request_id` 内 generator 状态连续推进

## 涉及代码（v0.8.0）

- `verl/workers/config/rollout.py` — 配置字段
- `verl/workers/rollout/utils.py` — seed 计算工具
- `verl/experimental/agent_loop/agent_loop.py` — 主 async rollout 路径
- `verl/trainer/main_ppo_sync.py` — TransferQueue 训练路径
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` — 引擎 seed 配置

## 与 main 分支差异

- v0.8.0 无 `full_determinism` 配置项
- TransferQueue 路径在 `main_ppo_sync.py`（main 分支在 `agent_loop_tq.py`）
- v0.8.0 的 `seed` 默认为 `null`，main 分支默认为 `42`
