<div align="center">

# Harness-RL

### Black-Box Reinforcement Learning with Action-Args Decoupling for Central-Agent Multi-Agent Harnesses

[![Paper](https://img.shields.io/badge/arXiv-2608.29641-b31b1b.svg)](https://arxiv.org/abs/2608.29641)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](slime/LICENSE)

Official implementation of **Harness-RL**, accepted at PCC 2026.

</div>

Harness-RL trains language-model agents inside a central-agent multi-agent
harness without flattening the execution into a single token sequence. It
constructs black-box trajectories from interface calls and session prefix trees,
then applies **Conflict-Aware Policy Optimization (CAPO)** to route gradients
from action tokens and argument tokens to their corresponding parameter
subspaces.

The implementation supports central-only training and joint multi-agent
training. StackPlanner provides the production agent harness, while slime
provides distributed rollout, SGLang serving, Megatron training, checkpointing,
and weight synchronization.

## Repository layout

```text
.
├── StackPlanner2.0/                  # Multi-agent harness and runtime
├── slime/                            # Distributed RL training framework
├── slime/slime_plugins/stackplanner/ # Harness-RL integration and CAPO
├── configs/                          # Portable example configurations
├── scripts/                          # Data, probing, service, and evaluation tools
└── train_stackplanner.py             # Training entry point
```

## Optional LoongReflect plugin

The `loongreflect` branch adds reversible reflection-tree rollouts and the
LONG-SLIME Joint / Fast-Slow EMA self-distillation objectives. Launch explicitly
with `python train_loongreflect.py --help`; existing StackPlanner, OLE and CAPO
training defaults are preserved. See [configuration, algorithm differences and
validation](docs/loongreflect.md) before running experiments.

## Installation

The supported setup is Linux x86_64 with NVIDIA GPUs, a compatible CUDA driver,
Git, Make, and Python 3.12.

```bash
git clone --branch Harness-RL https://github.com/jiangxinke/Harness-RL.git
cd Harness-RL

make doctor
make setup
source scripts/training_env.sh
```

The setup is stored under `$HOME/.cache/harness-rl` by default. To use another
location, set `SP_ENV_PREFIX`, `SP_SOURCE_ROOT`, or `SP_BUILD_CACHE` when running
`make setup`.

## Prepare data

Download and convert the pinned ASearcher training split:

```bash
make prepare-data
```

Prepare the evaluation manifest:

```bash
make prepare-eval
```

Generated datasets are written to `data/` and are not tracked by Git.

## Prepare models

Place the Hugging Face checkpoint and its Megatron torch-distributed conversion
at locations of your choice. slime includes the conversion utility:

```bash
python slime/tools/convert_hf_to_torch_dist.py --help
```

Update `hf_checkpoint`, `reference_checkpoint`, `tokenizer`, the distributed
topology, and the model architecture arguments in the selected file under
`configs/`.

## Configure retrieval

Harness-RL uses the Search-R1 retriever through the standard StackPlanner
`web_search` interface.

Connection failures and malformed retriever responses invalidate the whole
training episode, including errors handled inside frozen specialists. Such
episodes receive zero loss masks and retryable infrastructure metadata;
ordinary empty search results remain valid tool outcomes.

```bash
make search-setup
python scripts/prepare_search_r1_assets.py \
  --asset-dir "$HOME/.cache/harness-rl/search-r1"
make search-start
make configure-runtime
```

Set `SP_SEARCH_R1_ROOT`, `SP_SEARCH_R1_CUDA_DEVICES`, or `SP_SERVICE_HOST` to
match your deployment. The committed configuration contains no credentials and
uses loopback service addresses by default.

Central-only runs also require a frozen OpenAI-compatible specialist endpoint:

```bash
SP_FROZEN_MODEL_PATH=/path/to/model make frozen-start
```

## Train

Three portable templates are provided:

| Configuration | Training policy | CAPO scope |
|---|---|---|
| `configs/experiment_a_central.yaml` | Central agent only | Disabled |
| `configs/experiment_b_all_agents.yaml` | Central and specialist policies | Action/argument routes across actor roles |
| `configs/experiment_c_capability_decoupled.yaml` | Central agent only | Action/argument routes for the central policy |

The templates use relative repository paths and start in dry-run mode. Edit the
model paths and hardware topology, inspect the resolved command, then set
`dry_run` to `false`.

```bash
python train_stackplanner.py \
  --config configs/experiment_a_central.yaml \
  --print-command

python train_stackplanner.py \
  --config configs/experiment_a_central.yaml
```

Resume from a slime checkpoint with:

```bash
python train_stackplanner.py \
  --config configs/experiment_a_central.yaml \
  --resume-from /path/to/torch-dist-checkpoint
```

## Optional OLE sample selection

OLE is disabled by default in all three templates. Enable it for A, B, or C
with `ole_enabled: true` in the configuration or the CLI switch:

```bash
python train_stackplanner.py \
  --config configs/experiment_a_central.yaml \
  --ole-enabled --ole-threshold-percent 0.4 --ole-k 1.0 \
  --ole-init-threshold 0.0
```

Use `--no-ole-enabled` to override an enabled configuration. The existing
`dry_run` setting still applies. OLE adds two slime plugin hooks only when
enabled; the disabled launch command and training behavior are unchanged.

Following the reference implementation in `../OLE`, each prompt group is
scored as `mean(episode_reward) + ole_k * std(episode_reward)` using the
population standard deviation. Scores strictly below the **previous rollout
batch's threshold** have their policy advantages set to zero after advantage
computation and whitening. The current batch sets the next threshold to the
`1 - ole_threshold_percent` quantile (0.4 targets the upper 40%; ties and
batch-to-batch changes can change the actual retained fraction).

Each episode counts once even when it produces many agent/session segments
or CAPO children. Infrastructure failures and episodes without actor tokens
are excluded from scoring. Rewards, critic returns, entropy/KL terms, loss
masks, CAPO routes, and batch scheduling retain their existing behavior.
Scores use the original episode rewards before GRPO normalization.

Thresholds and discard counts are stored in the standard rollout data-source
checkpoint under `checkpoints/rollout/`, and `--resume-from` restores them.
Keep these files together with the model checkpoint. Evaluation bypasses OLE.
Training-only replay with OLE requires rollout dumps collected with OLE enabled;
it reuses their recorded selection decisions. The rollout log includes
`ole/threshold`, `ole/next_threshold`, `ole/discarded_groups`,
`ole/discarded_episodes`, and cumulative `ole/discarded_total`.

## Build CAPO routing artifacts

Experiments B and C require routing metadata generated from successful rollout
records. These files are runtime inputs and remain outside version control.

```bash
torchrun --nproc-per-node 1 scripts/probe_stackplanner_capabilities.py \
  --model /path/to/hugging-face-checkpoint \
  --model-digest <sha256> \
  --rollout-glob "outputs/<run-id>/trajectories/*.jsonl" \
  --output-dir outputs/capability/central \
  --taxonomy central-capability
```

Use `--taxonomy all-agent-action-output` for joint multi-agent training, then
point the capability fields in the corresponding configuration to the generated
directory.

## Evaluate

slime's in-training evaluation uses greedy sampling (`temperature=0`,
`top_p=1`) for both actor and frozen roles. Training keeps the configured
temperature. The built-in StackPlanner rollout requires `top_p=1`: its adapter
does not yet retain the exact nucleus-token metadata required by slime's
training converter. Other values are rejected during configuration, before
generation. Custom generators can supply their own top-p replay support.
The upstream branch review and regression commands are
recorded in [the bug-fix review](docs/upstream-bugfix-review.md).

Generate predictions with the production StackPlanner graph:

```bash
python scripts/generate_evaluation_predictions.py \
  --manifest data/evaluation_630.jsonl \
  --output-dir outputs/evaluation \
  --checkpoint /path/to/checkpoint \
  --checkpoint-digest <sha256> \
  --actor-base-url http://127.0.0.1:18001/v1 \
  --actor-model Qwen2.5-3B-Instruct \
  --frozen-base-url http://127.0.0.1:18002/v1 \
  --frozen-model Qwen2.5-3B-Instruct
```

Score the generated predictions with an OpenAI-compatible judge:

```bash
python scripts/evaluate_predictions.py \
  --manifest data/evaluation_630.jsonl \
  --predictions outputs/evaluation/predictions.jsonl \
  --output-dir outputs/evaluation/report \
  --judge-base-url http://127.0.0.1:18002/v1
```

## Citation

```bibtex
@article{jiang2026harnessrl,
  title   = {Harness-RL: Black-Box Reinforcement Learning with Action-Args
             Decoupling for Central-Agent Multi-Agent Harnesses},
  author  = {Jiang, Xinke and Zhang, Zhixin and Yang, Zhibang and Gao, Jiaran
             and Qiu, Rihong and Chen, Shijin and Chu, Xu and Zhao, Junfeng
             and Wang, Yasha},
  journal = {arXiv preprint arXiv:2608.29641},
  year    = {2026}
}
```
