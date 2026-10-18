# Optional LoongReflect training

This branch ports the retained training implementation from
[LONG-SLIME](https://github.com/EunTilofy/LONG-SLIME), revision
`90634e03f68933744762476c659c8dcf38b1448d`. The optional package is
`slime_plugins.loongreflect`. Its code is available under Apache-2.0, as is
the reference repository; the Slime upstream license remains in `slime/LICENSE`.

The normal `train_stackplanner.py` entry point and its OLE and CAPO hooks keep
their existing behavior. Select LoongReflect explicitly with
`train_loongreflect.py`. This selects the reference reflective search harness;
it is a separate training preset, not an OLE/CAPO combination.

## Implemented method

The harness records an append-only search/reflection tree, archives failed
branches, and resumes from checkpoints using a corrective update. Every sampled
model turn becomes one compact Slime sample. Shared turns are trained once,
and rewards propagate from the best descendant outcome. Sampled token IDs and
their rollout log probabilities are preserved; observations, controller
updates and synthetic closing suffixes receive no loss.

The EMA teacher privately sees the reference answer and scores the exact
student continuation. The student rollout never receives that teacher prompt.
Dense per-token rewards replace group-relative advantages. Distillation uses
the clipped, nonnegative k3 reverse-KL estimator only on sampled `<reflect>`
tokens. Backtrack tokens remain outside the fast mask, following the reference.
The EMA coefficient follows a cosine schedule from 0.996 to 1.0.

* `joint`: one optimizer update of policy loss plus 0.5 times distillation.
* `fast_slow`: K actual distillation-only optimizer updates, then one
  policy-only update on the same rollout. Teacher and old-policy probabilities
  stay fixed for all substeps. K defaults to 3.
* `policy_only` and `distill_only`: retained ablations.

The existing Slime custom loss and advantage hooks run these objectives.
Small, conditional runtime additions carry the optional batch fields, score
the private teacher, run the ordered optimizer updates, and save/restore EMA
tensors. Optimizer and scheduler state use ordinary Slime checkpoints; the
EMA file is saved per rank beside each checkpoint. Resume fails if that EMA
state is missing or incompatible. Learning-rate schedule sizing counts the
actual optimizer substeps.

This integration uses the existing **full-parameter Megatron trainer**. It
does not import the reference repository's separate LoRA/lightweight checkpoint
implementation. Context parallelism is restricted to 1, and top-p to 1. Routing
replay, critic/OPD mixing and optimizer reset between substeps are rejected.
Teacher and student prompts can have different lengths; teacher sequences
remain within the configured sequence budget. Full-parameter EMA stores an
additional CPU copy of the model and increases host memory and checkpoint use.

## Run

Prepare a JSONL training split with nonempty `query` and `answer` fields.
`answer` may also be a list of aliases. Rows explicitly marked as evaluation
splits are rejected. Independently check train/evaluation disjointness when
data provenance is not recorded.

After installing the normal Harness-RL GPU runtime and preparing HF and
Megatron checkpoints, validate and print a portable command:

```bash
python train_loongreflect.py \
  --data /path/to/train.jsonl \
  --hf-checkpoint /path/to/hf-model \
  --megatron-checkpoint /path/to/actor-megatron \
  --reference-checkpoint /path/to/reference-megatron \
  --save-dir /path/to/saves/reflect \
  --search-url http://localhost:8002/retrieve \
  --mode joint --steps 100
```

Add `--execute` to launch. The default model preset is Qwen2.5-3B; select another
literal Slime model preset with `--model-args`. Resource counts, batch sizes and
token limits are CLI options. To select Fast-Slow, pass `--mode fast_slow
--fast-updates 3`. `--steps` counts outer rollout steps. Resume with
`--resume /path/to/checkpoints`, retaining the original topology and total
outer-step target. Configuration and checkpoints are written under `save-dir`;
choose a persistent output directory. Fresh runs reject nonempty outputs.

Retriever/generator transport failures and malformed service responses mark
the whole episode as retryable infrastructure failure with zero policy and
reflection masks. The partial tree remains available for diagnosis. Empty
search results remain valid outcomes. This adapts the reference harness to
Harness-RL's existing failure-isolation contract.

## Reference implementation versus paper

The reference repository explicitly differs from the final algorithm in
[LoongReflect](https://arxiv.org/abs/2608.11967). This port preserves those
differences rather than claiming a paper reproduction:

| Component | Retained implementation used here | Paper's final method |
| --- | --- | --- |
| Teacher | Answer-aware current-branch prompt | Answer-masked global history and control hints |
| Credit | Dense best-descendant tree credit | Terminal outcome GRPO on the active path |
| Fast mask | Reflection only | Reflection and backtrack spans |
| Coordination | Sequential fast updates, then policy | Look-ahead, conflict projection, restore and fused commit |
| Training answer checks | Reference-based retry feedback | Terminal-outcome credit |

Evaluation routes to the label-blind adapter: no reference-based retries are
used during generation. The retained evaluation metric named `f1` is reference
token recall, not harmonic precision/recall F1, and legacy summaries exclude
incomplete outputs. Report these distinctions and exclusions when using that
adapter. No held-out benchmark score or historical score is asserted here.

## Validation

```bash
TORCHDYNAMO_DISABLE=1 python -m pytest tests -q -o addopts=''
LOONGREFLECT_TEST_DEVICE=cuda TORCHDYNAMO_DISABLE=1 \
  python -m pytest tests/test_loongreflect_integration.py -q -o addopts=''
```

The integration tests execute the production Slime sample converter,
advantage hook, policy loss, custom-loss dispatcher and autograd on a small
causal Transformer. They check real Joint/Fast-Slow optimizer updates, fixed
teacher/old-policy probabilities, masks and EMA checkpoint round trips. They
also execute the production actor orchestration and teacher-forward methods
with external services replaced. A singleton topology shim substitutes for
Megatron process-group queries in these tests; it is not used in training.
Full Megatron/SGLang model serving and distributed checkpoint conversion need
the GPU runtime and production checkpoints and are a separate validation step.

Validation of this port: 305 CPU checks passed with one existing skip, including
the original Harness-RL/OLE/CAPO tests and Slime rollout scheduling checks.
Four independent CUDA runs passed all 44 integration checks in total, including
the real loss/optimizer tests in all four training modes. These are small-model
checks; they do not establish full-scale Megatron/SGLang training correctness
or benchmark quality.
