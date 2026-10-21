# Grounded CUAD final-answer supervision

`scripts/dcc_grounded_final_builder.py` builds an independent TRAIN annotation
dataset from a closed native CUAD collection. It keeps the original prompt and
the complete, naturally closed thought as masked conditioning. The supervised
target is a new JSON answer copied from observed source spans plus a training
EOS. These targets are synthetic reference answers, not accepted native
completions. This is supervised fine-tuning, not RL.

The builder validates the original collection and physical traces before
opening TRAIN labels. Every positive target must contain the complete gold set
within one actual request's effective source observations. Whitespace matching
maps back to exact original document offsets. Combining contiguous observed
chunks is an explicit opt-in. Empty targets use public TRAIN annotations;
partial observations do not establish absence. The existing native selector is
unchanged. There is no held-out data access, truncation or partial-gold fallback.

Build with `build --request REQUEST.json --dataset DATASET`, then use `check`
with the same arguments to independently rederive and compare the dataset.
The request names the parent source root and commit, profile, closed collection,
world size, collection ID, TRAIN labels, local tokenizer, sequence limit and
minimum positive/negative counts. The manifest reports all exclusions and
refuses training if any otherwise eligible full sequence is overlength.

`scripts/dcc_grounded_final_sft.py` requires a separate, caller-reviewed
five-field approval from `training_binding`. The actual trainer must run from
its clean committed checkout. Training-code and parent-collection commits are
separate. Local model metadata must identify the same TRAIN-pruned Qwen3 MoE
derivative and source revision, with fewer than 30 billion parameters.

Run `--cpu-preflight` before launching with torchrun. Defaults are fresh NF4
q/v/o LoRA, rank16/alpha32, dropout0, 32 updates and learning rate1e-5. Each rank
contributes one complete example per update. The global sampler uses uniform
task decks with a 2:1 positive/negative quota. Prompt, thought, review fields
and placeholder confidence values receive no direct loss; unmasked answer JSON
and EOS receive per-example mean token loss. All ranks average example gradients.
Checkpoints include initial, every eight updates, final and completion markers.

Evaluation uses `--corrected-lora-approval APPROVAL.json` together with
`--adapter CHECKPOINT` and `--native-lora-derivative-id DERIVATIVE`. Keep the
ordinary native-thinking and newly frozen evaluation-source receipt arguments.
The native accepted-only and DPO approval flags are mutually exclusive with
this approval. Saved configuration, actual attention tensor headers and the
checkpoint's completed update count are checked again during comparison.
The recorded `corrected_lora_binding` distinguishes collection, training and
evaluation sources. A step-zero checkpoint can test loading; it is not a
trained result.

Compare a fresh base run against the adapted run using the same evaluation
source, model, tasks, harness, budgets and sampling configuration. Neither
successful training nor reduced TRAIN loss establishes benchmark improvement.
