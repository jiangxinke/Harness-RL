# Optional vLLM evaluation

The vLLM backend accelerates greedy benchmark generation with fused MoE
inference and continuous batching. HF remains the default for generation and
the training runtime. This backend does not compute trainable log probabilities
and rejects nonzero sampling temperatures.

Use a separate environment with a CUDA-compatible driver:

```bash
python -m venv "$VLLM_ENV"
"$VLLM_ENV/bin/python" -m pip install -r configs/dcc/requirements-vllm.txt
CUDA_VISIBLE_DEVICES='' "$VLLM_ENV/bin/python" -c \
  'from vllm import LLM, SamplingParams; from vllm.lora.request import LoRARequest'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=slime "$VLLM_ENV/bin/python" -m pytest -q \
  tests/test_dcc_vllm_policy.py
```

The environment pins vLLM 0.11.2 and Transformers 4.57.6. vLLM pins its own
compatible PyTorch version; installing it into the BF16 or NF4 training
environment would replace those dependencies. Reuse the existing base model
cache and local adapter checkpoint. No model merge or duplicate base checkpoint
is needed. The upstream Qwen3 MoE implementation supports LoRA and maps q/v
adapters into its fused qkv projection while retaining the o projection.
This backend accepts local attention-only PEFT adapters with ranks up to 64;
the current training configuration uses q/v/o with rank 8.

The Python integration API is:

```python
from slime_plugins.dcc.vllm_policy import load_vllm_policy

policy = load_vllm_policy(
    model_name,
    adapter=adapter_path,  # None for the base policy
    batch_size=8,
    max_prompt_tokens=16384,
    max_model_len=24576,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.88,
    max_num_batched_tokens=8192,
    seed=42,
)
# Inside an async function:
completion = await policy(messages, max_new_tokens=8192, temperature=0)
await policy.close()
```

Use one launcher process, without multi-rank `torchrun`. Select exactly as many
visible GPUs as `tensor_parallel_size`: TP=1 requires one visible GPU; a single
TP engine may use several explicitly assigned GPUs. The loader rejects a
nonzero launcher/local rank, a launcher world size other than one, mismatched
GPU visibility, or an existing distributed process group. It temporarily clears
launcher rendezvous variables during engine creation, restores them afterward,
uses spawned workers, and explicitly selects the pinned V1 uni/MP executor.
These executors assign worker ranks and a fresh TCP rendezvous internally;
the `external_launcher` executor is not used.

Set GPU visibility before creating an engine. Guard standalone launchers with
`if __name__ == "__main__":` so worker subprocesses cannot rerun the launcher.
The queue serializes engine access and batches requests. `close()` drains the
queue, then calls V1 `EngineCoreClient.shutdown()` to finalize its background
resources and engine/worker subprocesses. It is idempotent. Run each development
comparison in a dedicated process and verify exit before reassigning its GPUs.
Allow the run owner to schedule GPU memory and concurrency.

Prompt tokenization and truncation call the same `encode_messages` helper as
HFPolicy, with thinking disabled. The engine receives those exact token IDs.
Prompt plus requested completion must fit `max_model_len`; an insufficient
context budget raises an error and never silently shortens generation. KV
cache uses the BF16 model dtype. LoRA dtype is explicitly `auto`, which resolves
to BF16 with this base. HF/PEFT can store and compute its adapter branch in
FP32; the vLLM adapter computation can therefore differ even with the same
checkpoint. Validate on development tasks and retain that precision provenance.
Prefix caching and chunked prefill are enabled;
`max_num_batched_tokens` limits each scheduling iteration. The memory fraction
includes the base weights, engine allocations and KV cache. Measure actual
memory and throughput before increasing concurrency.

Save `policy.provenance` beside the prediction manifest in a new external output
directory. It records model/tokenizer revisions, resolved model revision when
available, runtime versions, adapter configuration/path/size/modification time,
precision, seed, context/token budgets, memory fraction and cache settings.
Adapter metadata identifies the supplied local checkpoint without expensive
weight hashing. Actual prompt/completion IDs and finish reasons are retained
in `PolicyCompletion`; log probabilities are absent. Changing the backend or
adapter requires a new output directory. Runtime paths belong in external
artifacts, not committed configuration.

Validate base and adapter generation on independent web or CUAD development
tasks with identical semantic budgets before a full run. Compare wall-clock
time, generated tokens, answer validity and actual evidence use. Fused kernels
can produce different greedy outputs from HF BF16. Evaluating an NF4-trained
adapter on this BF16 base also changes base precision and must be reported as
such; it is not a continuation of an NF4 evaluation. CPU adapter and batching
tests do not establish GPU compatibility or a speedup.

Upstream implementation reference:
[vLLM 0.11.2 Qwen3 MoE](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/models/qwen3_moe.py).
The executor rank/rendezvous and shutdown behavior are defined in
[UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/executor/uniproc_executor.py),
[MultiprocExecutor](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/executor/multiproc_executor.py),
and [EngineCoreClient](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/engine/core_client.py).
