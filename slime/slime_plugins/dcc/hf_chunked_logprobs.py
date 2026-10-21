"""Optional Qwen3 MoE raw completion scores with bounded LM-head logits.

Not the default policy scorer. Preserves full prompt/completion tokens and raw
SUM/mean semantics; chunking is over response head positions only. Requires
standard attention LoRA or an unwrapped HF Qwen3 MoE model, without PEFT prompt
learning/modules-to-save. Decoder checkpointing remains the caller's setting.
"""
import torch
from torch.utils.checkpoint import checkpoint


def completion_logprobs_chunked(model, prompt_ids, completion_ids, *, head_chunk_tokens=128):
    from .hf_compute import compute_context
    with compute_context(model):
        return _completion_logprobs_chunked(model, prompt_ids, completion_ids,
                                           head_chunk_tokens=head_chunk_tokens)


def _completion_logprobs_chunked(model, prompt_ids, completion_ids, *, head_chunk_tokens=128):
    if type(head_chunk_tokens) is not int or head_chunk_tokens < 1:
        raise ValueError("Positive integer head chunk size required")
    if not prompt_ids or not completion_ids:
        raise ValueError("Nonempty prompt and complete response required")
    if any(type(token) is not int or token < 0 for token in list(prompt_ids) + list(completion_ids)):
        raise ValueError("Actual nonnegative integer token IDs required")
    peft = getattr(model, "peft_config", None)
    if peft is not None:
        for config in peft.values():
            if (str(config.peft_type).split(".")[-1] != "LORA"
                    or config.modules_to_save or config.is_prompt_learning
                    or getattr(config, "use_dora", False)
                    or getattr(config, "alora_invocation_tokens", None) is not None
                    or set(config.target_modules or ()) != {"q_proj", "v_proj", "o_proj"}
                    or getattr(config, "target_parameters", None)
                    or getattr(config, "trainable_token_indices", None)
                    or getattr(config, "layer_replication", None)):
                raise ValueError("Chunked scorer requires standard q/v/o LoRA; aLoRA and other extensions are unsupported")
        base = model.get_base_model()
    else:
        base = model
    if getattr(base.config, "model_type", None) != "qwen3_moe":
        raise ValueError("Chunked scorer is approved only for HF Qwen3 MoE")
    if getattr(base.config, "pretraining_tp", 1) != 1:
        raise ValueError("Tensor-parallel LM head is not supported")
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=model.device)
    # Standard injected LoRA layers, including disable_adapter state, reside in
    # this actual decoder. No top-level prompt tuning or adapter_names routing.
    # aLoRA is rejected: its top-level forward injects invocation offsets that
    # direct decoder forwarding cannot reproduce.
    hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    length = len(completion_ids)
    response = hidden[:, -(length + 1):-1, :]
    target = ids[:, -length:]
    head = base.get_output_embeddings()

    def score_chunk(value, labels):
        logits = head(value).float()
        return logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)

    chunks = []
    for start in range(0, length, head_chunk_tokens):
        value, labels = response[:, start:start + head_chunk_tokens, :], target[:, start:start + head_chunk_tokens]
        score = (checkpoint(score_chunk, value, labels, use_reentrant=False)
                 if torch.is_grad_enabled() else score_chunk(value, labels))
        chunks.append(score)
    return torch.cat(chunks, dim=1)[0]


BACKEND_RECIPE_VERSION = "dcc-logprob-backend-recipe-v1"
ORIGINAL_IMPLEMENTATION = "hf-response-logits-fp32-raw-logprob-v1"
CHUNKED_IMPLEMENTATION = "qwen3-moe-checkpointed-head-fp32-raw-logprob-v1"


def resolve_logprob_backend(backend="original", head_chunk_tokens=None):
    """Select one explicit implementation; original scoring remains default.

    The returned public recipe is saved beside the actual training source
    identity. Mathematical raw-completion SUM semantics are unchanged.
    """
    from functools import partial
    from .hf_policy import completion_logprobs

    if backend == "original":
        if head_chunk_tokens is not None:
            raise ValueError("Head chunks require explicit qwen3-moe-chunked backend")
        scorer, implementation, chunks = completion_logprobs, ORIGINAL_IMPLEMENTATION, None
    elif backend == "qwen3-moe-chunked":
        chunks = 128 if head_chunk_tokens is None else head_chunk_tokens
        if type(chunks) is not int or chunks < 1:
            raise ValueError("Positive integer head chunk size required")
        scorer = partial(completion_logprobs_chunked, head_chunk_tokens=chunks)
        implementation = CHUNKED_IMPLEMENTATION
    else:
        raise ValueError("Unknown logprob backend")
    return scorer, {"version": BACKEND_RECIPE_VERSION, "backend": backend,
                    "implementation_version": implementation, "head_chunk_tokens": chunks}
