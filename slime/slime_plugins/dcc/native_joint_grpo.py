"""Opt-in CUAD GRPO over raw thoughts and actually sampled constrained JSON.

The original thought-only runtime is unchanged. JSON old scores come only from
post-grammar generation scores; current/reference scores use the same legal
prefix support. Greedy JSON traces are intentionally not accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from .native_behavior_logprobs import StreamingRawThoughtLogprobs
from .native_hf_context import validated_thought_step

ALGORITHM = "online-native-HF-joint-thought-masked-JSON-GRPO-v1"
JSON_BEHAVIOR_SOURCE = "streaming-generation-post-xgrammar-multinomial-T1-v1"
JSON_SURROGATE_VARIANT = "joint-JSON-weighted-token-GRPO-v1"


def validate_json_surrogate_weight(weight, *, joint_json_policy, policy_surrogate="token-grpo",
                                   online_web=False, benchmarks=()):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("JSON surrogate weight must be finite and positive")
    if weight != 1 and (not joint_json_policy or policy_surrogate != "token-grpo"
            or online_web or any(benchmark != "cuad" for benchmark in benchmarks)):
        raise ValueError("Nondefault JSON surrogate weight requires CUAD joint JSON token-GRPO only")


def json_surrogate_configuration(weight):
    return dict(version=JSON_SURROGATE_VARIANT, thought_weight=1, json_weight=weight,
        scope="token-clipped-policy-surrogate-only; reference-KL-unweighted",
        denominator="group_size*max_steps*(thought_tokens+json_tokens)",
        observed_length_normalization=False, behavior_and_grammar_unchanged=True)


def weighted_joint_policy_loss(current, old, advantage, *, thought_count, group_size,
                               max_steps, thought_tokens, json_tokens, json_surrogate_weight,
                               clip=0.2):
    """Weight JSON surrogates after token clipping; retain the fixed two-phase budget.

    Actual phase boundaries include their real stop tokens. Empty or truncated
    JSON contributes only its sampled positions; no missing token is filled in.
    The weight-one runtime deliberately uses the original loss instead.
    """
    validate_json_surrogate_weight(json_surrogate_weight, joint_json_policy=True)
    if current.ndim != 1 or old.shape != current.shape or not current.numel():
        raise ValueError("Aligned nonempty one-dimensional joint logprobs required")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Finite actual behavior and current joint scores required")
    if old.requires_grad or not math.isfinite(advantage) or not 0 < clip < 1:
        raise ValueError("Detached old scores, finite advantage and clip in (0,1) required")
    if any(type(v) is not int or v < 1 for v in (group_size, max_steps, thought_tokens, json_tokens)):
        raise ValueError("Positive fixed joint objective budgets required")
    if (type(thought_count) is not int or not 0 <= thought_count <= min(thought_tokens, current.numel())
            or current.numel() - thought_count > json_tokens):
        raise ValueError("Actual phase boundary must fit both declared sampled budgets")
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    weights = torch.ones_like(surrogate)
    weights[thought_count:] = json_surrogate_weight
    loss = -(surrogate * weights).sum() / (group_size * max_steps * (thought_tokens + json_tokens))
    if not torch.isfinite(loss):
        raise FloatingPointError("Finite weighted joint policy loss required")
    return loss


class StreamingMaskedJSONLogprobs(StreamingRawThoughtLogprobs):
    """Observe scores after xgrammar, with all later warpers neutral.

    Reuse bounded storage and actual selected-token alignment, including the
    first real EOS. This class has a separate behavior-source identifier and
    must never be used for greedy decoding or before another probability warp.
    """


@dataclass(frozen=True)
class JointSegment:
    prompt_ids: list[int]
    completion_ids: list[int]
    thought_logprobs: list[float]
    json_logprobs: list[float]
    thought_count: int
    json_public_schema: dict
    json_stop_ids: list[int]

    @property
    def behavior_logprobs(self):
        return self.thought_logprobs + self.json_logprobs


def joint_group_signal(results, *, group_size, max_steps, thought_tokens, json_tokens,
                       eos_ids, advantage_normalization="std"):
    from .hf_native_policy import END_THINK, JointNativeHFCompletion
    from .hf_native_structured import native_structured_schema_kind
    from .online_grpo import group_signal

    if type(json_tokens) is not int or json_tokens < 1 or any(r.benchmark != "cuad" for r in results):
        raise ValueError("Joint JSON policy requires positive JSON budget and CUAD-only episodes")
    signal = group_signal(results, group_size=group_size, max_steps=max_steps,
        thought_tokens=thought_tokens, advantage_normalization=advantage_normalization)
    if signal.discarded_reason is not None:
        return signal
    groups = []
    for result in results:
        segments = []
        for step in result.steps:
            if not isinstance(step, JointNativeHFCompletion):
                raise ValueError("Fresh stochastic JSON completions required; greedy traces are unsupported")
            prompt, thought, behavior = validated_thought_step(step, thought_tokens=thought_tokens)
            final = list(step.completion_ids[len(thought):])
            values = step.json_logprobs
            if (len(final) > json_tokens or tuple(step.phase_lengths) != (len(thought), len(final))
                    or type(values) is not list or len(values) != len(final)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-6 for v in values)
                    or step.json_logprobs_source != (JSON_BEHAVIOR_SOURCE if final else None)
                    or step.json_stop_ids != list(eos_ids)
                    or not eos_ids or END_THINK in eos_ids
                    or any(type(t) is not int or t < 0 for t in final)
                    or native_structured_schema_kind(step.json_public_schema) != "cuad"):
                raise ValueError("Aligned actual masked JSON behavior, original schema and stop IDs required")
            if final and (not thought or thought[-1] != END_THINK or thought.count(END_THINK) != 1
                          or set(thought).intersection(eos_ids) or set(final[:-1]).intersection(eos_ids)):
                raise ValueError("Actual thought close and first-real-EOS JSON boundaries required")
            if step.native_status == "completed" and (not final or final[-1] not in eos_ids):
                raise ValueError("Successful JSON must retain its actual EOS")
            if thought:
                segments.append(JointSegment(prompt, thought + final, behavior, list(values),
                    len(thought), step.json_public_schema, list(step.json_stop_ids)))
        groups.append(segments)
    return replace(signal, segments=groups)


def json_prefix_bitmasks(compiler, schema, tokens, *, eos_ids):
    """Rebuild the generation support before each actual JSON token, on CPU.

    Only packed masks are retained. No sampled token is filled in or retried.
    A truncated legal JSON prefix is valid input; an illegal token fails closed.
    """
    import xgrammar as xgr
    from .hf_native_structured import compile_native_structured_schema, native_structured_schema_kind

    if native_structured_schema_kind(schema) != "cuad":
        raise ValueError("Joint JSON replay supports exact CUAD public schemas only")
    grammar = compile_native_structured_schema(compiler, schema)
    # xgrammar normalizes stop-ID order; retain the original order in traces.
    if sorted(grammar.tokenizer_info.stop_token_ids) != sorted(eos_ids):
        raise ValueError("Replay compiler stop IDs differ from the actual generation stop IDs")
    vocab_size = grammar.tokenizer_info.vocab_size
    bitmasks = xgr.allocate_token_bitmask(len(tokens), vocab_size)
    matcher = xgr.GrammarMatcher(grammar)
    for index, token in enumerate(tokens):
        if type(token) is not int or not 0 <= token < vocab_size or matcher.is_terminated():
            raise ValueError("Out-of-range token or tokens after the actual JSON EOS")
        matcher.fill_next_token_bitmask(bitmasks, index)
        if not ((int(bitmasks[index, token // 32]) >> (token % 32)) & 1):
            raise ValueError("Actual JSON token is outside its grammar prefix support")
        if not matcher.accept_token(token):
            raise ValueError("Actual JSON token is rejected by the exact generation grammar")
    return bitmasks


def grammar_token_logprobs(logits, labels, bitmasks):
    """Differentiable normalization over a fixed legal-token support.

    Use PyTorch masked_fill in the current-policy graph, rather than relying on
    xgrammar's generation-only in-place CPU/Triton mutation for autograd.
    """
    vocab = logits.shape[-1]
    if (bitmasks.dtype != torch.int32 or bitmasks.ndim != 2
            or bitmasks.shape != (logits.shape[-2], (vocab + 31) // 32)):
        raise ValueError("Packed grammar masks must align to response positions and model vocabulary")
    packed = bitmasks.to(logits.device)
    ids = torch.arange(vocab, device=logits.device)
    # Advanced indexing owns this int32 buffer. Keep shifts int32 and in-place
    # so a vocabulary-wide int64 temporary does not double mask memory.
    words = packed[:, ids // 32]
    words.bitwise_right_shift_((ids % 32).to(torch.int32)).bitwise_and_(1)
    allowed = words.bool()
    masked = logits.float().masked_fill(~allowed, -torch.inf)
    return masked.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - masked.logsumexp(-1)


def segment_scorer(segment, compiler, *, backend, head_chunk_tokens=None):
    """Create one immutable mask set for current and frozen-base reference.

    Thought positions retain raw normalization. Both phases share one decoder
    forward; head chunking bounds expanded vocabulary masks and FP32 logits.
    """
    from .hf_chunked_logprobs import completion_logprobs_chunked
    from .hf_compute import compute_context

    count = segment.thought_count
    bitmasks = json_prefix_bitmasks(compiler, segment.json_public_schema,
        segment.completion_ids[count:], eos_ids=segment.json_stop_ids)
    if backend not in {"original", "qwen3-moe-chunked"}:
        raise ValueError("Unknown joint JSON scorer backend")

    def score(model, prompt_ids, completion_ids):
        if prompt_ids != segment.prompt_ids or completion_ids != segment.completion_ids:
            raise ValueError("Joint scorer must retain its exact full prompt and sampled two-stage tokens")
        if backend == "qwen3-moe-chunked":
            return completion_logprobs_chunked(model, prompt_ids, completion_ids,
                head_chunk_tokens=128 if head_chunk_tokens is None else head_chunk_tokens,
                json_bitmasks=bitmasks, mask_start=count)
        ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
        length = len(completion_ids)
        with compute_context(model):
            output = model(input_ids=ids, use_cache=False, logits_to_keep=length + 1)
        logits = output.logits[0, :-1].float()
        labels = ids[0, -length:]
        raw = logits[:count].gather(-1, labels[:count, None]).squeeze(-1) - logits[:count].logsumexp(-1)
        if length == count:
            return raw
        return torch.cat((raw, grammar_token_logprobs(logits[count:], labels[count:], bitmasks)))
    return score
