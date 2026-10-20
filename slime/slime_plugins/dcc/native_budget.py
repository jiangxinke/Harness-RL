"""Explicit native thought budgets; v2 defaults stay unchanged."""
from __future__ import annotations
from dataclasses import dataclass
from .eval_sampling import EvaluationSampling

LEGACY_PROTOCOL = 'native-thinking2507-real-close-two-stage-4096-4096-v2'
BUDGET_PROTOCOL = 'native-thinking2507-real-close-two-stage-budgeted-v3'


@dataclass(frozen=True, slots=True)
class NativeBudget:
    thought_tokens: int = 4096
    thought_sampling_profile: str = 'greedy'

    def __post_init__(self):
        if type(self.thought_tokens) is not int or self.thought_tokens not in (4096, 8192, 16384):
            raise ValueError('Native thought budget must be explicitly 4096, 8192 or 16384')
        if self.thought_sampling_profile not in ('greedy', 'thinking'):
            raise ValueError('Native thought sampling must be greedy or thinking')

    @property
    def legacy(self):
        return self.thought_tokens == 4096 and self.thought_sampling_profile == 'greedy'

    @property
    def protocol(self):
        return LEGACY_PROTOCOL if self.legacy else BUDGET_PROTOCOL

    @property
    def total_tokens(self):
        return self.thought_tokens + 4096

    @property
    def thought_sampling(self):
        return (EvaluationSampling() if self.thought_sampling_profile == 'greedy' else
                EvaluationSampling(temperature=0.6, top_p=0.95, top_k=20))


def effective_sampling(sampling):
    """Pinned vLLM 0.11.2 normalizes greedy top_k=-1 to 0 (disabled)."""
    result = sampling.to_dict()
    if sampling.temperature == 0:
        result.update(top_p=1.0, top_k=0, min_p=0.0)
    return result


def budget_from_configuration(config):
    if config.get('protocol') == LEGACY_PROTOCOL:
        return NativeBudget()
    if config.get('protocol') == BUDGET_PROTOCOL:
        budget = NativeBudget(config.get('thought_max_tokens'), config.get('thought_sampling_profile'))
        if budget.legacy:
            raise ValueError('Default 4096+4096 greedy uses the original v2 protocol')
        return budget
    raise ValueError('Unsupported native evaluation protocol')


def approved_training_receipt(evaluation_receipt, source_commit, protocol):
    """Caller inputs approve actual training separately from evaluation source.

    Values never come from adapter self-claims. The real training config must
    subsequently match this complete expected receipt and protocol exactly.
    """
    from .native_thinking_eval import validate_source_receipt
    if source_commit is None and protocol is None:
        return None
    if source_commit is None or protocol not in {LEGACY_PROTOCOL, BUDGET_PROTOCOL}:
        raise ValueError('Explicit native LoRA training source and supported protocol are required together')
    receipt = dict(evaluation_receipt, source_commit=source_commit)
    return validate_source_receipt(receipt)
