"""CPU draw-level controls; fixtures contain no benchmark labels or completions."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1] / "slime")]

import pytest
import torch

from dcc_native_dpo import (
    OfflineNativeDPOTrainer, ReferenceIdentity, _sequence_sum,
    backward_pair_low_memory, cache_reference_scores, dpo_loss,
)
from dcc_native_dpo_data import NativeFinalCandidate, NativeSequence, PreferencePair, PublicFinalPrefix
from dcc_native_dpo_train import GlobalPairDeck, parser, run_dpo_loop, synchronize_pair_gradients
from slime_plugins.dcc.native_dpo_sampling import (
    DOMAIN_BALANCED_PAIR_SAMPLING, DomainBalancedPairDeck, balanced_pair_sampling_configuration,
)


def pairs():
    """Real G5 domain counts (14/3), independently invented token fixtures."""
    result = []
    for i in range(17):
        domain = "cuad" if i < 14 else "widesearch"
        prefix = PublicFinalPrefix(f"prefix-{i}", f"task-{i}", domain,
            "cuad_train" if domain == "cuad" else "independent_curriculum",
            '[{"role":"user","content":"invented CPU fixture"}]',
            '{"type":"object","properties":{"answer":{"type":"array"}},"required":["answer"],"additionalProperties":false}',
            "CPU-fixture-harness", '{"identity":"CPU-fixture"}')
        candidates = []
        for branch, reward in ((0, 1.0), (1, 0.0)):
            seed = 2 * i + branch
            sequence = NativeSequence((i + 1,),
                (i + 2 + 3 * branch, 151668, i + 4 + branch, 151645), (2, 2))
            candidates.append(NativeFinalCandidate(prefix, sequence, f"trace-{seed}",
                f"episode-{seed}", seed, '{"identity":"CPU-fixture"}', reward, "CPU-TRAIN-reward"))
        result.append(PreferencePair(*candidates))
    return tuple(result)


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.linspace(-0.03, 0.04, 11, dtype=torch.float64))
        self.register_buffer("base", torch.linspace(-0.8, 0.8, 11, dtype=torch.float64))
        self.disabled = False
        self.reference_prompts = []
        self.calls = 0

    @contextmanager
    def disable_adapter(self):
        before, self.disabled = self.disabled, True
        try:
            yield
        finally:
            self.disabled = before

    def save_pretrained(self, destination):
        # Whole-loop publication is exercised without pretending to save PEFT.
        (destination / "synthetic_adapter.json").write_text(json.dumps(self.theta.detach().tolist()))


class TinyTokenizer:
    def save_pretrained(self, destination):
        (destination / "synthetic_tokenizer.json").write_text('{"CPU_fixture":true}\n')


def score(model, prompt, completion):
    model.calls += 1
    if model.disabled:
        model.reference_prompts.append(tuple(prompt))
    values = model.base if model.disabled else model.base + model.theta * (1 + prompt[0] % 3)
    ids = torch.tensor([token % 11 for token in completion], dtype=torch.long)
    return values.log_softmax(0)[ids]


def identity():
    return ReferenceIdentity("CPU-base", "CPU-tokenizer", "CPU-float64-test", "base")


def optimizer(parameters):
    return torch.optim.AdamW(parameters, lr=0.003, weight_decay=0.0)


def direct_gradient(model, draws, references):
    model.zero_grad(set_to_none=True)
    loss = sum(dpo_loss(_sequence_sum(model, pair.chosen.sequence, score),
        _sequence_sum(model, pair.rejected.sequence, score), references[pair.key], beta=0.1)
        for pair in draws) / len(draws)
    loss.backward()
    return loss.detach(), model.theta.grad.detach().clone()


def test_fixed_two_domain_batches_cover_every_pair_uniformly_and_deterministically():
    rows = pairs()
    first, repeat, other = [DomainBalancedPairDeck(rows, seed=seed) for seed in (47, 47, 48)]
    counts, observed_other = Counter(), []
    for _ in range(350):
        draw = first.next(8)
        assert draw == repeat.next(8)
        observed_other.append(draw != other.next(8))
        assert len(draw) == 8
        assert sum(i < 14 for i in draw) == sum(i >= 14 for i in draw) == 4
        counts.update(draw)
        info = first.describe(draw)
        assert info["global_pair_draws"] == 8
        assert info["global_unique_pairs"] == len(set(draw)) < 8
        assert sum(row["draws"] for row in info["pair_multiplicities"]) == 8
        assert info["domain_draw_counts"] == {"cuad": 4, "widesearch": 4}
        # Unequal/empty owners preserve every draw and its multiplicity.
        partitions = [[i for i in draw if i % 8 == rank] for rank in range(8)]
        assert sum((Counter(part) for part in partitions), Counter()) == Counter(draw)
    assert any(observed_other)
    assert set(counts) == set(range(17))
    for indices in (range(14), range(14, 17)):
        assert max(counts[i] for i in indices) - min(counts[i] for i in indices) <= 1
    assert sum(counts[i] for i in range(14)) == sum(counts[i] for i in range(14, 17)) == 1400


@pytest.mark.parametrize("size", [0, 1, 3, True, 2.0])
def test_balanced_batch_rejects_invalid_or_unequal_quota(size):
    with pytest.raises(ValueError, match="positive even"):
        DomainBalancedPairDeck(pairs(), seed=47).next(size)


def test_requires_both_domains_and_retains_legacy_default_schedule():
    with pytest.raises(ValueError, match="nonempty CUAD and WideSearch"):
        DomainBalancedPairDeck(pairs()[:14], seed=47)
    assert parser().get_default("pair_sampling") == "global-deck"
    legacy = GlobalPairDeck(17, seed=47)
    assert [len(legacy.next(8)) for _ in range(6)] == [8, 8, 1, 8, 8, 1]


def test_explicit_repetitions_match_direct_per_draw_gradient_loss_and_adam_updates():
    rows = pairs()
    deck = DomainBalancedPairDeck(rows, seed=47)
    actual = TinyAdapter()
    expected = deepcopy(actual)
    refs = cache_reference_scores(expected, rows, identity(), logprob_fn=score)
    trainer = OfflineNativeDPOTrainer(actual, rows, identity(), logprob_fn=score,
        optimizer_factory=optimizer, allow_repeated_pairs=True)
    opt = optimizer(expected.parameters())
    for _ in range(3):
        indices = deck.next(8)
        draws = [rows[i] for i in indices]
        opt.zero_grad(set_to_none=True)
        expected_loss, gradient = direct_gradient(expected, draws, refs)
        expected_norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
        opt.step()
        actual.calls = 0
        result = trainer.step(draws)
        assert result["local_pair_draws"] == 8
        assert result["local_unique_pairs"] == len(set(indices))
        assert actual.calls == 4 * len(set(indices))  # two preview + two replay, once per unique pair.
        assert result["dpo_loss_sum"] / 8 == pytest.approx(expected_loss.item(), abs=2e-7)
        assert result["gradient_norm"] == pytest.approx(float(expected_norm), rel=3e-6, abs=2e-7)
        torch.testing.assert_close(actual.theta, expected.theta, rtol=3e-6, atol=2e-8)
        torch.testing.assert_close(actual.theta.grad, expected.theta.grad, rtol=3e-6, atol=2e-7)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(trainer.optimizer.state[actual.theta][key],
                opt.state[expected.theta][key], rtol=3e-6, atol=2e-8)
    assert all(ref.identity == identity() for ref in trainer.references.values())
    assert not actual.disabled


def test_default_trainer_still_refuses_repeated_pairs():
    rows = pairs()
    trainer = OfflineNativeDPOTrainer(TinyAdapter(), rows, identity(), logprob_fn=score,
                                      optimizer_factory=optimizer)
    with pytest.raises(ValueError, match="Duplicate pair"):
        trainer.step([rows[0], rows[0]])


def test_eight_owner_partitions_all_reduce_every_rank_and_match_one_global_adam_step():
    rows, world = pairs(), 8
    indices = DomainBalancedPairDeck(rows, seed=47).next(8)
    reference_model = TinyAdapter()
    refs = cache_reference_scores(reference_model, rows, identity(), logprob_fn=score)
    global_model = deepcopy(reference_model)
    _, expected_gradient = direct_gradient(global_model, [rows[i] for i in indices], refs)
    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 1.0)
    global_optimizer = optimizer(global_model.parameters())
    global_optimizer.step()
    local_draws = [[rows[i] for i in indices if i % world == rank] for rank in range(world)]
    assert any(not batch for batch in local_draws)
    expected_local = []
    for batch in local_draws:
        model = TinyAdapter()
        for pair, multiplicity in Counter(batch).items():
            backward_pair_low_memory(model, pair, refs[pair.key], beta=0.1,
                logprob_fn=score, scale=multiplicity * world / len(indices))
        expected_local.append(model.theta.grad.clone() if model.theta.grad is not None
                              else torch.zeros_like(model.theta))
    reduced_sum = torch.stack(expected_local).sum(0)
    torch.testing.assert_close(reduced_sum / world, expected_gradient, rtol=3e-6, atol=2e-7)
    all_reduce_calls = []
    for rank in range(world):
        model = TinyAdapter()
        def all_reduce(tensor, op, rank=rank):
            torch.testing.assert_close(tensor, expected_local[rank], rtol=3e-6, atol=2e-7)
            all_reduce_calls.append(rank)
            tensor.copy_(reduced_sum)
        distributed = SimpleNamespace(all_reduce=all_reduce, ReduceOp=SimpleNamespace(SUM="SUM"))
        trainer = OfflineNativeDPOTrainer(model, rows[rank::world], identity(), logprob_fn=score,
            optimizer_factory=optimizer, world_size=world, allow_repeated_pairs=True,
            gradient_sync=lambda parameters, count, distributed=distributed:
                synchronize_pair_gradients(parameters, count, distributed=distributed))
        assert len(trainer.references) == len(rows[rank::world])
        assert Counter(model.reference_prompts) == Counter({pair.chosen.sequence.prompt: 2
                                                           for pair in rows[rank::world]})
        result = trainer.step(local_draws[rank], global_pair_count=8)
        assert result["optimizer_steps"] == 1 and result["global_pairs"] == 8
        torch.testing.assert_close(model.theta, global_model.theta, rtol=3e-6, atol=2e-8)
        assert list(trainer.optimizer.state.values())[0]["step"].item() == 1
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(trainer.optimizer.state[model.theta][key],
                global_optimizer.state[global_model.theta][key], rtol=3e-6, atol=2e-8)
    assert all_reduce_calls == list(range(world))


def test_opt_in_whole_loop_records_draw_multiplicity_and_sampler_configuration(tmp_path):
    output = tmp_path / "balanced"
    trainer = run_dpo_loop(TinyAdapter(), TinyTokenizer(), pairs(), identity(), output=output,
        steps=2, pairs_per_step=8, pair_sampling=DOMAIN_BALANCED_PAIR_SAMPLING,
        seed=47, learning_rate=0.003, logprob_fn=score)
    assert trainer.optimizer_steps == 2
    state = torch.load(output / "final" / "training_state.pt", weights_only=False)
    assert state["sampler_state"]["version"] == "native-two-domain-fixed-pair-draw-state-v1"
    assert set(state["sampler_state"]["remaining_decks"]) == {"cuad", "widesearch"}
    assert "sampler_rng" not in state and state["resume_supported"] is False
    cfg = json.loads((output / "final" / "training_config.json").read_text())
    assert cfg["pair_sampling"] == balanced_pair_sampling_configuration(seed=47, global_draws=8)
    steps = [json.loads(line) for line in (output / "train_rank0.jsonl").read_text().splitlines()
             if json.loads(line)["event"] == "step"]
    for row in steps:
        stats = row["pair_sampling_statistics"]
        assert row["global_pairs"] == stats["global_pair_draws"] == 8
        assert stats["global_unique_pairs"] < 8
        assert stats["domain_draw_counts"] == {"cuad": 4, "widesearch": 4}
        assert sum(item["draws"] for item in stats["pair_multiplicities"]) == 8


def test_legacy_loop_keeps_partial_tail_default_metadata_and_checkpoint_state(tmp_path):
    output = tmp_path / "legacy"
    run_dpo_loop(TinyAdapter(), TinyTokenizer(), pairs(), identity(), output=output,
        steps=3, pairs_per_step=8, seed=47, learning_rate=0.003, logprob_fn=score)
    cfg = json.loads((output / "final" / "training_config.json").read_text())
    assert "pair_sampling" not in cfg
    steps = [json.loads(line) for line in (output / "train_rank0.jsonl").read_text().splitlines()
             if json.loads(line)["event"] == "step"]
    assert [row["global_pairs"] for row in steps] == [8, 8, 1]
    assert all("pair_sampling_statistics" not in row and "local_pair_draws" not in row for row in steps)
    state = torch.load(output / "final" / "training_state.pt", weights_only=False)
    assert "sampler_rng" in state and "sampler_deck" in state and "sampler_state" not in state
    assert state["resume_supported"] is False
