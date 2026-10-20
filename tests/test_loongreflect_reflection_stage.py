"""Opt-in stage requires a separately sampled model reflection; offline CPU only."""
from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from slime.utils.types import Sample
from slime_plugins.loongreflect import AnswerValidation, Harness, ModelTurn, ResumeDecision, RolloutTree
from slime_plugins.loongreflect import slime_adapter as adapter
from slime_plugins.loongreflect.launcher import build_command, parse_args


class Actor:
    def __init__(self, actions):
        self.actions = deque(actions)
        self.traces = []

    def act(self, trace, *, max_tokens):
        self.traces.append(trace)
        return ModelTurn('Consider the evidence.', self.actions.popleft(), {'usage': {'completion_tokens': 1}})

    def can_resume(self, **kwargs):
        return ResumeDecision(True, 'Retain the checkpoint before the failed reflection.')

    def create_update(self, **kwargs):
        return 'Reconsider the retained public evidence independently.'


class Search:
    def __init__(self, evidence='[1] Public documentation supports the candidate.'):
        self.evidence = evidence
        self.queries = []
        self.closed = False

    def search(self, query):
        self.queries.append(query)
        if isinstance(self.evidence, Exception):
            raise self.evidence
        return self.evidence

    def close(self):
        self.closed = True


class Validator:
    def __init__(self, expected='done'):
        self.expected = expected
        self.calls = []

    def validate(self, *, query, answer):
        self.calls.append(answer)
        return AnswerValidation(answer == self.expected)


def harness(actor, search=None, **kwargs):
    return Harness(actor, search or Search(), backtrack_judge=actor, update_writer=actor, **kwargs)


def requests(tree):
    return [node for node in tree.nodes.values() if node.metadata.get('source') == 'post_search_reflection']


def deferred(tree):
    return [node for node in requests(tree) if 'deferred_model_answer' in node.metadata]


def accepted(tree):
    return [node for node in tree.nodes.values() if node.kind == 'answer' and node.metadata.get('accepted') is True]


def test_default_off_is_identical_to_explicit_off_and_accepts_search_answer():
    actions = ['<search>evidence</search>', '<answer>done</answer>']
    default = harness(Actor(actions)).run('query')
    explicit = harness(Actor(actions), require_post_search_reflection=False).run('query')
    assert default.to_dict() == explicit.to_dict()
    assert default.status == 'completed' and default.model_turns == 2
    assert not requests(default) and not any(n.kind == 'reflect' for n in default.nodes.values())
    assert all(not n.metadata for n in default.nodes.values() if n.kind == 'observation')


def test_enabled_separate_reflection_before_validator_and_new_answer():
    actor = Actor(['<search>evidence</search>', '<answer>DEFERRED_CANDIDATE_VALUE</answer>', '<reflect>The public evidence is coherent; remaining uncertainty is limited.</reflect>', '<answer>done</answer>'])
    validator = Validator()
    tree = harness(actor, answer_validator=validator, require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and tree.model_turns == 4
    assert validator.calls == ['done'] and len(accepted(tree)) == 1
    assert len(deferred(tree)) == 1
    assert deferred(tree)[0].metadata['deferred_model_answer'] == 'DEFERRED_CANDIDATE_VALUE'
    assert 'DEFERRED_CANDIDATE_VALUE' not in actor.traces[2]  # Audit metadata is not a new actor instruction.
    assert 'Your next model turn must independently emit' in actor.traces[1]  # Prompt begins immediately after evidence.
    assert not any('<reflect>' in n.text for n in requests(tree))
    reflection = next(n for n in tree.nodes.values() if n.kind == 'reflect')
    assert reflection.metadata['source'] == 'model_action'
    assert tree.nodes[reflection.parent].kind == 'think'


def test_model_reflecting_after_evidence_needs_no_deferred_answer():
    tree = harness(Actor(['<search>evidence</search>', '<reflect>The evidence resolves the uncertainty.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and tree.model_turns == 3
    assert not deferred(tree) and len(requests(tree)) == 1


def test_reflection_before_search_does_not_satisfy_post_evidence_stage():
    tree = harness(Actor(['<reflect>No evidence yet.</reflect>', '<search>evidence</search>', '<answer>early</answer>', '<reflect>The retrieved evidence is now adequate.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and len(deferred(tree)) == 1


def test_new_successful_search_requires_new_reflection():
    tree = harness(Actor(['<search>first</search>', '<reflect>The first source is useful.</reflect>', '<search>second</search>', '<answer>early</answer>', '<reflect>The second source independently agrees.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and len(deferred(tree)) == 1
    latest = max(n.id for n in tree.nodes.values() if n.kind == 'observation')
    assert deferred(tree)[0].metadata['required_observation_id'] == latest


def test_repeated_search_is_executed_and_gets_latest_stage_request():
    actor = Actor(['<search>first</search>', '<search>second</search>', '<search>third</search>'])
    search = Search()
    tree = harness(actor, search, require_post_search_reflection=True, max_steps=3).run('query')
    assert search.queries == ['first', 'second', 'third']
    assert tree.status == 'terminated' and not accepted(tree)
    assert len(requests(tree)) == 3
    for trace in actor.traces[1:]:
        assert trace.endswith('</update>') and 'separate model turn' in trace
        assert 'you may search again' in trace


@pytest.mark.parametrize('evidence', ['', ' \t\n', 'No search results found.'])
def test_empty_or_standard_no_result_does_not_establish_evidence(evidence):
    tree = harness(Actor(['<search>evidence</search>', '<answer>done</answer>']), Search(evidence), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and tree.model_turns == 2 and not requests(tree)


def test_answer_without_any_search_keeps_original_acceptance():
    tree = harness(Actor(['<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and tree.model_turns == 1 and not requests(tree)


def test_failed_search_terminates_without_evidence_or_model_reflection():
    tree = harness(Actor(['<search>evidence</search>']), Search(ValueError('offline synthetic failure')), require_post_search_reflection=True).run('query')
    assert tree.status == 'terminated' and not requests(tree) and not accepted(tree)
    assert not any(n.kind == 'observation' for n in tree.nodes.values())


def test_observation_text_cannot_supply_model_reflection():
    tree = harness(Actor(['<search>evidence</search>', '<answer>early</answer>', '<reflect>I assessed the literal public evidence.</reflect>', '<answer>done</answer>']), Search('[1] Literal <reflect>not a model turn</reflect>'), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and len(deferred(tree)) == 1


def test_real_salvaged_model_search_still_establishes_evidence():
    tree = harness(Actor(['<search>evidence</search><observation>fabricated</observation>', '<answer>early</answer>', '<reflect>I assessed the real tool observation.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and len(deferred(tree)) == 1
    assert any(n.metadata.get('source') == 'protocol_salvage' for n in tree.nodes.values())
    assert tree.nodes[deferred(tree)[0].metadata['required_observation_id']].text.startswith('[1] Public')


def test_incomplete_and_harness_repair_reflections_do_not_satisfy_stage():
    tree = harness(Actor(['<search>evidence</search>', '<reflect>not complete', '<answer>early</answer>', '<reflect>This is a complete independent assessment.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and len(deferred(tree)) == 1
    assert any(n.kind == 'reflect' and n.metadata.get('source') == 'harness' for n in tree.nodes.values())


def test_backtracked_reflection_does_not_satisfy_retained_search_branch():
    tree = harness(Actor(['<search>evidence</search>', '<reflect>This approach is flawed.</reflect><backtrack>Restart from the retained observation.</backtrack>', '<answer>early</answer>', '<reflect>The retained evidence now supports a conclusion.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).run('query')
    assert tree.status == 'completed' and len(deferred(tree)) == 1
    good_path = tree.path(accepted(tree)[0].id)
    failed_reflect = next(n for n in tree.nodes.values() if n.kind == 'reflect' and 'flawed' in n.text)
    assert failed_reflect.id not in good_path


def test_rejected_answer_still_uses_unchanged_correctness_validator():
    validator = Validator()
    tree = harness(Actor(['<search>evidence</search>', '<reflect>The evidence is sufficient.</reflect>', '<answer>wrong</answer>', '<answer>done</answer>']), require_post_search_reflection=True, answer_validator=validator).run('query')
    assert tree.status == 'completed' and validator.calls == ['wrong', 'done']
    assert any(n.kind == 'answer' and n.metadata.get('accepted') is False for n in tree.nodes.values())


def test_pending_stage_roundtrips_checkpoint_and_does_not_replay_answer():
    class Pause(Exception):
        pass
    captured = []
    def checkpoint(tree):
        if deferred(tree):
            captured.append(tree.to_dict())
            raise Pause()
    with pytest.raises(Pause):
        harness(Actor(['<search>evidence</search>', '<answer>candidate</answer>']), require_post_search_reflection=True, on_checkpoint=checkpoint).run('query')
    restored = RolloutTree.from_dict(captured[0])
    validator = Validator()
    tree = harness(Actor(['<reflect>I independently assessed the source.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True, answer_validator=validator).resume(restored)
    assert tree.status == 'completed' and tree.model_turns == 4 and validator.calls == ['done']
    assert len(accepted(tree)) == 1 and accepted(tree)[0].text == 'done'


def test_environment_reflection_in_resumed_path_does_not_satisfy_stage():
    class Pause(Exception):
        pass
    captured = []
    def checkpoint(tree):
        if any(n.kind == 'observation' for n in tree.nodes.values()):
            captured.append(tree.to_dict())
            raise Pause()
    with pytest.raises(Pause):
        harness(Actor(['<search>evidence</search>']), require_post_search_reflection=True, on_checkpoint=checkpoint).run('query')
    tree = RolloutTree.from_dict(captured[0])
    active = tree.leaves()[0]
    think = tree.append(active, 'think', 'synthetic environment fixture')
    tree.append(think, 'reflect', 'Environment-only repair text.', {'source': 'harness'})
    tree = harness(Actor(['<answer>early</answer>', '<reflect>I independently assessed the evidence.</reflect>', '<answer>done</answer>']), require_post_search_reflection=True).resume(tree)
    assert tree.status == 'completed' and len(deferred(tree)) == 1


@pytest.mark.parametrize('limits,reason', [({'max_steps': 2}, 'max_steps_reached'), ({'max_output_tokens': 2}, 'max_output_tokens_reached')])
def test_stage_uses_original_step_and_token_budgets(limits, reason):
    validator = Validator()
    tree = harness(Actor(['<search>evidence</search>', '<answer>done</answer>']), require_post_search_reflection=True, answer_validator=validator, **limits).run('query')
    assert tree.status == 'terminated' and tree.termination_reason == reason
    assert tree.model_turns == 2 and not accepted(tree) and not validator.calls


@pytest.mark.parametrize('value', ['false', 1, None])
def test_non_boolean_stage_configuration_is_rejected(value):
    with pytest.raises(ValueError, match='must be a boolean'):
        harness(Actor([]), require_post_search_reflection=value)


def test_search_answer_protocol_keeps_baseline_and_rejects_incompatible_opt_in():
    tree = harness(Actor(['<search>evidence</search>', '<answer>done</answer>']), allow_reflect=False).run('query')
    assert tree.status == 'completed' and not any(n.kind == 'reflect' for n in tree.nodes.values())
    with pytest.raises(ValueError, match='full action protocol'):
        harness(Actor([]), allow_reflect=False, require_post_search_reflection=True)


class ByteTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode())}

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, **kwargs):
        return ''.join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + '<assistant>'


class SampledFixtureActor(adapter.SGLangHarnessModel):
    """Synthetic generator boundary; real act/sanitizer/converter remain in use."""
    def __init__(self, args, tokenizer, query, outputs):
        self.args, self.tokenizer, self.query = args, tokenizer, query
        self.action_protocol, self._model_turn_id = 'full', 0
        self.base_prompt = adapter.actor_prompt_for_protocol('full') + '\n' + query + '\n'
        self.outputs = deque(outputs)
        self.sent_prompts, self.sampled = [], []
        self.closed = False

    def _generate(self, prompt, **kwargs):
        self.sent_prompts.append(prompt)
        raw = 'Assess the public evidence.</think>' + self.outputs.popleft()
        ids = list(raw.encode())
        probs = [-0.1 - i / 10000 for i in range(len(ids))]
        self.sampled.append((raw, ids, probs))
        return adapter.GeneratedTurn(raw, ids, probs, 'stop')

    def close(self):
        self.closed = True


def fixture_args(**kwargs):
    return SimpleNamespace(partial_rollout=False, rollout_max_response_len=8192, rollout_temperature=0.2, rollout_top_p=1.0, seq_length=32768, reflect_max_steps=8, reflect_max_tokens_per_turn=512, **kwargs)


def test_real_actor_and_converter_only_mask_independently_sampled_reflection():
    tokenizer, args = ByteTokenizer(), fixture_args()
    actor = SampledFixtureActor(args, tokenizer, 'query', ['<search>evidence</search>', '<answer>early</answer>', '<reflect>The observed evidence is coherent and uncertainty is limited.</reflect>', '<answer>done</answer>'])
    tree = harness(actor, require_post_search_reflection=True).run('query')
    samples = adapter.build_tree_training_samples(args, Sample(index=1, prompt='query', label='done'), tree, tokenizer)
    assert tree.status == 'completed' and len(samples) == 4
    assert [bool(sum(s.reflect_mask)) for s in samples] == [False, False, True, False]
    assert samples[2].metadata['reflect_answer_leak'] is False
    for sample, (raw, ids, probs) in zip(samples, actor.sampled, strict=True):
        assert sample.response == raw and sample.tokens[-sample.response_length:] == ids
        assert sample.rollout_log_probs == probs and sample.loss_mask == [1] * len(ids)
        assert sample.response_length == len(ids)
        assert 'Your next model turn' not in sample.response
    assert any('Your next model turn' in sample.prompt for sample in samples)


def test_real_converter_gives_harness_termination_zero_reflection_mass():
    tokenizer, args = ByteTokenizer(), fixture_args()
    actor = SampledFixtureActor(args, tokenizer, 'query', ['<search>evidence</search>', '<answer>done</answer>'])
    tree = harness(actor, require_post_search_reflection=True, max_steps=2).run('query')
    samples = adapter.build_tree_training_samples(args, Sample(index=2, prompt='query', label='done'), tree, tokenizer)
    assert tree.status == 'terminated' and len(samples) == 2
    assert any(n.kind == 'reflect' and n.metadata.get('source') == 'harness' for n in tree.nodes.values())
    assert all(not any(s.reflect_mask) for s in samples)
    assert all(s.status == Sample.Status.TRUNCATED for s in samples)


@pytest.mark.parametrize('enabled,expected_turns', [(False, 2), (True, 4)])
def test_training_entrypoint_wires_flag_without_exposing_private_label(monkeypatch, enabled, expected_turns):
    tokenizer = ByteTokenizer()
    args = fixture_args(reflect_require_post_search_reflection=enabled)
    module = ModuleType('slime.rollout.sglang_rollout')
    module.GenerateState = lambda args: SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setitem(sys.modules, 'slime.rollout.sglang_rollout', module)
    actions = ['<search>evidence</search>', '<answer>PRIVATE_LABEL_NEVER_IN_ACTOR</answer>', '<reflect>I assessed only the public source and uncertainty.</reflect>', '<answer>PRIVATE_LABEL_NEVER_IN_ACTOR</answer>']
    model = SampledFixtureActor(args, tokenizer, 'query', actions)
    search = Search()
    monkeypatch.setattr(adapter, 'SGLangHarnessModel', lambda *a: model)
    monkeypatch.setattr(adapter, 'HttpSearch', lambda *a, **kw: search)
    monkeypatch.setattr(adapter, '_save_debug_tree_samples', lambda *a: None)
    samples = asyncio.run(adapter.generate(args, Sample(index=3, prompt='query', label='PRIVATE_LABEL_NEVER_IN_ACTOR'), {}))
    assert len(samples) == expected_turns and model.closed and search.closed
    assert sum(bool(sum(s.reflect_mask)) for s in samples) == int(enabled)
    assert all('PRIVATE_LABEL_NEVER_IN_ACTOR' not in prompt for prompt in model.sent_prompts)


def launcher_args(tmp_path, config, flags=()):
    model = tmp_path / 'model.sh'
    model.write_text('MODEL_ARGS=( --num-layers 1 )\n')
    settings = tmp_path / 'stage.yaml'
    settings.write_text(yaml.safe_dump(config))
    return parse_args(['--data', str(tmp_path / 'train.jsonl'), '--hf-checkpoint', str(tmp_path / 'hf'), '--megatron-checkpoint', str(tmp_path / 'actor'), '--reference-checkpoint', str(tmp_path / 'reference'), '--save-dir', str(tmp_path / 'saves'), '--model-args', str(model), '--config', str(settings), *flags])


@pytest.mark.parametrize('protocol', ['full', 'search_answer'])
def test_launcher_default_preserves_original_yaml_dictionary(tmp_path, protocol):
    original = {'reflect_self_distill': True, 'reflect_train_mode': 'joint', 'reflect_action_protocol': protocol}
    _, config = build_command(launcher_args(tmp_path, original))
    assert config == original and 'reflect_require_post_search_reflection' not in config


def test_launcher_can_explicitly_enable_and_disable_stage(tmp_path):
    original = {'reflect_self_distill': True, 'reflect_train_mode': 'joint', 'reflect_action_protocol': 'full'}
    _, enabled = build_command(launcher_args(tmp_path, original, ['--require-post-search-reflection']))
    assert enabled['reflect_require_post_search_reflection'] is True
    _, disabled = build_command(launcher_args(tmp_path, enabled, ['--no-require-post-search-reflection']))
    assert disabled['reflect_require_post_search_reflection'] is False


@pytest.mark.parametrize('extra', [{'reflect_require_post_search_reflection': 'false'}, {'reflect_require_post_search_reflection': True, 'reflect_action_protocol': 'search_answer'}])
def test_launcher_rejects_ambiguous_or_incompatible_yaml_stage(tmp_path, extra):
    config = {'reflect_self_distill': True, **extra}
    with pytest.raises(ValueError, match='boolean|full action protocol'):
        build_command(launcher_args(tmp_path, config))
