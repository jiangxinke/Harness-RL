"""Synthetic CPU declarations, tensors and fake engine only; no raw/label files."""
import ast
import asyncio
import copy
import enum
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest
import torch
from safetensors.torch import save_file

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / 'slime'), str(SOURCE / 'scripts')]
from slime_plugins.dcc import corrected_lora as check
from slime_plugins.dcc import native_thinking_eval as native
from slime_plugins.dcc.native_lora import validate_native_adapter
from slime_plugins.dcc.eval_sampling import EvaluationSampling
from slime_plugins.dcc.native_budget import NativeBudget
from slime_plugins.dcc.vllm_policy import AsyncVLLMPolicy, _adapter_metadata
from slime_plugins.dcc import EpisodeRunner

spec = importlib.util.spec_from_file_location('synthetic_original_native_fixtures', SOURCE / 'tests/test_dcc_native_thinking.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
PARENT = dict(fixture.RECEIPT)
EVAL = dict(PARENT, source_commit='a' * 40)
BUDGET = NativeBudget(4096, 'thinking')
MODEL = dict(hidden_size=4, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=2)
DERIVATIVE = 'synthetic/base-125e-cpu-only'


def approval_fixture():
    identity = dict(source_commit=PARENT['source_commit'], derivative_id=DERIVATIVE,
        model_revision=native.MODEL_REVISION, protocol=BUDGET.protocol,
        prompt_encoding_version=native.PROMPT_ENCODING_VERSION)
    manifest = dict(protocol=check.PROTOCOL, target_contract='synthetic-TRAIN-gold-JSON-with-real-masked-PH',
        native_only_training=False, training_method='supervised fine-tuning; not RL',
        mask_version=check.MASK_VERSION, original_journals_closed_before_labels=True,
        original_native_selector_modified=False, parent_source_receipt=copy.deepcopy(PARENT),
        parent_profile={'identity': identity}, rows=3, positive=2, negative=1,
        confidence_values_supervised=False, held_out_reads=False, launchable=True,
        synthetic_cpu_fixture_only=True)
    return dict(training_target_contract=check.PROTOCOL,
        training_source_receipt=dict(version='dcc-grounded-final-training-source-v1',
            source_commit='e' * 40, frozen=True),
        parent_collection_source_receipt=copy.deepcopy(PARENT),
        training_target_lineage=check.target_lineage(),
        corrected_final_binding=dict(version='dcc-grounded-final-adapter-binding-v1',
            derivative_id=DERIVATIVE, model_revision=native.MODEL_REVISION,
            generation_protocol=BUDGET.protocol, validated_synthetic_final_records=3,
            mask_version=check.MASK_VERSION, derived_manifest=manifest, native_only_training=False))


def adapter_fixture(tmp_path, *, step=8):
    path = tmp_path / 'synthetic-adapter'
    path.mkdir()
    adapter = dict(peft_type='LORA', r=16, lora_alpha=32, lora_dropout=0,
        target_modules=['q_proj', 'v_proj', 'o_proj'])
    (path / 'adapter_config.json').write_text(json.dumps(adapter))
    tensors = {}
    for target, outputs in [('q_proj', 4), ('v_proj', 2), ('o_proj', 4)]:
        tensors[f'base_model.model.model.layers.0.self_attn.{target}.lora_A.weight'] = torch.zeros(16, 4)
        tensors[f'base_model.model.model.layers.0.self_attn.{target}.lora_B.weight'] = torch.zeros(outputs, 16)
    save_file(tensors, str(path / 'adapter_model.safetensors'))
    approval = approval_fixture()
    config = dict(approval, steps=32, lora_rank=16, lora_alpha=32, lora_dropout=0,
        target_modules=['q_proj', 'v_proj', 'o_proj'], warm_start=False,
        training_method='annotation-supervised grounded CUAD gold-final SFT; not RL',
        native_thinking=dict(enabled=False, protocol=check.PROTOCOL, origin=check.ORIGIN,
            requires_new_explicit_evaluation_binding=True),
        loss_configuration=dict(version=check.MASK_VERSION, example_weighting='global-equal-example',
            token_weighting='per-example-mean-over-unmasked-JSON-and-EOS-tokens'))
    (path / 'training_config.json').write_text(json.dumps(config))
    (path / 'checkpoint_complete.json').write_text(json.dumps(dict(complete=True, optimizer_steps=step)))
    return path, config, approval


def validate(path, approval, **extra):
    return validate_native_adapter(path, source_receipt=EVAL, approved_derivative_id=DERIVATIVE,
        model_config=MODEL, evaluation_protocol=BUDGET.protocol,
        approved_corrected=approval, **extra)


def test_separate_three_sources_and_truthful_non_native_binding(tmp_path):
    path, _, approval = adapter_fixture(tmp_path)
    metadata, binding = validate(path, approval)
    assert metadata['config']['r'] == 16 and binding['tensor_pairs'] == 3
    assert binding['training_source_receipt']['source_commit'] == 'e' * 40
    assert binding['parent_collection_source_receipt'] == PARENT
    assert binding['evaluation_source_receipt'] == EVAL
    assert binding['native_only_training'] is False and binding['metadata_binding_only'] is True
    assert binding['checkpoint_optimizer_steps'] == 8
    check.validate_corrected_sidecar_binding(binding, approval, derivative_id=DERIVATIVE,
        source_receipt=EVAL, evaluation_protocol=BUDGET.protocol)
    approval['training_source_receipt']['source_commit'] = 'c' * 40
    assert binding['training_source_receipt']['source_commit'] == 'e' * 40


def test_old_native_branch_still_rejects_corrected_and_no_implicit_approval(tmp_path):
    path, _, _ = adapter_fixture(tmp_path)
    with pytest.raises(ValueError, match='Actual adapter TRAIN configuration differs'):
        validate_native_adapter(path, source_receipt=EVAL, approved_derivative_id=DERIVATIVE,
            model_config=MODEL, evaluation_protocol=BUDGET.protocol)
    with pytest.raises(ValueError, match='independent complete five-field'):
        validate(path, {})


@pytest.mark.parametrize('field', sorted(check.APPROVAL_FIELDS))
def test_every_saved_field_matches_independent_declaration(tmp_path, field):
    path, config, approval = adapter_fixture(tmp_path)
    config[field] = 'changed'
    (path / 'training_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='differs from independent'):
        validate(path, approval)


@pytest.mark.parametrize('case', ['thought_loss', 'native_claim', 'held_out', 'count_bool', 'base', 'old_source'])
def test_even_matching_self_claims_cannot_relabel_corrected_targets(tmp_path, case):
    path, config, approval = adapter_fixture(tmp_path)
    if case == 'thought_loss': approval['training_target_lineage']['loss_on_thought'] = True
    if case == 'native_claim': approval['corrected_final_binding']['native_only_training'] = True
    if case == 'held_out': approval['corrected_final_binding']['derived_manifest']['held_out_reads'] = True
    if case == 'count_bool': approval['corrected_final_binding']['validated_synthetic_final_records'] = True
    if case == 'base': approval['corrected_final_binding']['derived_manifest']['parent_profile']['identity']['derivative_id'] = 'wrong'
    if case == 'old_source': approval['training_source_receipt']['source_commit'] = PARENT['source_commit']
    config.update(approval)
    (path / 'training_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError): validate(path, approval)


@pytest.mark.parametrize('extra', [{'approved_dpo': {}}, {'approved_training_protocol': BUDGET.protocol}, {'approved_training_source_receipt': PARENT}])
def test_training_modes_mutually_exclusive(tmp_path, extra):
    path, _, approval = adapter_fixture(tmp_path)
    with pytest.raises(ValueError, match='mutually exclusive'): validate(path, approval, **extra)


@pytest.mark.parametrize('step', [False, -1, 33])
def test_complete_checkpoint_requires_real_typed_in_range_steps(tmp_path, step):
    path, _, approval = adapter_fixture(tmp_path, step=step)
    with pytest.raises(ValueError, match='incomplete or has invalid'): validate(path, approval)


def test_initial_checkpoint_is_explicit_step_zero_not_trained_winner(tmp_path):
    path, _, approval = adapter_fixture(tmp_path, step=0)
    assert validate(path, approval)[1]['checkpoint_optimizer_steps'] == 0


def test_actual_shape_completeness_guards_and_duplicate_json(tmp_path):
    path, _, approval = adapter_fixture(tmp_path)
    save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': torch.zeros(16, 5)}, str(path / 'adapter_model.safetensors'))
    with pytest.raises(ValueError, match='Adapter shape differs'): validate(path, approval)
    (path / 'training_config.json').write_text('{"steps":32,"steps":32}')
    with pytest.raises(ValueError, match='Duplicate JSON'): validate(path, approval)


def test_real_new_trainer_binding_producer_interoperates_without_loading_data():
    root = SOURCE / 'scripts'
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('synthetic_grounded_final_producer', root / 'dcc_grounded_final_sft.py')
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    expected = approval_fixture()
    actual = producer.training_binding(copy.deepcopy(expected['corrected_final_binding']['derived_manifest']),
        training_source_commit='e' * 40, derivative_id=DERIVATIVE)
    assert check.canonical(check.validate_corrected_approval(actual)) == check.canonical(expected)


@pytest.fixture
def fake_backend(monkeypatch):
    class Kind(enum.IntEnum): FINAL_ONLY = 2
    class Params:
        def __init__(self, **values):
            self.structured_outputs = None
            self.__dict__.update(values)
            if values.get('temperature') == 0:
                self.top_p, self.top_k, self.min_p = 1.0, 0, 0.0
    package, sampling = ModuleType('vllm'), ModuleType('vllm.sampling_params')
    package.SamplingParams = Params
    sampling.RequestOutputKind = Kind
    monkeypatch.setitem(sys.modules, 'vllm', package)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', sampling)
    monkeypatch.setattr(native, 'cuad_structured_options', lambda schema: {'structured_outputs': {'synthetic_original_schema': schema}})
    monkeypatch.setattr(native, 'preflight_cuad_structured', lambda: None)


class FakeEngine:
    def __init__(self, tokenizer):
        self.vllm_config = SimpleNamespace(structured_outputs_config=SimpleNamespace(backend='xgrammar',
            disable_fallback=True, reasoning_parser=None, reasoning_parser_plugin=None, enable_in_reasoning=False),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(to_dict=lambda: dict(MODEL))))
        self.scripts = [tokenizer.encode('Synthetic natural thought.</think>'), tokenizer.encode('{"answer":[]}') + [tokenizer.eos_token_id]]
        self.calls = []
    async def generate(self, prompt, params, request_id, *, lora_request):
        self.calls.append((copy.deepcopy(prompt), copy.deepcopy(params.__dict__)))
        ids = self.scripts.pop(0)
        yield SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'], finished=True,
            outputs=[SimpleNamespace(token_ids=ids, text='synthetic', finish_reason='stop', stop_reason=ids[-1])])
    def shutdown(self): pass


def run_fake(tmp_path, *, adapter=None, approval=None):
    tokenizer = fixture.FakeTokenizer()
    engine = FakeEngine(tokenizer)
    metadata = _adapter_metadata(adapter) if adapter else None
    base = AsyncVLLMPolicy(engine, tokenizer, max_prompt_tokens=16384,
        max_model_len=16384+BUDGET.total_tokens, eos_token_ids=[151643,151645],
        evaluation_sampling=EvaluationSampling(), web_structured_final=True, web_structured_actions=True,
        lora_request=SimpleNamespace(lora_path=str(adapter)) if adapter else None,
        provenance={'adapter': metadata} if adapter else {})
    journal = native.NativeTraceJournal(tmp_path / 'native_thinking_traces_rank0.jsonl')
    policy = native.NativeThinkingPolicy(base, source_receipt=EVAL, trace_sink=journal,
        budget=BUDGET, approved_derivative_id=DERIVATIVE if adapter else None,
        approved_corrected=approval)
    async def execute():
        bound = policy.bind_task('contract', 0)
        runner = EpisodeRunner(max_steps=2, max_new_tokens=BUDGET.total_tokens,
            initial_chunks=1, cuad_structured_actions=True)
        result = await runner.run(fixture.cuad_task(), bound, evaluation=True)
        record = result.to_dict()
        record['native_thinking_traces'] = bound.references
        config = native.native_configuration(EVAL, budget=BUDGET)
        manifest = dict(native_thinking=config, max_new_tokens=BUDGET.total_tokens,
            max_steps=2, world_size=1, seed=42, evaluation_sampling=EvaluationSampling().to_dict(),
            max_prompt_tokens=16384, inference_parameters={'async_engine':True,'max_model_len':16384+BUDGET.total_tokens})
        runtime = dict(policy.provenance, eos_token_ids=[151643,151645],
            max_prompt_tokens=16384, max_model_len=16384+BUDGET.total_tokens)
        if adapter:
            manifest.update(adapter=str(adapter), adapter_identity=metadata,
                native_lora_derivative_id=DERIVATIVE, corrected_training_approval=approval)
        await policy.close()
        audit = native.validate_native_sidecars(tmp_path, {'contract':record}, manifest,
            runtime, {'contract':0}, [fixture.cuad_task()], tokenizer=tokenizer)
        return engine, policy, record, manifest, runtime, audit
    return asyncio.run(execute())


def test_fake_matched_native_generation_preserves_exact_P_H_schema_budget_and_sidecars(tmp_path, fake_backend):
    adapter, _, approval = adapter_fixture(tmp_path)
    base_dir, trained_dir = tmp_path/'base', tmp_path/'trained'
    base_dir.mkdir(); trained_dir.mkdir()
    base = run_fake(base_dir)
    trained = run_fake(trained_dir, adapter=adapter, approval=approval)
    assert base[2]['status'] == trained[2]['status'] == 'completed'
    assert base[0].calls == trained[0].calls
    assert len(trained[0].calls) == 2
    phase1, phase2 = trained[1].traces[0]['phases']
    assert phase2['prompt_ids'] == phase1['prompt_ids'] + phase1['token_ids']
    assert 'native_lora_binding' not in trained[4]
    assert trained[4]['corrected_lora_binding'] == trained[1].traces[0]['corrected_lora_binding']
    corrupted = copy.deepcopy(trained[4]); corrupted['native_lora_binding'] = {}
    with pytest.raises(ValueError, match='separate adapter'):
        native.validate_native_sidecars(trained_dir, {'contract':trained[2]}, trained[3],
            corrupted, {'contract':0}, [fixture.cuad_task()], tokenizer=fixture.FakeTokenizer())


def test_cli_preflight_routes_explicit_corrected_adapter_and_rejects_base_or_default(tmp_path, monkeypatch, fake_backend):
    adapter, _, approval = adapter_fixture(tmp_path)
    receipt = tmp_path / '.dcc-native-source.json'; receipt.write_text(json.dumps(EVAL))
    approval_path = tmp_path / 'independent-approval.json'; approval_path.write_text(json.dumps(approval))
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace(to_dict=lambda:dict(MODEL)))
    args = SimpleNamespace(native_thinking=True, native_source_receipt=str(receipt),
        native_lora_derivative_id=DERIVATIVE, native_thought_tokens=4096, native_thought_sampling='thinking',
        corrected_lora_approval=str(approval_path), adapter=str(adapter), model='synthetic-model-not-loaded',
        inference_backend='vllm', vllm_async=True, web_structured_final=True, web_structured_actions=True,
        max_new_tokens=BUDGET.total_tokens)
    assert native.native_options(args, [fixture.cuad_task()], EvaluationSampling(), tmp_path)['protocol'] == BUDGET.protocol
    args.adapter = None
    with pytest.raises(ValueError, match='requires an actual adapter'): native.native_options(args, [fixture.cuad_task()], EvaluationSampling(), tmp_path)
    args.native_thinking = False
    with pytest.raises(ValueError): native.native_options(args, [], EvaluationSampling(), tmp_path)


def test_matched_comparator_rechecks_actual_corrected_checkpoint_and_rejects_stale_step(tmp_path, monkeypatch, fake_backend):
    import dcc_compare_runs as compare
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: fixture.FakeTokenizer())
    adapter, _, approval = adapter_fixture(tmp_path)
    output = tmp_path/'trained'; output.mkdir()
    _, _, record, manifest, runtime, _ = run_fake(output, adapter=adapter, approval=approval)
    model = tmp_path/'synthetic-model'; model.mkdir()
    (model/'config.json').write_text(json.dumps(MODEL))
    manifest.update(model=str(model), inference_backend='vllm', quantization='none')
    report = dict(generation_manifest=manifest, inference_runtime=runtime, tasks=1,
        prediction_coverage=1, status_counts={'completed':1}, policy_quantization='none',
        policy_inference_backend='vllm', complete=True)
    (output/'scores_strict.json').write_text(json.dumps(report))
    run = dict(output=output, manifest=manifest, runtime=runtime, predictions={'contract':record},
        status_counts={'completed':1}, _native_context=({'contract':0},[fixture.cuad_task()]))
    generation = {'native_thinking':manifest['native_thinking']}
    assert compare._validate_native_pair_run(run, generation, 'lora') is True
    # Sidecar declarations alone cannot replace the actual checkpoint check.
    runtime['corrected_lora_binding']['checkpoint_optimizer_steps'] = 7
    trace_path = output/'native_thinking_traces_rank0.jsonl'
    trace = json.loads(trace_path.read_text())
    trace['corrected_lora_binding']['checkpoint_optimizer_steps'] = 7
    trace_path.write_text(json.dumps(trace)+'\n')
    report['inference_runtime'] = runtime
    (output/'scores_strict.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='actual local adapter metadata/binding'):
        compare._validate_native_pair_run(run, generation, 'lora')


def test_no_native_generation_cannot_claim_corrected_sidecars(tmp_path):
    with pytest.raises(ValueError, match='requires the original native'):
        native.validate_native_sidecars(tmp_path, {}, {'corrected_training_approval':approval_fixture()}, {}, {}, [])
