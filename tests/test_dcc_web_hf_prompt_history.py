"""CPU-only native Web history checks with invented requests/model outputs."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("xgrammar")
from transformers import GenerationConfig

from slime_plugins.dcc.cuad_structured import cuad_final_schema
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.hf_native_policy import NativeHFPolicy, END_THINK
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.prompt_history import PromptHistoryConfig, PromptHistoryBudgetExceeded
from slime_plugins.dcc.web_hf_evaluation import OwnedEvaluationPolicy, verify_native_ownership
from slime_plugins.dcc.web_hf_prompt_history import FIELDS, native_web_history_plan
from slime_plugins.dcc.web_structured import web_final_schema
from test_dcc_hf_native_policy import ScriptedModel, Tokenizer, final_ids, tokenizer
from test_dcc_online_web_evaluate import evaluator, arguments, generate, references, make_adapter

URL = "https://example.org/invented-history-source"
CONFIG = PromptHistoryConfig(True, 2)


def history_messages(count=4, repetitions=20):
    messages = [{"role": "system", "content": "Use literal evidence and complete JSON."},
                {"role": "user", "content": "Find the invented public name."}]
    for index in range(count):
        url = URL + f"/{index}"
        messages += [{"role": "assistant", "content": json.dumps({"tool": "fetch_url", "arguments": {"url": url}})},
                     {"role": "user", "content": "Tool observation (source material):\n" + json.dumps({
                         "url": url, "text": 'literal {完整JSON} "quote" \\ ' * repetitions}, ensure_ascii=False)
                         + "\nRemaining turns: 2."}]
    messages.append({"role": "user", "content": "Final turn: return complete JSON with citations."})
    return messages


def budgets(tokenizer, messages):
    return dict(max_prompt_tokens=len(_chat_tokens(tokenizer, messages[:2] + messages[-5:])) + 420,
                max_model_len=10000, thought_tokens=4, json_tokens=256)


def native_policy(tokenizer, *, config=CONFIG, thoughts=None, sink=None, **options):
    limits = budgets(tokenizer, history_messages())
    text = json.dumps({"answer": [{"name": "Alpha", "citations": [URL]}]})
    model = ScriptedModel(thoughts or [[END_THINK]], [final_ids(tokenizer, text=text)])
    model.config = SimpleNamespace(vocab_size=151670, max_position_embeddings=10000)
    value = NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=256,
        max_prompt_tokens=limits['max_prompt_tokens'], trace_sink=sink, prompt_history_config=config, **options)
    return value, limits


def normalized(value):
    value = deepcopy(value)
    if isinstance(value, dict):
        value.pop('trace_id', None); value.pop('batch_phase_elapsed_seconds', None)
        return {k: normalized(v) for k,v in value.items()}
    if isinstance(value, list):return [normalized(v) for v in value]
    return value


@pytest.fixture(scope="module")
def tiny_fast_tokenizer(tmp_path_factory):
    """A real local HF/tokenizers checkpoint with Qwen native special IDs."""
    from tokenizers import Tokenizer as Backend, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    vocab = {f"unused-{i}": i for i in range(151670)}
    for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        del vocab[f"unused-{i}"]; vocab[token] = i
    for token, i in Tokenizer.tags.items():
        del vocab[f"unused-{i}"]; vocab[token] = i
    backend = Backend(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<|im_end|>',
        pad_token='<|endoftext|>', additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    result.chat_template = ("{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{% endfor %}"
                            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{% endif %}")
    path = tmp_path_factory.mktemp('native-tokenizer'); result.save_pretrained(path)
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def test_real_local_tiny_tokenizer_retains_native_open_think_and_full_json(tiny_fast_tokenizer):
    t = tiny_fast_tokenizer; messages = history_messages(); original = deepcopy(messages)
    plan = native_web_history_plan(t, messages, config=CONFIG, **budgets(t, messages))
    assert plan.report['compacted'] and messages == original
    assert plan.messages[:2] == messages[:2] and plan.messages[-5:] == messages[-5:]
    assert plan.prompt_ids == _chat_tokens(t, plan.messages)
    suffix = NativeHFPolicy.validate_tokenizer(t)['native_suffix_ids']
    assert plan.prompt_ids[-len(suffix):] == suffix
    assert t.decode(suffix, skip_special_tokens=False) == '<|im_start|>assistant\n<think>\n'
    assert END_THINK not in suffix and suffix.count(151667) == 1
    assert plan.report['omitted_message_indices'] and plan.report['omitted_observed_urls']
    previous = t.chat_template
    t.chat_template = previous.replace("<think>\\n", "<think>\\n</think>\\n")
    try:
        with pytest.raises(ValueError, match='native open-thinking suffix'):
            native_web_history_plan(t, messages, config=CONFIG, **budgets(t, messages))
    finally:t.chat_template = previous
    assert not torch.cuda.is_initialized()


def test_default_native_serialization_and_decoder_configs_are_identical(tokenizer):
    outputs=[]
    for config in (None, PromptHistoryConfig()):
        traces=[]; value, _ = native_policy(tokenizer, config=config, sink=traces.append)
        completion = value._generate_batch([history_messages(count=1)], web_final_schema(['name'], [URL]), ['cpu-default'])[0]
        assert not isinstance(completion, BaseException)
        assert not any(k in value.provenance for k in FIELDS)
        assert all(not any(k in row for k in FIELDS) for row in traces)
        outputs.append((normalized(asdict(completion)), normalized(traces), value.provenance,
                        [call['generation_config'].to_dict() for call in value.model.calls]))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('options', [dict(capture_thought_logprobs=True), dict(joint_json_policy=True), dict(allow_truncation=True)])
def test_native_history_rejects_unsupported_training_and_truncation(tokenizer, options):
    with pytest.raises(ValueError, match='Web evaluation-only'):native_policy(tokenizer, **options)


def test_native_history_rejects_cuad_before_any_model_submission(tokenizer):
    value, _ = native_policy(tokenizer)
    with pytest.raises(ValueError, match='Web evaluation schemas only'):
        value._generate_batch([history_messages()], cuad_final_schema())
    assert not value.model.calls
    with pytest.raises(ValueError, match='PromptHistoryConfig'):native_policy(tokenizer, config=True)


def test_mixed_batch_keeps_each_successful_plan_aligned_after_failed_preflight(tokenizer):
    good_a, bad, good_b = history_messages(), history_messages(), history_messages()
    good_b[1]['content'] += ' Distinct task B.'
    bad[-2]['content'] = 'Tool observation (source material):\n' + json.dumps({'url':URL, 'text':'X'*15000})
    value, limits = native_policy(tokenizer, mixed_schema_batching=True)
    value.model.thoughts = [[END_THINK], [END_THINK]]; value.model.finals *= 2
    traces=[]; value.trace_sink=traces.append; schema=web_final_schema(['name'], [URL])
    rows=value._generate_batch([good_a,bad,good_b], [schema,schema,schema], ['A','bad','B'])
    assert isinstance(rows[1], PromptHistoryBudgetExceeded)
    assert [tr['task_id'] for tr in traces] == ['A','B']
    for request, completion, trace in zip([good_a,good_b], [rows[0],rows[2]], traces, strict=True):
        plan=native_web_history_plan(tokenizer,request,config=CONFIG,**limits)
        assert completion.prompt_ids == trace['prompt_ids'] == plan.prompt_ids
        assert trace['prompt_history_original_messages'] == request and trace['prompt_history'] == plan.report
        assert trace['thought_ids'] == [END_THINK] and trace['json_ids'][-1] == tokenizer.eos_token_id
    assert len(value.model.calls) == 2


def owned_records(tokenizer):
    messages=history_messages(); traces=[]; value, limits=native_policy(tokenizer,sink=traces.append)
    schema=web_final_schema(['name'],[URL]); completion=value._generate_batch([messages],schema,['cpu-owned'])[0]
    stored=json.loads(json.dumps(asdict(completion)))
    owner=dict(task_id='cpu-owned',shard_index=0,turn=0,messages=messages,json_schema=schema,
               request_status='native',delivery='delivered',trace_id=completion.trace_id,completion=stored)
    result=dict(task_id='cpu-owned',benchmark='widesearch',split='validation',status='completed',
                rollout_labels_removed=True,rollout_scored=False,steps=[stored],metrics={},answer=json.loads(completion.text)['answer'])
    task=Task('cpu-owned','widesearch','validation','Find the invented public name.',metadata={'evaluation':{'required':['name']}})
    return [task],[result],[owner],json.loads(json.dumps(traces)),limits


def verify(tokenizer, records, *, enabled=True):
    tasks,predictions,owners,traces,limits=records
    return verify_native_ownership(tasks,predictions,owners,traces,rank=0,profile='raw',eos_ids=[151645,151643],tokenizer=tokenizer,
        **(dict(prompt_history_config=CONFIG,history_budgets=limits) if enabled else {}))


@pytest.mark.parametrize('damage', [None,'original','omitted_indices','omitted_urls','report_type',
                                  'compact_ids','declaration','behavior_capture','thought_close','json_eos','flag_off'])
def test_ownership_reconstructs_plan_and_rejects_tampering(tokenizer, damage):
    records=owned_records(tokenizer); _,results,owners,traces,_=records; tr=traces[0]
    if damage=='original':tr['prompt_history_original_messages'][1]['content']+=' forged'
    if damage=='omitted_indices':tr['prompt_history']['omitted_message_indices']=[]
    if damage=='omitted_urls':tr['prompt_history']['omitted_observed_urls']=[]
    if damage=='report_type':tr['prompt_history']['compacted']=1
    if damage=='compact_ids':tr['prompt_ids'][10]+=1;owners[0]['completion']['prompt_ids'][10]+=1
    if damage=='declaration':tr['native_web_prompt_history']['version']='forged'
    if damage=='behavior_capture':tr['thought_logprobs']=[-.1]
    if damage in {'thought_close','json_eos'}:
        i=0 if damage=='thought_close' else -1
        owners[0]['completion']['completion_ids'][i]=10065
        tr['thought_ids' if damage=='thought_close' else 'json_ids'][i]=10065
    if damage is None:assert verify(tokenizer,records)['all_native_traces_owned'] is True
    else:
        with pytest.raises(ValueError):verify(tokenizer,records,enabled=damage!='flag_off')


@pytest.mark.parametrize('oversized', ['latest','task'])
@pytest.mark.parametrize('cancelled', [False,True])
def test_oversized_context_persists_error_owner_and_remains_failure(tokenizer, oversized, cancelled):
    messages=history_messages();messages[1 if oversized=='task' else -2]['content']+=' X'*16000
    value,limits=native_policy(tokenizer);owners=[];schema=web_final_schema(['name'],[URL])
    async def run():
        parent=value.bind_task('cpu-error',0)
        class Bound:
            async def generate_structured(self,*args,**kwargs):
                if cancelled:await asyncio.sleep(.01)
                return await parent.generate_structured(*args,**kwargs)
        owner=OwnedEvaluationPolicy(Bound(),task_id='cpu-error',rank=0,sink=owners.append)
        try:
            pending=asyncio.create_task(owner.generate_structured(messages,json_schema=schema,max_new_tokens=260,temperature=1.))
            if cancelled:
                await asyncio.sleep(.001);pending.cancel()
            with pytest.raises(asyncio.CancelledError if cancelled else PromptHistoryBudgetExceeded):await pending
        finally:await value.close()
    asyncio.run(run())
    assert not value.model.calls
    record=owners[0]
    assert record['prompt_history_original_messages']==messages
    assert record['original_prompt_ids']==_chat_tokens(tokenizer,messages)
    assert record['prompt_history']['actual_prompt_tokens'] is None
    task=Task('cpu-error','widesearch','validation','invented',metadata={'evaluation':{'required':['name']}})
    result=dict(task_id=task.id,benchmark='widesearch',split='validation',
        status='infrastructure_error' if cancelled else 'context_budget_failure',rollout_labels_removed=True,
        rollout_scored=False,steps=[],metrics={})
    if not cancelled:
        result['metrics']['prompt_history_failure']=dict(reason='minimum_complete_context_exceeds_budget',
            diagnostics=record['prompt_history'],request_messages=messages,generation_submitted=False)
    records=([task],[result],owners,[],limits)
    assert verify(tokenizer,records)['native_traces']==0
    record['original_prompt_ids'][-1]+=1
    with pytest.raises(ValueError,match='failure provenance'):verify(tokenizer,records)


@pytest.fixture
def native_evaluator(evaluator,tokenizer,monkeypatch):
    # Invented Web tools/reference rows from the shared evaluator fixture;
    # the real NativeHFPolicy is exercised over a scripted CPU model seam.
    monkeypatch.setattr(type(tokenizer),'__len__',lambda self:151670,raising=False)
    tokenizer.chat_template='invented native CPU char template'
    original_load=evaluator.load_policy_model
    def load(*args,**kwargs):
        model,_=original_load(*args,**kwargs)
        model.config=SimpleNamespace(vocab_size=151670,max_position_embeddings=65536,
            to_dict=lambda:{'model_type':'invented-CPU','max_position_embeddings':65536})
        model.device=torch.device('cpu')
        model.generation_config=GenerationConfig(eos_token_id=[151645,151643])
        actions=iter([{'plan':'Read','checks':'Use source','tool':'web_search','arguments':{'query':'invented'}},
                      {'plan':'Answer','checks':'Read evidence','answer':[{'name':'Alpha','country':'AA',
                          'citations':['https://example.com/invented-cpu-source']}]}])
        def generation(**options):
            cfg=options.get('generation_config',model.generation_config)
            thought=END_THINK in options.get('eos_token_id',cfg.eos_token_id)
            ids=[END_THINK] if thought else tokenizer.encode(json.dumps(next(actions)))+[tokenizer.eos_token_id]
            return torch.cat([options['input_ids'],torch.tensor([ids])],dim=1)
        model.generate=generation
        return model,tokenizer
    monkeypatch.setattr(evaluator,'NativeHFPolicy',NativeHFPolicy)
    monkeypatch.setattr(evaluator,'load_policy_model',load)
    from transformers import AutoConfig,AutoTokenizer
    def config_loader(*args,**kwargs):
        assert kwargs==dict(local_files_only=True,trust_remote_code=False)
        return SimpleNamespace(max_position_embeddings=65536)
    def tokenizer_loader(*args,**kwargs):
        assert kwargs==dict(local_files_only=True,trust_remote_code=False)
        return tokenizer
    monkeypatch.setattr(AutoConfig,'from_pretrained',config_loader)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',tokenizer_loader)
    return evaluator


def test_cli_default_manifest_matches_legacy_namespace(evaluator,tmp_path):
    args=arguments(evaluator,tmp_path/'default')
    assert args.prompt_history_whole_turns is False
    before=evaluator.manifest(args,references(('success',)))
    del args.prompt_history_whole_turns;del args.prompt_history_keep_recent_turns
    assert before==evaluator.manifest(args,references(('success',)))
    assert 'native_web_prompt_history' not in before['comparison']['harness']
    args.prompt_history_keep_recent_turns=2
    with pytest.raises(ValueError,match='requires --prompt-history-whole-turns'):
        evaluator.manifest(args,references(('success',)))


@pytest.mark.parametrize('base_enabled,adapter_enabled',[(False,False),(True,True),(False,True),(True,False)])
def test_history_must_match_base_and_adapter_before_any_score(native_evaluator,tmp_path,base_enabled,adapter_enabled):
    ev=native_evaluator;tasks=references(('success',))
    base=arguments(ev,tmp_path/'base',prompt_history_whole_turns=base_enabled)
    public=generate(ev,base,tasks,tmp_path);ev.score(base,tasks)
    adapter=arguments(ev,tmp_path/'adapter-run',prompt_history_whole_turns=adapter_enabled,
        adapter=make_adapter(tmp_path/'adapter'),base_output_dir=base.output_dir)
    asyncio.run(ev.generate(adapter,public));calls=len(ev.metric_calls)
    if base_enabled==adapter_enabled:
        assert ev.score(adapter,tasks)['matched_base']['f1_by_row_delta']==0
    else:
        with pytest.raises(ValueError,match='configuration, precision, runtime or closure differs'):ev.score(adapter,tasks)
        assert len(ev.metric_calls)==calls
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('enabled',[False,True])
def test_score_only_rejects_changed_history_flag_before_scoring(native_evaluator,tmp_path,enabled):
    ev=native_evaluator;tasks=references(('success',))
    args=arguments(ev,tmp_path/'run',prompt_history_whole_turns=enabled)
    generate(ev,args,tasks,tmp_path);args.prompt_history_whole_turns=not enabled
    with pytest.raises(ValueError,match='configuration, precision, runtime or closure differs'):ev.score(args,tasks)
    assert not ev.metric_calls


def test_score_only_rejects_changed_keep_recent_and_replay_tokenizer(native_evaluator,tmp_path,monkeypatch):
    ev=native_evaluator;tasks=references(('success',))
    args=arguments(ev,tmp_path/'run',prompt_history_whole_turns=True)
    generate(ev,args,tasks,tmp_path);args.prompt_history_keep_recent_turns=2
    with pytest.raises(ValueError,match='configuration, precision, runtime or closure differs'):ev.score(args,tasks)
    args.prompt_history_keep_recent_turns=1
    from transformers import AutoTokenizer
    def wrong(*a,**k):
        t=Tokenizer();t.chat_template='forged';return t
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',wrong)
    with pytest.raises(ValueError,match='replay tokenizer differs'):ev.score(args,tasks)
    assert not ev.metric_calls


@pytest.mark.parametrize('target',['runtime','configuration'])
def test_score_only_rejects_boolean_integer_history_declaration(native_evaluator,tmp_path,target):
    ev=native_evaluator;tasks=references(('success',))
    args=arguments(ev,tmp_path/'run',prompt_history_whole_turns=True)
    generate(ev,args,tasks,tmp_path)
    path=args.output_dir/('rank0.done.json' if target=='runtime' else 'config_rank0.json')
    record=json.loads(path.read_text())
    declaration=(record['runtime']['native_web_prompt_history']['configuration'] if target=='runtime'
                 else record['comparison']['harness']['native_web_prompt_history'])
    declaration['evaluation_only']=1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='runtime policy differs|configuration, precision, runtime or closure differs'):
        ev.score(args,tasks)
    assert not ev.metric_calls
