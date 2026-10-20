"""Synthetic CPU-only cached record parsing, provenance and runtime contracts."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import jsonschema
import pytest

from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc.episode import _WebEvidence
from slime_plugins.dcc.tools import WebTools
from slime_plugins.dcc.web_navigation import WebNavigationLedger, WEB_NAVIGATION_PREFIX
from slime_plugins.dcc.web_rows import (
    WEB_CACHED_ROWS_VERSION, WEB_CACHED_ROWS_LIMITS, WEB_CACHED_ROWS_INSTRUCTIONS,
    select_cached_rows, select_rows_schema, web_cached_rows_configuration,
    validate_web_cached_rows_records,
)
from slime_plugins.dcc.web_structured import (
    web_action_schema, web_structured_actions_configuration, web_structured_configuration,
)
ROOT = Path(__file__).parents[1]
HELPERS = runpy.run_path(str(ROOT/'tests/test_dcc_web_explicit_review.py'))
Policy, task = HELPERS['Policy'], HELPERS['task']
URL='https://fixture.example.org/records'
CONFIG=web_cached_rows_configuration(True)
CSV='id,name,value\r\n1,"Alpha, one",1.00\r\n2,"Beta\r\nline",2\r\n3,中文,3\r\n'


def req(**kwargs):
    return {'format':'csv','columns':['id','name'],**kwargs}


def select(source=CSV, request=None, **kwargs):
    return select_cached_rows(source, req() if request is None else request, **kwargs)


def web(source=CSV, budget=14000, enabled=True):
    instance=WebTools(web_cached_rows=enabled,max_page_chars=budget)
    instance.pages[URL]=source
    return instance


def fetch(instance, request=None, **kwargs):
    return asyncio.run(instance.fetch_url(URL, select_rows=req() if request is None else request, **kwargs))


def assert_spans(source, result):
    assert result['row_view']['kind']=='literal-record-windows'
    for row in result['row_view']['records']:
        assert source[row['start']:row['end']]==row['text']
        assert row['end']-row['start']==len(row['text'])
    if result['row_view']['records']:
        first=result['row_view']['records'][0]
        assert all(result[key]==first[key] for key in ['text','start','end'])


def test_csv_exact_cells_quoted_newlines_crlf_unicode_and_record_offsets():
    result=fetch(web())
    assert_spans(CSV,result)
    rows=result['row_view']['records']
    assert [r['cells']['name']['value'] for r in rows]==['Alpha, one','Beta\r\nline','中文']
    assert result['row_view']['header']['text']=='id,name,value\r\n'
    assert result['row_view']['header']['role'].endswith('not-a-data-record')
    assert result['row_view']['matching_records_after']==3
    assert result['row_view']['next_after'] is None
    assert len(json.dumps(result,ensure_ascii=False))<=14000


@pytest.mark.parametrize('op,value,expected',[
    ('eq','中文',['3']),('ne','中文',['1','2']),('contains','one',['1']),
    ('starts_with','Beta',['2']),('lex_ge','中文',['3']),('lex_le','B',['1']),
    ('exists',None,['1','2','3']),
])
def test_exact_string_predicates(op,value,expected):
    predicate={'column':'name','op':op}
    if op!='exists': predicate['value']=value
    result=select(request=req(where=[predicate]))
    assert [r['cells']['id']['value'] for r in result['records']]==expected


@pytest.mark.parametrize('op,value,expected', [('num_eq','1',['1']),('num_ge','2',['2','3']),('num_le','2',['1','2'])])
def test_explicit_numeric_operators_preserve_csv_spelling(op,value,expected):
    result=select(request=req(columns=['id','value'],where=[{'column':'value','op':op,'value':value}]))
    assert [r['cells']['id']['value'] for r in result['records']]==expected
    if op=='num_eq': assert result['records'][0]['cells']['value']['value']=='1.00'


def test_csv_explicit_header_row_bom_tsv_and_no_header_guessing():
    source='\ufefftitle\r\ncode\tvalue\r\nA\t"with\ttab"\r\n'
    result=select(source, {'format':'tsv','header_row':1,'columns':['code','value']})
    row=result['records'][0]
    assert row['text']==source[row['start']:row['end']]
    assert row['cells']['value']['value']=='with\ttab'
    with pytest.raises(ValueError): select(source, {'format':'tsv','columns':['code']})


@pytest.mark.parametrize('source', ['a,a\n1,2\n','a,\n1,2\n','a,b\n1\n','a\n1,2\n','a\n"unterminated', 'a\n1\n\n', ''])
def test_csv_rejects_ambiguous_or_malformed_entire_table(source):
    with pytest.raises(ValueError): select(source, req(columns=['a']))


def test_bad_record_after_first_match_rejects_instead_of_publishing_partial_table():
    result=fetch(web('a,b\n1,2\n3\n'), req(columns=['a'],limit=1))
    assert 'error' in result and 'text' not in result and 'row_view' not in result


@pytest.mark.parametrize('change',[
    {'columns':[]},{'columns':['id','id']},{'columns':[None]},{'format':[]},{'format':'xlsx'},
    {'where':None},{'where':[{'column':'id','op':[]}]},{'where':[{'column':'id','op':'eval','value':'1'}]},
    {'where':[{'column':'id','op':'exists','value':'1'}]},{'where':[{'column':'id','op':'eq'}]},
    {'where':[{'column':'id','op':'contains','value':True}]},{'where':[{'column':'id','op':'num_ge','value':'NaN'}]},
    {'after':True},{'after':-1},{'limit':True},{'limit':0},{'limit':65},{'header_row':1.0},
    {'unknown':'x'},{'path':'/records'},{'worksheet':'missing'},{'archive_member':'missing'},
])
def test_invalid_requests_are_explicit_errors(change):
    result=fetch(web(),req(**change))
    assert 'error' in result and 'text' not in result


def test_json_exact_path_pointer_escapes_array_indices_and_numeric_lexemes():
    source=' {"records":[{"a/b":{"~n":900719925474099312345},"items":[{"name":"中文"}],"v":1.2300e+02,"b":false,"n":null}]} '
    request={'format':'json','path':'/records','columns':['/a~1b/~0n','/items/0/name','/v','/b','/n','/absent']}
    result=fetch(web(source),request)
    assert_spans(source,result)
    cells=result['row_view']['records'][0]['cells']
    assert cells['/a~1b/~0n']['value']=='900719925474099312345'
    assert cells['/v']['value']=='1.2300e+02' and cells['/v']['kind']=='number'
    assert cells['/b']['value'] is False and cells['/n']['value'] is None
    assert cells['/absent']=={'present':False}
    for cell in cells.values():
        if cell['present']:
            literal=source[cell['start']:cell['end']]
            assert literal==cell['value'] if cell['kind']=='number' else json.loads(literal)==cell['value']


@pytest.mark.parametrize('source',[
 '[{"x":1,"x":2}]','{"outside":{"x":1,"x":2},"rows":[{"x":1}]}',
 '[{"x":NaN}]','[{"x":Infinity}]','[{"x":01}]','[{"x":1,}]','[{"x":truefalse}]',
 '[{"x":"\\ud800"}]','[{"x":"bad\nstring"}]','[] trailing','[', '[1,]', '{1:2}',
])
def test_json_strict_full_document_duplicate_keys_and_scalar_boundaries(source):
    with pytest.raises(ValueError): select(source,{'format':'json','columns':['/x']})


@pytest.mark.parametrize('path', ['records','$.records','/bad~2pointer','/~'])
def test_no_json_dotpath_or_invalid_pointer_guessing(path):
    with pytest.raises(ValueError): select('[]',{'format':'json','path':path,'columns':['/x']})


def test_json_array_index_is_exact_not_python_negative_or_leading_zero():
    result=select('[["A","B"]]',{'format':'json','columns':['/0','/01','/-1','/-']})['records'][0]['cells']
    assert result['/0']['value']=='A'
    assert all(result[key]=={'present':False} for key in ['/01','/-1','/-'])


@pytest.mark.parametrize('source,spec',[
 ('{}',{'format':'json','columns':['/x']}),
 ('[{"x":{"y":1}}]',{'format':'json','columns':['/x']}),
 ('[{"x":1}]',{'format':'json','columns':['/x'],'where':[{'column':'/x','op':'eq','value':'1'}]}),
 ('[{"x":true}]',{'format':'json','columns':['/x'],'where':[{'column':'/x','op':'num_eq','value':'1'}]}),
 ('[{"x":1}]',{'format':'json','columns':['/x'],'where':[{'column':'/x','op':'contains','value':'1'}]}),
 ('[{"x":"1"}]',{'format':'json','columns':['/x'],'header_row':0}),
])
def test_json_no_implicit_structure_or_type_coercion(source,spec):
    with pytest.raises(ValueError): select(source,spec)


def test_json_exact_scalar_types_and_explicit_decimal_predicates():
    source='[{"x":true,"n":1.00},{"x":"true","n":2},{"x":null,"n":3}]'
    request={'format':'json','columns':['/x','/n'],'where':[{'column':'/x','op':'eq','value':True},{'column':'/n','op':'num_eq','value':'1'}]}
    result=select(source,request)
    assert len(result['records'])==1 and result['records'][0]['cells']['/n']['value']=='1.00'


def test_cache_only_default_off_and_zero_match_never_mark_new_evidence(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('Network forbidden')
    monkeypatch.setattr('slime_plugins.dcc.tools._download_public', forbidden)
    missing=WebTools(web_cached_rows=True)
    result=fetch(missing)
    assert 'cache' in result['error'] and missing.pages=={} and not missing._read_windows
    disabled=web(enabled=False)
    assert 'disabled' in fetch(disabled)['error'] and not disabled._read_windows
    instance=web()
    result=fetch(instance,req(where=[{'column':'id','op':'eq','value':'none'}]))
    assert result['row_view']['records']==[] and 'text' not in result
    assert instance.pages[URL]==CSV and not instance._read_windows
    evidence=_WebEvidence(); evidence.observe('fetch_url',result)
    assert evidence.fetches==0 and not evidence.fetched_sources


def test_limit_cursor_and_char_budget_page_over_real_matching_records():
    source='id,name\n'+''.join(f'{i},name{i}\n' for i in range(30))
    instance=web(source,budget=3500)
    seen=[]; after=0
    while True:
        result=fetch(instance,req(after=after,limit=64))
        assert 'error' not in result
        assert len(json.dumps(result,ensure_ascii=False))<=3500
        assert_spans(source,result)
        view=result['row_view']; assert view['matching_records_after']==30-after
        seen += [row['record_index'] for row in view['records']]
        if view['next_after'] is None: break
        assert view['next_after']>after
        after=view['next_after']
    assert seen==list(range(30))
    assert fetch(instance,req(after=31))['error']
    assert fetch(instance,req(),start=1)['error']


def test_overlarge_row_and_metadata_are_atomic_errors_without_reader_progress():
    for source, request in [('id,name\n1,'+'a'*6000+'\n',req()),('id,name\n',req(columns=['id']))]:
        instance=web(source,budget=300)
        result=fetch(instance,request)
        assert 'error' in result and not instance._read_windows
        assert len(json.dumps(result,ensure_ascii=False))<=300


def test_each_literal_record_has_truthful_repeat_telemetry_and_ledger_spans():
    instance=web('id,name\n1,A\n2,B\n')
    first=fetch(instance)
    assert all(r['read_progress']['new_text'] for r in first['row_view']['records'])
    second=fetch(instance)
    assert all(not r['read_progress']['new_text'] and r['read_progress']['previous_identical_reads']==1
               for r in second['row_view']['records'])
    ledger=WebNavigationLedger()
    ledger.observe_events([{'tool':'fetch_url','observation':first}])
    snapshot=json.loads(ledger.render().removeprefix(WEB_NAVIGATION_PREFIX))
    spans={(w['start'],w['end']) for w in snapshot['recent_windows']}
    assert spans=={(r['start'],r['end']) for r in first['row_view']['records']}
    assert (first['row_view']['header']['start'], first['row_view']['header']['end']) not in spans
    additional=[w for w in snapshot['recent_windows'] if w.get('kind')=='literal-record-window']
    assert len(additional)==1 and additional[0]['selector_version']==WEB_CACHED_ROWS_VERSION


def test_normal_cached_fetch_and_selector_share_actual_window_history():
    instance=web('id,name\n1,A\n')
    ordinary=asyncio.run(instance.fetch_url(URL,start=8))
    selected=fetch(instance)
    assert selected['text']==ordinary['text']
    assert selected['row_view']['records'][0]['read_progress']['new_text'] is False


def test_xlsx_and_archive_require_explicit_unambiguous_observed_boundaries():
    source='Workbook\n\nWorksheet: "Sheet"\nid,name\n1,A\n'
    start=source.index('\nWorksheet:')
    sheet={'name':'Sheet','start':start,'end':len(source)}
    result=select(source,req(worksheet='Sheet'),worksheets=[sheet])
    assert result['records'][0]['text']=='1,A\n'
    row=result['records'][0]; assert source[row['start']:row['end']]==row['text']
    for sections,request in [([sheet],req()),([sheet,sheet],req(worksheet='Sheet')),
                             ([{**sheet,'name_truncated':True}],req(worksheet='Sheet')),
                             ([{**sheet,'start':start+1}],req(worksheet='Sheet'))]:
        with pytest.raises(ValueError): select(source,request,worksheets=sections)
    source='Archive\nmember\nid,name\n1,A\n'
    member={'name':'data.csv','status':'readable','format':'csv','content_start':source.index('id,name'),'content_end':len(source)}
    result=select(source,req(archive_member='data.csv'),archives=[member])
    assert source[result['records'][0]['start']:result['records'][0]['end']]=='1,A\n'
    with pytest.raises(ValueError): select(source,req(),archives=[member])
    with pytest.raises(ValueError): select(source,req(archive_member='data.csv'),archives=[{**member,'format':'json'}])


def test_resource_bounds_fail_instead_of_incomplete_success(monkeypatch):
    for key, value, source, request in [
        ('source_chars',5,'id,name\n1,A\n',req()),('records',1,'id,name\n1,A\n2,B\n',req()),
        ('nodes',2,'[{"x":1}]',{'format':'json','columns':['/x']}),
        ('depth',1,'[{"x":{"y":1}}]',{'format':'json','columns':['/x/y']}),
    ]:
        with monkeypatch.context() as scoped:
            scoped.setitem(WEB_CACHED_ROWS_LIMITS,key,value)
            with pytest.raises(ValueError): select(source,request)


@pytest.mark.parametrize('review,navigation',[(False,False),(True,False),(False,True),(True,True)])
def test_actual_episode_opt_in_uses_existing_turns_schema_and_guard(review,navigation):
    class Cached(WebTools):
        async def web_search(self,query,count=8):
            return {'results':[{'url':URL,'title':'Synthetic records','snippet':'navigation'}]}
        async def fetch_url(self,url,start=0,select_rows=None):
            if select_rows is None and url not in self.pages: self.pages[url]=CSV
            return await super().fetch_url(url,start,select_rows)
    actions=[{'tool':'web_search','arguments':{'query':'synthetic'}},
             {'tool':'fetch_url','arguments':{'url':URL}},
             {'tool':'fetch_url','arguments':{'url':URL,'select_rows':req(where=[{'column':'id','op':'eq','value':'1'}])}},
             {'answer':[{'name':'Alpha, one','citations':[URL]}]}]
    if review: actions=[HELPERS['reviewed'](action) for action in actions]
    policy=Policy(actions)
    result=asyncio.run(EpisodeRunner(max_steps=4,max_new_tokens=97,web_tools_factory=Cached,
        web_cached_rows=True,web_structured_final=True,web_structured_actions=True,
        web_explicit_review=review,web_navigation_ledger=navigation).run(task(),policy,evaluation=True))
    assert result.status=='completed' and len(result.steps)==4 and len(result.tool_events)==3
    assert all(call[2]['max_new_tokens']==97 for call in policy.calls)
    assert WEB_CACHED_ROWS_INSTRUCTIONS in result.messages[0]['content']
    assert 'SYNTHETIC_NOT_FOR_PROMPT' not in json.dumps(policy.calls)
    assert 'select_rows' not in json.dumps(policy.calls[0][2]['json_schema'])
    assert 'select_rows' in json.dumps(policy.calls[1][2]['json_schema'])
    assert 'select_rows' not in json.dumps(policy.calls[-1][2]['json_schema'])
    assert result.metrics['evidence_guard']['accepted'] is True
    assert_spans(CSV,result.tool_events[2]['observation'])


def test_schema_off_unchanged_and_selector_on_requires_search():
    off=web_action_schema(['name'],[],successful_search=True)
    on=web_action_schema(['name'],[],successful_search=True,cached_rows=True)
    selected={'tool':'fetch_url','arguments':{'url':URL,'select_rows':req()}}
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(selected,off)
    jsonschema.validate(selected,on)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(selected,web_action_schema(['name'],[],successful_search=False,cached_rows=True))
    with pytest.raises(ValueError): EpisodeRunner(web_cached_rows=True)
    with pytest.raises(ValueError): web_cached_rows_configuration(1)


@pytest.mark.parametrize('damage',[None,'manifest','runtime','protocol','version','bool','limits_type','actions','backend','mode'])
def test_paired_protocol_requires_complete_matching_supported_identity(tmp_path,damage):
    helpers=runpy.run_path(str(ROOT/'tests/test_dcc_compare_runs.py'))
    pair=helpers['pair'](tmp_path,[helpers['web']('synthetic')])
    for run,declaration in zip(pair[:2],pair[2:],strict=True):
        for record in (run['manifest'],run['runtime'],declaration['common']['generation']):
            record.update(web_cached_rows=deepcopy(CONFIG),web_structured_final=web_structured_configuration(True),
                          web_structured_actions=web_structured_actions_configuration(True))
    if damage in {'manifest','runtime'}: pair[0][damage].pop('web_cached_rows')
    elif damage=='protocol': pair[2]['common']['generation'].pop('web_cached_rows')
    elif damage in {'version','bool','limits_type'}:
        value=pair[0]['manifest']['web_cached_rows']
        if damage=='limits_type': value['limits']['depth']=32.0
        else: value['version' if damage=='version' else 'enabled']='bad' if damage=='version' else 1
    elif damage=='actions': pair[0]['manifest'].pop('web_structured_actions')
    elif damage=='backend': pair[0]['runtime']['backend']='hf'
    elif damage=='mode':
        for record in (pair[1]['manifest'],pair[1]['runtime'],pair[3]['common']['generation']): record.pop('web_cached_rows')
    if damage:
        with pytest.raises(ValueError): helpers['COMPARE']['validate_protocols'](*pair)
    else: helpers['COMPARE']['validate_protocols'](*pair)


@pytest.mark.parametrize('damage',[None,'runtime','manifest','version','bool','limits_type','actions','backend'])
def test_actual_score_only_validates_cache_mode_identity(tmp_path,damage):
    helpers=runpy.run_path(str(ROOT/'tests/test_dcc_model_scoring.py'))
    tasks=helpers['logical_predictions'](tmp_path)
    for path in [*tmp_path.glob('manifest_rank*.json'),*tmp_path.glob('inference_runtime*.json')]:
        record=json.loads(path.read_text())
        record.update(web_cached_rows=deepcopy(CONFIG),web_structured_final=web_structured_configuration(True),
                      web_structured_actions=web_structured_actions_configuration(True))
        if (damage=='runtime' and path.name.startswith('inference_runtime') or damage=='manifest' and path.name.startswith('manifest')):
            record.pop('web_cached_rows')
        elif damage=='version': record['web_cached_rows']['version']='bad'
        elif damage=='bool': record['web_cached_rows']['enabled']=1
        elif damage=='limits_type': record['web_cached_rows']['limits']['depth']=32.0
        elif damage=='actions': record.pop('web_structured_actions')
        elif damage=='backend': record['inference_backend' if path.name.startswith('manifest') else 'backend']='hf'
        path.write_text(json.dumps(record))
    if damage:
        with pytest.raises(ValueError): helpers['cli']()['_score_predictions'](tmp_path,tasks)
    else: helpers['cli']()['_score_predictions'](tmp_path,tasks)


@pytest.mark.parametrize("enabled,bounded,navigation", [(False, False, False), (True, False, False), (True, True, True)])
def test_rows_evaluator_loader_metadata_resume_and_completed_runtime_validation(monkeypatch, tmp_path, enabled, bounded, navigation):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    policy = Policy([{"answer": []}], validate=False)
    policy.provenance = {"backend": "vllm", "model": "example/model"}
    async def close():
        pass
    policy.close = close
    if enabled:
        policy.provenance.update(web_cached_rows=deepcopy(CONFIG), web_structured_final=web_structured_configuration(True),
                                 web_structured_actions=web_structured_actions_configuration(True))
    if bounded:
        from slime_plugins.dcc.web_whitespace import web_bounded_whitespace_configuration
        policy.provenance["web_bounded_whitespace"] = web_bounded_whitespace_configuration(True)
    if navigation:
        from slime_plugins.dcc.web_navigation import web_navigation_configuration
        policy.provenance["web_navigation_ledger"] = web_navigation_configuration(True)
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task()])
    args.web_cached_rows = enabled
    args.web_navigation_ledger = navigation
    args.web_bounded_whitespace = bounded
    args.max_steps = 1
    if enabled:
        args.web_structured_final = args.web_structured_actions = True
    seen = []
    def load(*a, **kwargs):
        seen.append(kwargs)
        return policy
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    asyncio.run(evaluate(args))
    assert len(seen) == 1
    for filename in ["manifest_rank0.json", "inference_runtime.json"]:
        record = json.loads((tmp_path / filename).read_text())
        if enabled:
            assert record["web_cached_rows"] == CONFIG
        else:
            assert "web_cached_rows" not in record
    assert (seen[0].get("web_cached_rows") is True) is enabled
    assert (seen[0].get("web_navigation_ledger") is True) is navigation
    asyncio.run(evaluate(args))  # completed unchanged resume makes no second model load
    assert len(seen) == 1
    if enabled:
        runtime = json.loads((tmp_path / "inference_runtime.json").read_text())
        runtime["web_cached_rows"]["version"] = "unsupported"
        (tmp_path / "inference_runtime.json").write_text(json.dumps(runtime))
        with pytest.raises(ValueError, match="cached rows"):
            asyncio.run(evaluate(args))
        runtime["web_cached_rows"] = deepcopy(CONFIG)
        (tmp_path / "inference_runtime.json").write_text(json.dumps(runtime))
        args.web_cached_rows = False
        with pytest.raises(ValueError, match="manifest|settings|configuration"):
            asyncio.run(evaluate(args))


def test_rows_evaluator_rejects_loader_that_omits_enabled_runtime_declaration(monkeypatch, tmp_path):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    policy = Policy([{"answer": []}], validate=False)
    policy.provenance = {"backend": "vllm", "model": "example/model"}
    async def close():
        pass
    policy.close = close
    evaluate, args = helpers["stub_evaluation"](monkeypatch, tmp_path, policy, [task()])
    args.web_cached_rows = args.web_structured_actions = args.web_structured_final = True
    args.max_steps = 1
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=lambda *a, **kw: policy))
    with pytest.raises(ValueError, match="cached rows"):
        asyncio.run(evaluate(args))
    assert not (tmp_path / "inference_runtime.json").exists()


def test_rows_real_loader_attests_option_using_cpu_mocked_engine(monkeypatch):
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_vllm_policy.py"))
    helpers["runtime"].__wrapped__(monkeypatch)
    import transformers
    from slime_plugins.dcc import vllm_policy
    monkeypatch.setattr(vllm_policy, "preflight_web_structured", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(StructuredOutputsConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3_moe"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: helpers["Tokenizer"]())
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(eos_token_id=1))
    for enabled in [False, True]:
        policy = vllm_policy.load_vllm_policy("synthetic/model", web_structured_final=True,
                    web_structured_actions=True, web_cached_rows=enabled)
        if enabled:
            assert policy.provenance["web_cached_rows"] == CONFIG
        else:
            assert "web_cached_rows" not in policy.provenance


def test_rows_dependencies_fail_before_loading_models_or_data_and_legacy_absence_is_off():
    evaluate = runpy.run_path(str(ROOT / "scripts/dcc_model_evaluate.py"))["evaluate"]
    with pytest.raises(ValueError, match="web-cached-rows requires"):
        asyncio.run(evaluate(SimpleNamespace(web_cached_rows=True)))
    with pytest.raises(ValueError, match="requires structured Web actions"):
        EpisodeRunner(web_cached_rows=True)
    from slime_plugins.dcc.vllm_policy import load_vllm_policy
    with pytest.raises(ValueError, match="requires structured Web actions"):
        load_vllm_policy("synthetic/model", web_cached_rows=True)
    assert validate_web_cached_rows_records(({}, {})) == web_cached_rows_configuration(False)
    assert CONFIG["version"] == WEB_CACHED_ROWS_VERSION
    assert validate_web_cached_rows_records(({}, {"web_cached_rows": web_cached_rows_configuration(False)}))["enabled"] is False


@pytest.mark.parametrize('review,bounded',[(False,False),(True,False),(False,True),(True,True)])
def test_actual_pinned_decoder_compiles_and_accepts_selector_with_review_whitespace(review,bounded):
    xgrammar=pytest.importorskip('xgrammar')
    from xgrammar.testing import _is_grammar_accept_string
    from slime_plugins.dcc.web_review import web_review_schema
    from slime_plugins.dcc.web_review_decoder import web_review_grammar
    from slime_plugins.dcc.web_whitespace import web_bounded_whitespace_grammar
    schema=web_action_schema(['name'],[URL],successful_search=True,cached_rows=True)
    if review: schema=web_review_schema(schema)
    if bounded:
        grammar=xgrammar.Grammar.from_ebnf(web_bounded_whitespace_grammar(schema,explicit_review=review))
    elif review: grammar=xgrammar.Grammar.from_ebnf(web_review_grammar(schema))
    else: grammar=xgrammar.Grammar.from_json_schema(json.dumps(schema))
    action={'tool':'fetch_url','arguments':{'url':URL,'select_rows':{'format':'json','columns':['/name'],
        'where':[{'column':'/rank','op':'eq','value':'species'}],'path':'/results','limit':16}}}
    if review: action=HELPERS['reviewed'](action)
    assert _is_grammar_accept_string(grammar,json.dumps(action).replace('/','\\/'))


@pytest.mark.parametrize('benchmark,evaluation', [('cuad',True),('widesearch',False)])
def test_rows_flag_inert_for_contract_and_training(benchmark,evaluation):
    from slime_plugins.dcc import Task
    target=task() if benchmark=='widesearch' else Task('synthetic','cuad','validation','Find clause',document='Clause',gold=[])
    results=[]; calls=[]
    for enabled in [False,True]:
        policy=Policy([{'answer':[]}],validate=False)
        result=asyncio.run(EpisodeRunner(max_steps=1,web_structured_final=True,web_structured_actions=True,
            web_cached_rows=enabled).run(target,policy,evaluation=evaluation))
        results.append(result); calls.append(policy.calls)
    assert calls[0]==calls[1]
    assert all(getattr(results[0],key)==getattr(results[1],key) for key in ['messages','tool_events','answer','status','metrics'])


def test_actual_existing_extractors_supply_literal_member_and_worksheet_spans():
    from slime_plugins.dcc.tools import extract_page
    helpers=runpy.run_path(str(ROOT/'tests/test_dcc_web_archive.py'))
    sections=[]
    source=extract_page(helpers['zipped'](('rows.csv','id,name\n1,"Archive member: fake"\n'),('rows.json','[{"name":"中文"}]')),
                        'application/zip',archive_sections=sections)
    instance=web(source); instance.page_archives[URL]=sections
    result=fetch(instance,req(archive_member='rows.csv'))
    assert_spans(source,result)
    assert result['row_view']['records'][0]['cells']['name']['value']=='Archive member: fake'
    result=fetch(instance,{'format':'json','archive_member':'rows.json','columns':['/name']})
    assert_spans(source,result)
    assert result['row_view']['records'][0]['cells']['/name']['value']=='中文'
    helpers=runpy.run_path(str(ROOT/'tests/test_dcc_tools.py'))
    rows='<row r="1"><c t="inlineStr"><is><t>id</t></is></c><c t="inlineStr"><is><t>name</t></is></c></row>'
    rows+='<row r="2"><c><v>1</v></c><c t="inlineStr"><is><t>Worksheet: fake</t></is></c></row>'
    worksheets=[]
    source=extract_page(helpers['xlsx_workbook']([('Escaped "sheet"\nname',rows)]),'application/zip',xlsx_sections=worksheets)
    instance=web(source); instance.page_worksheets[URL]=worksheets
    result=fetch(instance,req(worksheet='Escaped "sheet"\nname'))
    assert_spans(source,result)
    assert result['row_view']['records'][0]['cells']['name']['value']=='Worksheet: fake'
    assert 'error' in fetch(instance,{'format':'tsv','worksheet':'Escaped "sheet"\nname','columns':['id']})


def test_cached_rows_preflight_rejects_non_action_mode_before_importing_engine():
    from slime_plugins.dcc.web_structured import preflight_web_structured
    with pytest.raises(ValueError,match='requires structured actions'):
        preflight_web_structured(cached_rows=True)


@pytest.mark.parametrize('request_value',[
    {'format':'json','columns':['/\ud800']},
    {'format':'json','columns':['/x'],'path':'/\ud800'},
    {'format':'csv','columns':['id'],'where':[{'column':'id','op':'eq','value':'\ud800'}]},
    {'format':'csv','columns':['id'],'worksheet':'\ud800'},
])
def test_unpaired_argument_surrogates_fail_before_observation_encoding(request_value):
    result=fetch(web('[{"x":"A"}]' if request_value['format']=='json' else CSV),request_value)
    assert 'error' in result and 'row_view' not in result
    json.dumps(result,ensure_ascii=False).encode('utf-8',errors='strict')
