"""Invented CPU-only fixtures; no real benchmark/model/tokenizer/labels read."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'scripts')]
import dcc_grounded_final_builder as B
import dcc_grounded_final_sft as T

SOURCE = Path(os.environ.get('DCC_SYNTHETIC_RUNTIME_SOURCE',
    str(HERE.parent))).resolve()

class SyntheticTokenizer:
    """Artificial single-character tokenizer, never official Qwen."""
    eos_token_id = 151645
    chat_template = 'synthetic-template'
    tags = {'<|im_start|>':151644, '<|im_end|>':151645, '<|endoftext|>':151643,
            '<think>':151667, '</think>':151668}
    def __len__(self): return 160000
    def encode(self, text, *, add_special_tokens=False):
        return self(text, add_special_tokens=False, return_offsets_mapping=True)['input_ids']
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        ids, offsets, position = [], [], 0
        for part in re.split('('+'|'.join(re.escape(tag) for tag in self.tags)+')', text):
            if part in self.tags:
                ids.append(self.tags[part]); offsets.append((position, position+len(part)))
            else:
                ids.extend(10000+ord(c) for c in part)
                offsets.extend((position+i,position+i+1) for i in range(len(part)))
            position += len(part)
        return dict(input_ids=ids, offset_mapping=offsets)
    def decode(self, ids, *, skip_special_tokens):
        reverse = {v:k for k,v in self.tags.items()}
        return ''.join(reverse[t] if t in reverse else chr(t-10000) for t in ids
                       if not (skip_special_tokens and t in reverse))
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        text = ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)
        if add_generation_prompt: text += '<|im_start|>assistant\n<think>\n'
        return self.encode(text) if tokenize else text

def chunks(document, ranges):
    return [dict(message_index=2,hit_index=i,start=a,end=b,text=document[a:b])
            for i,(a,b) in enumerate(ranges)]

def observation(document, ranges):
    return {'role':'user','content':B.SOURCE_PREFIXES[0]+json.dumps({'document_length':len(document),
        'hits':[dict(start=a,end=b,text=document[a:b],score=1.) for a,b in ranges]})}

def synthetic_dataset(directory):
    """Small invented dataset testing exact trainer revalidation independently of builder."""
    tok = SyntheticTokenizer()
    identity = dict(derivative_id='synthetic-derivative',model_revision='synthetic-revision',
        protocol='synthetic-native-protocol',tokenizer_vocabulary_size=len(tok),
        chat_template=tok.chat_template,primary_eos_token_id=tok.eos_token_id,
        eos_token_ids=[151643,151645],max_model_len=10000)
    rows=[]
    for positive in (True,False):
        target = B.encode_target(tok,['Observed clause.'] if positive else [],final_review=False,
                                 eos_ids=identity['eos_token_ids'])
        p,h=[20,21],[22,B.END_THINK]
        row=dict(protocol=B.PROTOCOL,benchmark='cuad',split='train',task_id=str(positive),
            positive=positive,gold=['Observed clause.'] if positive else [],
            span_mappings=[dict(target_text='Observed clause.')] if positive else [],
            prompt_ids=p+h,parent=dict(original_prompt_ids=p,actual_thought_ids=h),
            target_is_actual_native_generation=False,original_prompt_and_thought_receive_loss=False,
            target_eos_is_synthetic_training_token=True,**target)
        row['total_tokens']=len(row['prompt_ids'])+len(row['target_ids']); rows.append(row)
    payload=B.rows_bytes(rows)
    manifest=dict(protocol=B.PROTOCOL,launchable=True,native_only_training=False,overlength=[],
        rows=2,positive=1,negative=1,parent_source_receipt=dict(source_commit='f'*40),
        parent_profile=dict(identity=identity),max_sequence_tokens=10000,
        rows_bytes=len(payload),rows_blake2b_256=B.digest(payload))
    directory.mkdir()
    (directory/'rows.jsonl').write_bytes(payload)
    (directory/'manifest.json').write_text(B.canonical(manifest))
    approval=T.training_binding(manifest,training_source_commit='e'*40,derivative_id='synthetic-derivative')
    return tok,rows,manifest,approval

class Contracts(unittest.TestCase):
    def test_whitespace_mapping_uses_exact_source_and_complete_gold(self):
        document='Clause A.  Alpha\t preserves\nBeta rights!  Alpha preserves Beta rights!'
        got=B.map_gold(document,['Alpha preserves Beta rights!'],chunks(document,[(0,len(document))]))
        self.assertEqual(got[0]['match'],'exact')
        self.assertEqual(got[0]['document_start'],document.rfind('Alpha'))
        document='Alpha\t preserves\nBeta rights!'
        got=B.map_gold(document,['Alpha preserves Beta rights!'],chunks(document,[(0,len(document))]))
        self.assertEqual(got[0]['target_text'],document)
        self.assertEqual(got[0]['match'],'whitespace-offset-mapped')
        self.assertIsNone(B.map_gold(document,['Alpha preserves beta rights!'],chunks(document,[(0,len(document))])))
        self.assertIsNone(B.map_gold(document,['Alpha preserves Beta rights?'],chunks(document,[(0,len(document))])))
        self.assertIsNone(B.map_gold(document,[document,'missing'],chunks(document,[(0,len(document))])))
        self.assertEqual(B.map_gold(document,[],chunks(document,[(0,5)])),[])
    def test_single_chunk_default_union_explicit_and_gaps_reject(self):
        document='Alpha preserves Beta rights!'
        adjacent=chunks(document,[(0,14),(14,len(document))])
        self.assertIsNone(B.map_gold(document,[document],adjacent))
        self.assertEqual(B.map_gold(document,[document],adjacent,allow_contiguous_union=True)[0]['target_text'],document)
        overlap=chunks(document,[(0,17),(10,len(document))])
        self.assertEqual(B.map_gold(document,[document],overlap,allow_contiguous_union=True)[0]['target_text'],document)
        self.assertIsNone(B.map_gold(document,[document],chunks(document,[(0,14),(15,len(document))]),allow_contiguous_union=True))
    def test_only_complete_offset_bound_source_in_original_P_is_evidence(self):
        document='Same quotation. gap Same quotation.'; start=document.rfind('Same')
        msg=observation(document,[(start,len(document))]); wrong=observation(document,[(0,15)])
        self.assertEqual(B.observed_chunks(document,[msg],'system\n'+msg['content']+'assistant')[0]['start'],start)
        for p in ('Same quotation.', wrong['content'], msg['content'].replace('"start": '+str(start),'"start": 0')):
            with self.subTest(p=p),self.assertRaises(ValueError): B.observed_chunks(document,[msg],p)
        with self.assertRaises(ValueError): B.observed_chunks(document,[msg],'P before actual source in H')
        broken=deepcopy(msg);broken['content']=broken['content'].replace('"start": '+str(start),'"start": false')
        with self.assertRaises(ValueError):B.observed_chunks(document,[broken],broken['content'])
    def test_actual_PH_and_single_natural_close_required(self):
        B.verify_conditioning([10,11],[12,B.END_THINK],[10,11,12,B.END_THINK],eos_ids=[151645])
        for h,phase2 in (([12],[10,11,12]),([12,B.END_THINK,B.END_THINK],[10,11,12,B.END_THINK,B.END_THINK]),
                         ([12,151645,B.END_THINK],[10,11,12,151645,B.END_THINK]),
                         ([12,B.END_THINK],[10,11,B.END_THINK])):
            with self.subTest(h=h),self.assertRaises(ValueError):B.verify_conditioning([10,11],h,phase2,eos_ids=[151645])
    def test_full_json_offset_masks_only_confidence_and_review_values(self):
        tok=SyntheticTokenizer();span='Observed "quote" 0.9\nrights'
        target=B.encode_target(tok,[span],final_review=True,eos_ids=[151643,151645])
        text,start,ignored=B.target_text([span],final_review=True)
        offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        self.assertEqual(target['target_ids'][:-1],tok.encode(text))
        for i,(a,b) in enumerate(offsets):
            self.assertEqual(target['loss_mask'][i],a>=start and not any(a<r and b>l for l,r in ignored))
        embedded=text.index('0.9'); self.assertTrue(all(target['loss_mask'][embedded:embedded+3]))
        self.assertTrue(target['loss_mask'][-1]);self.assertEqual(target['target_ids'][-1],151645)
        # Conservative masking when a tokenizer token crosses a confidence boundary.
        class Merged(SyntheticTokenizer):
            def __call__(self,text,*,add_special_tokens,return_offsets_mapping):
                base=super().__call__(text,add_special_tokens=False,return_offsets_mapping=True)
                marker=text.index('0.9',text.index('"confidence"'));left=marker-1;right=marker+4
                return dict(input_ids=base['input_ids'][:left]+[159999]+base['input_ids'][right:],
                            offset_mapping=base['offset_mapping'][:left]+[(left,right)]+base['offset_mapping'][right:])
            def decode(self,ids,*,skip_special_tokens):
                result=[]
                for v in ids:result.append(':0.9}' if v==159999 else super().decode([v],skip_special_tokens=skip_special_tokens))
                return ''.join(result)
        merged=B.encode_target(Merged(),['Clause.'],final_review=False,eos_ids=[151643,151645])
        idx=merged['target_ids'].index(159999);self.assertFalse(merged['loss_mask'][idx])
    def test_global_sampler_exact_quotas_decks_and_order_invariance(self):
        rows=[dict(task_id=f'p{i:02}',positive=True) for i in range(44)]+[dict(task_id=f'n{i:02}',positive=False) for i in range(32)]
        plan=T.sample_plan(rows,seed=47,world=8,steps=32)
        tasks=[rows[i] for batch in plan for i in batch]
        self.assertEqual((len(tasks),sum(r['positive'] for r in tasks)),(256,171))
        for positive,count in ((True,44),(False,32)):
            ids=[r['task_id'] for r in tasks if r['positive']==positive]
            for left in range(0,len(ids),count):self.assertEqual(len(set(ids[left:left+count])),len(ids[left:left+count]))
        reordered=list(reversed(rows));other=T.sample_plan(reordered,seed=47,world=8,steps=32)
        self.assertEqual([[rows[i]['task_id'] for i in b] for b in plan],[[reordered[i]['task_id'] for i in b] for b in other])
        self.assertEqual(T.sample_plan(rows,seed=47,world=8,steps=32),plan)
        with self.assertRaises(ValueError):T.sample_plan(rows[:44],seed=47,world=8,steps=32)
    def test_reviewed_dataset_exact_binding_digest_mask_and_protocol(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-approved-') as temporary:
            path=Path(temporary)/'dataset';tok,rows,manifest,approval=synthetic_dataset(path)
            loaded,_,binding=T.load_reviewed_rows(path,approval,tok,training_source_commit='e'*40,derivative_id='synthetic-derivative')
            self.assertEqual(loaded,rows);self.assertEqual(binding,approval)
            self.assertFalse(binding['corrected_final_binding']['native_only_training'])
            self.assertFalse(binding['training_target_lineage']['loss_on_thought'])
            wrong=deepcopy(approval);wrong['corrected_final_binding']['native_only_training']=True
            with self.assertRaises(ValueError):T.load_reviewed_rows(path,wrong,tok,training_source_commit='e'*40,derivative_id='synthetic-derivative')
            rows[0]['loss_mask'][0]=True;payload=B.rows_bytes(rows);(path/'rows.jsonl').write_bytes(payload)
            with self.assertRaisesRegex(ValueError,'bytes'):T.load_reviewed_rows(path,approval,tok,training_source_commit='e'*40,derivative_id='synthetic-derivative')
            manifest['rows_bytes']=len(payload);manifest['rows_blake2b_256']=B.digest(payload)
            (path/'manifest.json').write_text(B.canonical(manifest));approval=T.training_binding(manifest,training_source_commit='e'*40,derivative_id='synthetic-derivative')
            with self.assertRaisesRegex(ValueError,'token mask'):T.load_reviewed_rows(path,approval,tok,training_source_commit='e'*40,derivative_id='synthetic-derivative')
    def test_actual_logprobs_tiny_causal_LoRA_gradients_match_full_forward_oracle(self):
        import torch
        torch.set_num_threads(1);torch.manual_seed(47)
        _,_,hf=T.import_runtime(SOURCE)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(32,8);self.base=torch.nn.Linear(8,32,bias=False)
                self.lora_A=torch.nn.Parameter(torch.randn(8,2)*.1);self.lora_B=torch.nn.Parameter(torch.randn(2,32)*.1)
                self.embedding.weight.requires_grad_(False);self.base.weight.requires_grad_(False)
                self.last_logits=None;self.hidden=None
            @property
            def device(self):return self.lora_A.device
            def forward(self,input_ids,use_cache=False,logits_to_keep=None):
                x=self.embedding(input_ids).detach().requires_grad_(True);x.retain_grad();self.hidden=x
                context=x.cumsum(1);logits=self.base(context)+context@self.lora_A@self.lora_B
                if logits_to_keep:logits=logits[:,-logits_to_keep:]
                logits.retain_grad();self.last_logits=logits
                return SimpleNamespace(logits=logits)
        model=Tiny();oracle=deepcopy(model)
        row=dict(prompt_ids=[1,2,3,4],target_ids=[5,6,7,8,9],loss_mask=[True,False,True,False,True])
        optimizer=torch.optim.SGD([model.lora_A,model.lora_B],lr=.01)
        before=model.lora_B.detach().clone();loss=T.backward_row(model,row,hf.completion_logprobs)
        ids=torch.tensor([row['prompt_ids']+row['target_ids']]);logits=oracle(ids).logits
        response=logits[0,len(row['prompt_ids'])-1:-1].float();targets=torch.tensor(row['target_ids'])
        lp=response.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
        expected=-lp[torch.tensor(row['loss_mask'])].mean();expected.backward()
        self.assertAlmostEqual(loss,float(expected.detach()),places=6)
        self.assertTrue(torch.allclose(model.lora_A.grad,oracle.lora_A.grad,atol=1e-7))
        self.assertTrue(torch.allclose(model.lora_B.grad,oracle.lora_B.grad,atol=1e-7))
        self.assertEqual(model.last_logits.grad[0,1].abs().sum().item(),0)
        self.assertEqual(model.last_logits.grad[0,3].abs().sum().item(),0)
        self.assertEqual(model.last_logits.grad[0,-1].abs().sum().item(),0)
        self.assertGreater(model.hidden.grad[0,:4].abs().sum().item(),0)
        self.assertTrue(torch.isfinite(model.lora_A.grad).all());self.assertGreater(model.lora_A.grad.norm().item(),0)
        optimizer.step();self.assertFalse(torch.equal(model.lora_B,before))
    def test_CPU_preflight_never_calls_model_or_CUDA(self):
        import transformers
        import torch
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-preflight-') as temporary:
            path=Path(temporary)/'dataset';tok,rows,manifest,approval=synthetic_dataset(path)
            approval_path=Path(temporary)/'approval.json';approval_path.write_text(B.canonical(approval))
            _,_,hf=T.import_runtime(SOURCE)
            with patch.object(T,'training_source_identity',return_value=approval['training_source_receipt']),\
                 patch.object(T,'validate_base_metadata',return_value={'synthetic_metadata_only':True}),\
                 patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=tok),\
                 patch.object(hf,'load_policy_model',side_effect=AssertionError('model loading forbidden')),\
                 patch.object(torch.cuda,'set_device',side_effect=AssertionError('CUDA forbidden')),\
                 patch('builtins.print') as printer:
                code=T.main(['--runtime-source-root',str(SOURCE),'--training-source-commit','e'*40,
                    '--derivative-id','synthetic-derivative','--model','synthetic-local-not-real',
                    '--dataset',str(path),'--approval',str(approval_path),'--output-dir',str(Path(temporary)/'never-created'),
                    '--cpu-preflight','--plan-world-size','8'])
                result=B.strict_json(printer.call_args.args[0])
            self.assertEqual(code,0);self.assertEqual(result['actual_examples'],256)
            self.assertEqual(result['positive_examples'],171);self.assertEqual(result['negative_examples'],85)
            self.assertFalse(result['CUDA_initialized']);self.assertFalse((Path(temporary)/'never-created').exists())



class IdentityContracts(unittest.TestCase):
    def test_source_guard_checks_actual_entry_root_HEAD_tracked_files_and_clean_code(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-source-') as temporary:
            root=Path(temporary);(root/'scripts').mkdir()
            def git(arguments,**kwargs):
                command=arguments[3:]
                value='e'*40 if command==['rev-parse','HEAD'] else ''
                return SimpleNamespace(returncode=0,stdout=value)
            with patch.object(T,'__file__',str(root/'scripts/dcc_grounded_final_sft.py')),\
                 patch.object(B,'__file__',str(root/'scripts/dcc_grounded_final_builder.py')),\
                 patch.object(T.subprocess,'run',side_effect=git) as run:
                self.assertEqual(T.training_source_identity(root,'e'*40)['source_commit'],'e'*40)
                calls=[call.args[0][3:] for call in run.call_args_list]
                self.assertTrue(any(call[:2]==['ls-files','--error-unmatch'] for call in calls))
                self.assertTrue(any(call[:2]==['status','--porcelain'] for call in calls))
                with self.assertRaisesRegex(ValueError,'commit differs'):T.training_source_identity(root,'d'*40)
                def dirty(arguments,**kwargs):
                    result=git(arguments,**kwargs)
                    if arguments[3]=='status':result.stdout=' M scripts/dcc_grounded_final_sft.py'
                    return result
                with patch.object(T.subprocess,'run',side_effect=dirty),self.assertRaisesRegex(ValueError,'modified or untracked'):
                    T.training_source_identity(root,'e'*40)
            with self.assertRaisesRegex(ValueError,'inside'):T.training_source_identity(root,'e'*40)
    def test_model_identity_uses_real_metadata_and_loaded_config(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-base-') as temporary:
            root=Path(temporary)
            pruning=dict(derivative_model_id='synthetic-derivative',source_revision='synthetic-revision',
                source_model_id='synthetic-repository',logical_parameters=1000,experts_per_layer=2,calibration_split='train')
            config=dict(model_type='qwen3_moe',num_experts=2,hidden_size=4,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=1,dcc_expert_pruning=pruning)
            artifact=dict(**pruning,physically_removed_experts=True)
            (root/'config.json').write_text(B.canonical(config));(root/'expert-pruning.json').write_text(B.canonical(artifact))
            identity=dict(derivative_id='synthetic-derivative',model_revision='synthetic-revision',model_repository='synthetic-repository')
            metadata=T.validate_base_metadata(root,identity)
            T.validate_loaded_base(SimpleNamespace(config=SimpleNamespace(to_dict=lambda:config)),metadata)
            broken=deepcopy(config);broken['num_experts']=3
            with self.assertRaisesRegex(ValueError,'Loaded base'):T.validate_loaded_base(SimpleNamespace(config=SimpleNamespace(to_dict=lambda:broken)),metadata)
            for key in ('model_revision','derivative_id','model_repository'):
                wrong=dict(identity);wrong[key]='different'
                with self.subTest(key=key),self.assertRaises(ValueError):T.validate_base_metadata(root,wrong)

if __name__=='__main__':unittest.main(verbosity=2)
