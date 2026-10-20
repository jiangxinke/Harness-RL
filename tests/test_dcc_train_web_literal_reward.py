"""Synthetic-only contract controls; no actual task/label/log access."""
from pathlib import Path
import copy,importlib.util,json,random,sys,unittest
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dcc_train_web_literal_reward_test_module',HERE/'scripts/dcc_train_web_literal_reward.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def fixtures(gold=None):
    public={'id':'synthetic-new-protocol-case','benchmark':'widesearch','split':'train',
            'metadata':{'evaluation':{'required':['identifier','amount','nullable']}}}
    gold=copy.deepcopy(gold if gold is not None else [
        {'identifier':'00000000000000017','amount':'12345678901234567890.12345678901234567890','nullable':None},
        {'identifier':'CaseSensitive-X','amount':'-0.00000000000000000000100','nullable':'NA'}])
    label={'task_id':public['id'],'benchmark':'widesearch','split':'train',
           'gold':gold,'evaluation':copy.deepcopy(public['metadata']['evaluation']),
           'label_protocol':M.LABEL_PROTOCOL,'train_reward':M.configuration()}
    return public,label


class SyntheticControls(unittest.TestCase):
    def setUp(self):self.public,self.label=fixtures()
    def score(self,prediction,label=None,public=None,expected=None):
        return M.evaluate_train_literal_rows(prediction,label or self.label,public or self.public,
            expected_protocol=M.PROTOCOL if expected is None else expected)
    def assert_zero(self,prediction):
        score=self.score(prediction);self.assertEqual(score['train_exact_row_f1'],0);return score
    def test_gold_self_copy(self):
        score=self.score(copy.deepcopy(self.label['gold']))
        self.assertEqual(score['train_exact_row_f1'],1);self.assertEqual(score['train_exact_table_match'],1)
        self.assertFalse(score['grounding_evaluated'])
    def test_native_json_self_copy(self):
        self.assertEqual(self.score(json.dumps(self.label['gold']))['train_exact_row_f1'],1)
    def test_order_and_object_order_do_not_matter(self):
        rows=[dict(reversed(list(row.items())))for row in reversed(self.label['gold'])]
        self.assertEqual(self.score(rows)['train_exact_row_f1'],1)
    def test_decimal_spelling_and_precision_never_coerce(self):
        for wrong in ['12345678901234567890.12345678901234567891','1.2345678901234568e+19',
                      '12345678901234567000','12345678901234567890.123456789012345678900']:
            row=copy.deepcopy(self.label['gold'][0]);row['amount']=wrong
            score=self.score([row]);self.assertEqual(score['train_exact_row_f1'],0)
    def test_native_numeric_cells_are_invalid(self):
        for number in [0,17,1.25,True,False,float('inf'),float('nan')]:
            row=copy.deepcopy(self.label['gold'][0]);row['amount']=number
            self.assertFalse(self.assert_zero([row])['prediction_valid'])
    def test_numeric_json_lexemes_never_round(self):
        for number in ['9007199254740993','1.000000000000000000001','1e9999999999','NaN','Infinity']:
            raw='[{"identifier":"00000000000000017","amount":'+number+',"nullable":null}]'
            self.assertFalse(self.assert_zero(raw)['prediction_valid'])
    def test_leading_zero_id_case_and_space_are_exact(self):
        for wrong in ['17','000000000000000017','00000000000000017 ', ' 00000000000000017']:
            row=copy.deepcopy(self.label['gold'][0]);row['identifier']=wrong;self.assert_zero([row])
        row=copy.deepcopy(self.label['gold'][1]);row['identifier']='casesensitive-x';self.assert_zero([row])
    def test_unicode_codepoints_are_not_normalized(self):
        public,label=fixtures([{'identifier':'é','amount':'１','nullable':'A B'}])
        for column,value in [('identifier','e\u0301'),('amount','1'),('nullable','AB')]:
            row=copy.deepcopy(label['gold'][0]);row[column]=value
            self.assertEqual(self.score([row],label,public)['train_exact_row_f1'],0)
    def test_null_has_no_string_alias(self):
        for wrong in ['null','None','NA','N/A','',False]:
            row=copy.deepcopy(self.label['gold'][0]);row['nullable']=wrong;self.assert_zero([row])
    def test_missing_and_extra_fields_invalidate_entire_table(self):
        for change in ['missing','extra','capitalized']:
            rows=copy.deepcopy(self.label['gold'])
            if change=='missing':del rows[0]['nullable']
            if change=='extra':rows[0]['unexpected']='ignored?' 
            if change=='capitalized':rows[0]['Amount']=rows[0].pop('amount')
            self.assertFalse(self.assert_zero(rows)['prediction_valid'])
    def test_citations_only_are_excluded(self):
        rows=copy.deepcopy(self.label['gold'])
        for row in rows:row['citations']=['https://example.org/synthetic-source']
        self.assertEqual(self.score(rows)['train_exact_row_f1'],1)
        for malformed in [[],[''],['same','same'],'https://example.org',None]:
            rows[0]['citations']=malformed;self.assertFalse(self.assert_zero(rows)['prediction_valid'])
    def test_predicted_duplicates_are_extra_false_positives(self):
        rows=copy.deepcopy(self.label['gold']);rows.append(copy.deepcopy(rows[0]))
        score=self.score(rows);self.assertEqual(score['train_num_prediction_rows'],3)
        self.assertEqual(score['train_num_true_positive_rows'],2)
        self.assertAlmostEqual(score['train_exact_row_precision'],2/3)
        self.assertEqual(score['train_exact_row_recall'],1);self.assertAlmostEqual(score['train_exact_row_f1'],.8)
    def test_wrong_value_sharing_partial_entity_key_is_not_dropped(self):
        rows=copy.deepcopy(self.label['gold']);extra=copy.deepcopy(rows[0]);extra['amount']='WRONG';rows.append(extra)
        self.assertAlmostEqual(self.score(rows)['train_exact_row_f1'],.8)
    def test_gold_duplicate_is_contract_error(self):
        label=copy.deepcopy(self.label);label['gold'].append(copy.deepcopy(label['gold'][0]))
        with self.assertRaises(M.RewardContractError):self.score(label['gold'],label)
    def test_all_required_cells_are_compound_identity(self):
        label=copy.deepcopy(self.label);other=copy.deepcopy(label['gold'][0]);other['amount']='second literal value';label['gold'].append(other)
        self.assertEqual(self.score(label['gold'],label)['train_exact_row_f1'],1)
        self.assertLess(self.score(self.label['gold'],label)['train_exact_row_f1'],1)
    def test_empty_table_conventions(self):
        public,label=fixtures([])
        self.assertEqual(self.score([],label,public)['train_exact_row_f1'],1)
        self.assertEqual(self.score(self.label['gold'],label,public)['train_exact_row_f1'],0)
        self.assertEqual(self.score([])['train_exact_row_f1'],0)
    def test_empty_or_all_null_row_invalid(self):
        for row in [{'identifier':None,'amount':None,'nullable':None},
                    {'identifier':'','amount':'','nullable':None}]:
            self.assertFalse(self.assert_zero([row])['prediction_valid'])
            label=copy.deepcopy(self.label);label['gold']=[row]
            with self.assertRaises(M.RewardContractError):self.score([row],label)
    def test_protocol_requires_both_explicit_caller_and_complete_label(self):
        for expected in ['', 'strict_exact_proxy','csv-inference-v2']:
            with self.assertRaises(M.RewardContractError):self.score(self.label['gold'],expected=expected)
        for key in ['train_reward','label_protocol']:
            label=copy.deepcopy(self.label);label.pop(key)
            with self.assertRaises(M.RewardContractError):self.score(label['gold'],label)
        for change in ['version','omission','extra','bool-int']:
            label=copy.deepcopy(self.label)
            if change=='version':label['train_reward']['version']='another'
            if change=='omission':label['train_reward'].pop('row_key')
            if change=='extra':label['train_reward']['relax']=True
            if change=='bool-int':label['train_reward']['limits']['columns']=True
            with self.assertRaises(M.RewardContractError):self.score(label['gold'],label)
    def test_legacy_evaluation_not_reinterpreted(self):
        label=copy.deepcopy(self.label);label['evaluation'].update(unique_columns=['identifier'],eval_pipeline={})
        with self.assertRaises(M.RewardContractError):self.score(label['gold'],label)
    def test_dev_test_other_benchmarks_rejected(self):
        for side in ['label','public']:
            for field,value in [('split','validation'),('split','test'),('benchmark','cuad')]:
                label,public=copy.deepcopy(self.label),copy.deepcopy(self.public)
                (label if side=='label'else public)[field]=value
                with self.assertRaises(M.RewardContractError):self.score(label['gold'],label,public)
    def test_mismatched_ids_columns_or_public_gold_rejected(self):
        for change in ['id','columns','publicgold','surrogate']:
            label,public=copy.deepcopy(self.label),copy.deepcopy(self.public)
            if change=='id':label['task_id']='other'
            if change=='columns':public['metadata']['evaluation']['required'].reverse()
            if change=='publicgold':public['gold']=[]
            if change=='surrogate':label['gold'][0]['amount']='\ud800'
            with self.assertRaises(M.RewardContractError):self.score(label['gold'],label,public)
    def test_only_native_json_no_markdown_or_duplicate_object_keys(self):
        for raw in ['```json\n[]\n```','| identifier |\n| --- |\n| x |',
                    '[{"identifier":"one","identifier":"two","amount":"x","nullable":null}]', '{}']:
            self.assertFalse(self.assert_zero(raw)['prediction_valid'])
    def test_resource_limits_fail_closed(self):
        row=copy.deepcopy(self.label['gold'][0]);row['amount']='x'*65537
        self.assertFalse(self.assert_zero([row])['prediction_valid'])
        self.assertFalse(self.assert_zero([copy.deepcopy(self.label['gold'][0])]*4097)['prediction_valid'])
        row['amount']='\ud800';self.assertFalse(self.assert_zero([row])['prediction_valid'])
    def test_metric_names_do_not_impersonate_official_metrics(self):
        result=self.score(self.label['gold'])
        self.assertFalse({'score','f1_by_row','f1_by_item','strict_exact_proxy'}&set(result))
        self.assertEqual(result['evaluation_protocol'],M.PROTOCOL)
    def test_seeded_synthetic_self_copy_and_mutation_property(self):
        rng=random.Random(47);alphabet='aAzZ019 -._é中'
        for index in range(200):
            rows=[{'identifier':str(index).zfill(30),'amount':''.join(rng.choice(alphabet)for _ in range(48)),
                   'nullable':None if index%2 else 'null'}]
            public,label=fixtures(rows)
            self.assertEqual(self.score(rows,label,public)['train_exact_row_f1'],1)
            wrong=copy.deepcopy(rows);wrong[0]['amount']+='x'
            self.assertEqual(self.score(wrong,label,public)['train_exact_row_f1'],0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not any(name in sys.modules for name in ['pandas','torch','vllm','transformers','slime_plugins'])
    receipt={'schema':'dcc-train-native-literal-reward-synthetic-controls-v1','passed':result.wasSuccessful(),
             'test_methods':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'seeded_self_copy_and_wrong_cell_cases':400,'real_task_or_label_reads':False,
             'production_module_imported_or_modified':False,'GPU_model_network_calls':False,
             'selection_or_training_started':False,'offline_DPO_supported':False}
    (HERE/'CPU-CONTROLS.actual.private.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));raise SystemExit(not result.wasSuccessful())
