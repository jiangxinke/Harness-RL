"""Synthetic CPU control for constant tokenizer-vocabulary lookup cost."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'scripts')]
import dcc_grounded_final_builder as B
import dcc_grounded_final_sft as T
import test_grounded_final as F

class Counted(F.SyntheticTokenizer):
    def __init__(self):self.length_calls=0
    def __len__(self):
        self.length_calls+=1
        if self.length_calls>1:raise AssertionError('Tokenizer vocabulary looked up repeatedly')
        return 160000

class TokenizerSize(unittest.TestCase):
    def test_only_one_actual_vocabulary_lookup_for_all_rows_and_tokens(self):
        with tempfile.TemporaryDirectory(prefix='dcc-synthetic-vocabulary-') as temporary:
            path=Path(temporary)/'dataset';_,rows,manifest,approval=F.synthetic_dataset(path)
            tokenizer=Counted()
            loaded,_,_=T.load_reviewed_rows(path,approval,tokenizer,training_source_commit='e'*40,derivative_id='synthetic-derivative')
            self.assertEqual(loaded,rows);self.assertEqual(tokenizer.length_calls,1)
    def test_cached_limit_still_rejects_negative_upperbound_and_boolean_tokens(self):
        for invalid in (-1,160000,True):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory(prefix='dcc-synthetic-invalid-token-') as temporary:
                path=Path(temporary)/'dataset';_,rows,manifest,approval=F.synthetic_dataset(path)
                rows[0]['target_ids'][0]=invalid;payload=B.rows_bytes(rows)
                manifest['rows_bytes']=len(payload);manifest['rows_blake2b_256']=B.digest(payload)
                (path/'rows.jsonl').write_bytes(payload);(path/'manifest.json').write_text(B.canonical(manifest))
                approval=T.training_binding(manifest,training_source_commit='e'*40,derivative_id='synthetic-derivative')
                tokenizer=Counted()
                with self.assertRaisesRegex(ValueError,'token IDs exceed'):
                    T.load_reviewed_rows(path,approval,tokenizer,training_source_commit='e'*40,derivative_id='synthetic-derivative')
                self.assertEqual(tokenizer.length_calls,1)

if __name__=='__main__':unittest.main(verbosity=2)
