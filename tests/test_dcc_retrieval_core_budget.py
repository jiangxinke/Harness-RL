"""Preserve bounded public retrieval cores before expanding their context."""
import copy
import random
import unittest

from slime_plugins.dcc.retrieval import expand_document_context
from slime_plugins.dcc.tools import DocumentTools

def merged_intervals(hits):
    merged = []
    for h in sorted(hits,key=lambda h:(h['start'],h['end'])):
        if merged and h['start'] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],h['end'])
        else:
            merged.append([h['start'],h['end']])
    return merged

def contains(hits,start,end):
    return any(lo<=start and hi>=end for lo,hi in merged_intervals(hits))

def observation(tools,intervals):
    return dict(document_length=len(tools.document),hits=[dict(start=a,end=b,text=tools.document[a:b],score=float(len(intervals)-i)) for i,(a,b) in enumerate(intervals)],extra={'keep':['metadata']})

def assert_valid(test,tools,old,new,cap):
    test.assertEqual(set(new),set(old))
    test.assertEqual(new['extra'],old['extra']) if 'extra' in old else None
    test.assertLessEqual(sum(h['end']-h['start'] for h in new['hits']),cap)
    for h in new['hits']:
        test.assertEqual(h['text'],tools.document[h['start']:h['end']])
        test.assertGreater(h['end'],h['start'])
    spans = sorted((h['start'],h['end']) for h in new['hits'])
    test.assertTrue(all(a[1]<b[0] for a,b in zip(spans,spans[1:])))

class RetrievalRegression(unittest.TestCase):
    def setUp(self):
        self.tools = DocumentTools(''.join(chr(0x400+i) for i in range(240)))

    def check(self,intervals,context,cap):
        obs = observation(self.tools,intervals)
        before = copy.deepcopy(obs)
        new = expand_document_context(self.tools,obs,context_chars=context,max_chars=cap)
        self.assertEqual(obs,before)
        assert_valid(self,self.tools,obs,new,cap)
        return obs,new

    def test_real_failure_shape_preserves_both_cores(self):
        obs,new = self.check([(10,20),(100,110)],5,25)
        self.assertTrue(all(contains(new['hits'],h['start'],h['end']) for h in obs['hits']))

    def test_spare_budget_expands_higher_rank_first(self):
        _,new = self.check([(10,20),(100,110)],5,30)
        self.assertEqual([(h['start'],h['end']) for h in new['hits']],[(5,25),(100,110)])

    def test_over_budget_cores_keep_whole_ranked_hits_and_later_overlap(self):
        _,new = self.check([(10,30),(60,80),(20,35)],1,25)
        self.assertEqual([(h['start'],h['end']) for h in new['hits']],[(10,35)])
        self.assertEqual(new['hits'][0]['score'],3.)

    def test_oversize_first_core_is_skipped_without_clipping(self):
        _,new = self.check([(0,40),(80,90)],1,20)
        self.assertEqual([(h['start'],h['end']) for h in new['hits']],[(79,91)])

    def test_overlap_adjacency_bridge_and_source_bounds(self):
        _,new = self.check([(20,40),(0,25),(40,50),(80,90)],800,100)
        self.assertEqual([(h['start'],h['end']) for h in new['hits']],[(0,50),(80,90)])
        _,bridge = self.check([(0,10),(20,30),(8,22)],2,30)
        self.assertEqual([(h['start'],h['end']) for h in bridge['hits']],[(0,30)])

    def test_zero_context_is_same_observation_even_with_small_cap(self):
        obs = observation(self.tools,[(0,40)])
        self.assertIs(expand_document_context(self.tools,obs,context_chars=0,max_chars=1),obs)

    def test_invalid_hits_and_options_fail_without_mutating_input(self):
        for change in [dict(start=True),dict(end=999),dict(text='not source'),dict(start=-1),dict(end=0)]:
            obs=observation(self.tools,[(0,10)])
            obs['hits'][0].update(change)
            before=copy.deepcopy(obs)
            with self.assertRaises(ValueError): expand_document_context(self.tools,obs,context_chars=1,max_chars=50)
            self.assertEqual(obs,before)
        for kwargs in [dict(context_chars=-1),dict(context_chars=True),dict(max_chars=0),dict(max_chars=True)]:
            with self.assertRaises(ValueError): expand_document_context(self.tools,observation(self.tools,[(0,10)]),**kwargs)

    def test_seeded_interval_invariants_including_small_budgets(self):
        rng=random.Random(47)
        for _ in range(500):
            intervals=[]
            for _ in range(rng.randint(0,8)):
                lo=rng.randrange(len(self.tools.document))
                intervals.append((lo,rng.randint(lo+1,len(self.tools.document))))
            cap=rng.randint(1,len(self.tools.document))
            obs,new=self.check(intervals,rng.randint(1,100),cap)
            core_size=sum(hi-lo for lo,hi in merged_intervals(obs['hits']))
            if core_size<=cap:
                self.assertTrue(all(contains(new['hits'],lo,hi) for lo,hi in intervals))


if __name__ == '__main__':
    unittest.main(verbosity=2)
