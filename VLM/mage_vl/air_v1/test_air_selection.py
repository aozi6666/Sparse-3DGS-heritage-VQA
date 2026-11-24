"""Synthetic control checks only; no datasets/video/model/GPU/API access."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import unittest

import numpy as np
from air_selection import (AIRConfig, coarse_events, localized_candidates,
    parse_score, rank_interval_candidates, select_frames, uniform_indices)


def score_record(score):
    return dict(score=score,raw_text='Malformed output' if score is None else f'Score: {score}\nReasoning: synthetic control')


class Callbacks:
    def __init__(self,mode='positive'):
        self.mode=mode;self.scored=[];self.similar=[];self.cache={};self.created=0
    def scores(self,indices):
        self.scored.append(indices[:]);out={}
        for i in indices:
            if i not in self.cache:
                value=5 if self.mode=='positive' else 2 if self.mode=='negative' else None
                if self.mode=='one':value=5 if not self.cache else 2
                self.cache[i]=score_record(value);self.created+=1
            out[i]=deepcopy(self.cache[i])
        return out
    def similarities(self,indices):
        self.similar.extend(indices)
        return {i:0.5 for i in indices}


def run(callbacks=None,config=None,total=1000,on_round=None):
    callbacks=callbacks or Callbacks()
    indices=list(range(0,total,10))
    if indices[-1]!=total-1:indices.append(total-1)
    return select_frames(total,30.0,indices,[0.5]*len(indices),callbacks.scores,
        callbacks.similarities,config or AIRConfig(),on_round)


class ParseAndInputs(unittest.TestCase):
    def test_explicit_score_only(self):
        self.assertEqual(parse_score('Score: 5\nReasoning: Three people.'),5)
        self.assertEqual(parse_score(' Score: 2 / 5 \nReasoning: irrelevant.'),2)
        for raw in ['3','Rating: 4','Reasoning: 5 people','Score: 10','Score: 5.0','Score: 5\nScore: 3','Score: 5\nScore: invalid','Score: five','']:
            self.assertIsNone(parse_score(raw),raw)
    def test_config_invalid(self):
        for kw in [dict(final_budget=0),dict(candidates_per_round=True),dict(max_iterations=-1),dict(local_growth=1),dict(gmm_gamma=float('nan')),dict(positive_threshold=5)]:
            with self.assertRaises(ValueError):AIRConfig(**kw)
    def test_uniform_exact_floor_and_short_video(self):
        import torch
        self.assertEqual(uniform_indices(50,32),torch.linspace(0,49,32).round().long().tolist())
        self.assertEqual(uniform_indices(3,32),[0,1,2])
        for value in [0,-1,True,1.2]:
            with self.assertRaises(ValueError):uniform_indices(value)
    def test_input_mismatch_and_nonfinite(self):
        cb=Callbacks()
        cases=[([0,2],[.1]),([0,0],[.1,.2]),([2,0],[.1,.2]),([0,10],[.1,.2]),([0,2],[.1,float('nan')])]
        for idx,vals in cases:
            with self.assertRaises(ValueError):select_frames(10,30,idx,vals,cb.scores,cb.similarities)
        self.assertFalse(cb.scored)
    def test_bad_callback_identities(self):
        for result in [{}, {None:score_record(5)}, {'1.0':score_record(5)}]:
            with self.assertRaises(ValueError):
                select_frames(2,30,[0,1],[.5,.5],lambda _:result,lambda idx:{i:.5 for i in idx})
    def test_callback_raw_score_mismatch_stops(self):
        with self.assertRaises(ValueError):
            select_frames(2,30,[0,1],[.5,.5],lambda idx:{i:dict(score=5,raw_text='Score: 1') for i in idx},lambda idx:{i:.5 for i in idx})
    def test_similarity_nan_stops(self):
        cb=Callbacks()
        with self.assertRaises(ValueError):
            select_frames(100,30,[10,20,30],[.5]*3,cb.scores,lambda idx:{i:float('nan') for i in idx})


class Ranking(unittest.TestCase):
    def setUp(self):
        self.raw={0:.1,25:.1,50:.1,75:.1,100:.1,125:.1,150:.95,175:.95,199:.95}
        self.probed=[0,50,100,150,199]
    def test_high_potential_late_video_not_earliest(self):
        chosen,intervals=rank_interval_candidates(self.probed,self.raw,set(),200,2)
        self.assertEqual(chosen,[150,199])
        self.assertEqual(intervals[0]['start'],150)
    def test_already_analyzed_does_not_consume_C(self):
        chosen,_=rank_interval_candidates(self.probed,self.raw,{150,199},200,2)
        self.assertEqual(len(chosen),2)
        self.assertFalse(set(chosen)&{150,199})
        self.assertIn(100,chosen)
    def test_ranked_candidates_unique_and_bounded(self):
        chosen,_=rank_interval_candidates(self.probed,self.raw,set(),200,100)
        self.assertEqual(len(chosen),len(set(chosen)))
        self.assertTrue(all(0<=i<200 for i in chosen))
    def test_empty_interval_total_one(self):
        called=[]
        result=select_frames(1,30,[0],[.5],lambda x:called.append(x),lambda x:called.append(x))
        self.assertEqual(result['final_indices'],[0]);self.assertFalse(called)


class EventsAndSearch(unittest.TestCase):
    def test_constant_signal_is_explicit_not_fake_GMM(self):
        events,diagnostic=coarse_events([0,30,60],[.5]*3,30,AIRConfig())
        self.assertEqual(diagnostic['mode'],'degenerate_constant')
        self.assertEqual(diagnostic['fallback'],'no_event_survived_use_full_coarse_range')
        self.assertEqual(events,[dict(start=0,end=60)])
    def test_actual_two_cluster_GMM(self):
        indices=list(range(0,600,30));values=[.1]*10+[.9]*10
        events,diagnostic=coarse_events(indices,values,30,AIRConfig())
        self.assertEqual(diagnostic['mode'],'two_component_gmm')
        self.assertTrue(.1<diagnostic['threshold']<.9)
        self.assertEqual(events,[dict(start=300,end=570)])
    def test_localized_search_uses_native_indices_and_bounds(self):
        result=localized_candidates([500],[0,500,999],1000,32,AIRConfig())
        self.assertTrue(result);self.assertTrue(any(400<i<600 for i in result))
        self.assertEqual(len(result),len(set(result)))
        self.assertFalse(set(result)&{0,500,999})
        self.assertTrue(all(0<=i<1000 for i in result))
    def test_cumulative_early_stop(self):
        config=AIRConfig(final_budget=8,candidates_per_round=4,max_iterations=6)
        cb=Callbacks();result=run(cb,config)
        self.assertEqual(result['stop_reason'],'cumulative_budget_reached')
        self.assertEqual(len(result['rounds']),2)
        self.assertEqual(len(result['final_indices']),8)
        self.assertEqual(cb.created,8)
    def test_negative_fallback_is_exact_uniform_baseline(self):
        cb=Callbacks('negative');result=run(cb)
        self.assertEqual(result['final_indices'],uniform_indices(1000,32))
        self.assertTrue(result['fallback_reason'])
        self.assertEqual(result['invalid_score_indices'],[])
        self.assertLessEqual(cb.created,12*6)
    def test_invalid_never_becomes_neutral_or_positive(self):
        cb=Callbacks('invalid');result=run(cb)
        self.assertEqual(len(result['invalid_score_indices']),cb.created)
        self.assertEqual(result['final_indices'],uniform_indices(1000,32))
        self.assertTrue(all(not r['positive_indices'] for r in result['rounds']))
    def test_partially_positive_not_padded_with_unverified_neighbors(self):
        cb=Callbacks('one');result=run(cb)
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(len(result['final_indices']),1)
        self.assertGreater(len(result['rounds'][0]['new_neighbor_indices']),0)
        self.assertEqual(result['final_indices'],result['rounds'][0]['positive_indices'])
        new_neighbors=set(result['rounds'][0]['new_neighbor_indices'])
        self.assertFalse(new_neighbors&set(result['rounds'][0]['candidates']))
    def test_replay_from_callback_cache_does_not_repeat_expensive_calls(self):
        cb=Callbacks('positive');trace=[];first=run(cb,on_round=trace.append);created=cb.created
        second=run(cb)
        self.assertEqual(first,second);self.assertEqual(cb.created,created)
        self.assertEqual(trace,first['rounds'])
        self.assertEqual(sum(len(r['candidates']) for r in trace),len(first['analyzed_indices']))
    def test_round_callback_failure_is_resumable_from_saved_scores(self):
        cb=Callbacks('positive')
        def interrupt(trace):raise RuntimeError('simulated interruption after cached scores')
        with self.assertRaises(RuntimeError):run(cb,on_round=interrupt)
        created=cb.created;resumed=run(cb);self.assertEqual(created,12)
        fresh=run(Callbacks('positive'))
        self.assertEqual(resumed,fresh)
    def test_all_results_json_finite_unique_and_chronological(self):
        result=run(Callbacks('positive'));json.dumps(result,allow_nan=False)
        final=result['final_indices'];self.assertEqual(final,sorted(set(final)))
        self.assertLessEqual(len(final),32);self.assertTrue(all(0<=i<1000 for i in final))
        self.assertEqual(result['final_seconds'],[i/30 for i in final])


if __name__=='__main__':unittest.main()
