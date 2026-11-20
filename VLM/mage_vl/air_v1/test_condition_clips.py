"""自动证据路径CPU契约；临时数组不是评测数据。"""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import condition_clips as k
import condition_clips_run as r
import condition_clips_eval as ev
import air_protocol as p

class Conditions(unittest.TestCase):
 def test_first_and_final(self):
  self.assertIn('first_event',k.requirements('Which is first to start?'));self.assertIn('final_event',k.requirements('Who gets the final victory?'))
 def test_spatial_order_is_not_video_order(self):
  self.assertIn('spatial_order',k.requirements('In what order are four artworks in a single scene?'));self.assertNotIn('temporal_order',k.requirements('In what order are four artworks in a single scene?'))
 def test_explicit_items_all_kept(self):
  q='In what order?\n(a) plate.\n(b) cup.\n(c) straw.\n(d) snacks.'
  self.assertEqual(len(k.explicit_items(q)),4);self.assertIn('multiple_targets',k.requirements(q));self.assertEqual(k.query_texts('FULL',q,{'targets':['ignored extra']})[1:],['plate.','cup.','straw.','snacks.'])
 def test_relative_position(self):self.assertIn('relative_position',k.requirements('In which part of the video is he interviewed?'))
 def test_prompt_keeps_original_options(self):
  row=dict(question='Q?',bodies=[' one',' two',' three',' last option'])
  prompt=k.make_prompt(row,k.requirements('Q?'),'global',[0,100],[0,100],[])
  self.assertIn('D. last option',prompt);self.assertIn('original',prompt)

class Retrieval(unittest.TestCase):
 def setUp(self):
  self.ids=list(range(0,6000,30));self.features=np.random.default_rng(1).normal(size=(len(self.ids),768)).astype('float32');self.vec=np.ones((3,768),dtype='float32')
 def go(self,conditions):return k.propose(self.ids,self.features,self.vec,30.,6000,conditions)
 def test_fixed_six_nonoverlap_and_real_indices(self):
  w=self.go([]);self.assertEqual(len(w),6)
  for a in w:
   self.assertEqual(len(a['indices']),4);self.assertTrue(all(0<=i<6000 for i in a['indices']))
  for i,a in enumerate(w):
   for b in w[i+1:]:self.assertLessEqual(min(a['end'],b['end'])-max(a['start'],b['start']),1e-6)
 def test_first_has_contiguous_prefix(self):
  w=self.go(['first_event']);self.assertEqual([a['start'] for a in w[:2]],[0.,12.]);self.assertEqual(w[0]['origin'],'ordered_prefix')
 def test_final_has_tail(self):
  w=self.go(['final_event']);self.assertAlmostEqual(w[0]['end'],5999/30)
 def test_joint_budget(self):
  ix=k.combined_frames(6000,self.go(['first_event']));self.assertLessEqual(len(ix),32);self.assertEqual(ix,sorted(set(ix)));self.assertTrue(set(k.uniform_indices(6000,8)).issubset(ix))
 def test_deterministic_replay(self):self.assertEqual(self.go([]),self.go([]))
 def test_bad_numeric_signal_stops(self):
  self.features[0,0]=np.nan
  with self.assertRaises(ValueError):self.go([])

class Reports(unittest.TestCase):
 def report(self):return dict(facts=[dict(frame=100,text='visible event')],met=[dict(condition='target_identity',frames=[100])],missing=['timing unknown'],targets=['visible object'],claim='partial')
 def test_valid_json_and_fence(self):
  raw=json.dumps(self.report());self.assertIsNotNone(k.parse_report(raw,[100],['target_identity'])[0]);self.assertIsNotNone(k.parse_report('```json\n'+raw+'\n```',[100],['target_identity'])[0])
 def test_unseen_fact_ref_rejected(self):self.assertIsNone(k.parse_report(json.dumps(self.report()),[101],['target_identity'])[0])
 def test_unseen_condition_ref_rejected(self):
  x=self.report();x['met'][0]['frames']=[999];self.assertIsNone(k.parse_report(json.dumps(x),[100],['target_identity'])[0])
 def test_cannot_claim_supported_with_missing_conditions(self):
  x=self.report();x['claim']='supported';x['missing']=[];self.assertIsNone(k.parse_report(json.dumps(x),[100],['target_identity','first_event'])[0])
 def test_temporal_guard_rejects_late_first(self):
  x=self.report();x['met']=[dict(condition='first_event',frames=[10000])]
  g=k.guard(x,['first_event'],[dict(id=0,origin='ordered_prefix',indices=[0,1,2,3])],30,20000);self.assertTrue(g['issues']);self.assertFalse(g['machine_verified'])
 def test_final_guard_rejects_middle(self):
  x=self.report();x['met']=[dict(condition='final_event',frames=[100])];self.assertTrue(k.guard(x,['final_event'],[],30,20000)['issues'])
 def test_order_requires_all_four_event_windows(self):
  x=self.report();x['met']=[dict(condition='temporal_order',frames=[100,200])]
  w=[dict(id=0,indices=[100]),dict(id=1,indices=[200])];self.assertTrue(k.guard(x,['temporal_order'],w,30,20000,4)['issues'])
 def test_valid_local_claim_is_still_not_semantic_certification(self):
  x=self.report();x['claim']='supported';x['missing']=[];g=k.guard(x,['target_identity'],[],30,20000);self.assertEqual(g['status'],'model_supported_unverified');self.assertFalse(g['machine_verified'])
 def test_bad_format_retained_as_failure(self):self.assertIsNone(k.parse_report('Score: 5',[1],['target_identity'])[0])

class Guards(unittest.TestCase):
 def test_incomplete_eval_cannot_read_human_refs(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('manual reference read')):
   with self.assertRaises(ValueError):ev.summarize(dict(units={}))
 def test_run_needs_user_flag(self):
  with patch.object(r,'context',return_value=dict(units={})),patch.object(p,'disk_guard',return_value={}),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}),patch('sys.argv',['condition_clips_run.py','run']):
   with self.assertRaises(RuntimeError):r.main()
 def test_cached_call_identity_and_raw_parse(self):
  c={'signature':'sig'};obj=dict(signature='sig',index='1',task='global',prompt_sha256=p.digest('prompt'),indices=[100],generated_tokens=10,tensor_sha256='a'*64,raw_text='bad',report=None,parse_error=k.parse_report('bad',[100],['target_identity'])[1],inference_s=1.,preprocess_s=1.,peak_gib=10.)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'call.json';p.write_once(path,p.seal(obj));self.assertIsNotNone(r.load_call(path,c,'1','global','prompt',[100],['target_identity']))
   with self.assertRaises(ValueError):r.load_call(path,c,'1','global','changed',[100],['target_identity'])

if __name__=='__main__':unittest.main()
