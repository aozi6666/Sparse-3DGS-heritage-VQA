"""条件编译、时序与预算契约；临时测试输入不作为评测数据。"""
import json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import condition_grounding as k
import condition_grounding_run as r
import condition_grounding_qa as q

class Parse(unittest.TestCase):
 def row(self,s):return dict(question=s,bodies=['one','two','three','four'])
 def obj(self,question='A man is interviewed.'):
  return dict(evidence='visual',events=[dict(target='man',action='being interviewed',question_quote=question)],relation='independent',time_range=None,time_quote='',unresolved=[])
 def go(self,d,s):return k.parse(json.dumps(d),self.row(s))
 def test_good_parse(self):
  s='A man is interviewed.';self.assertIsNotNone(self.go(self.obj(s),s)[0])
 def test_first_six_minutes_numeric(self):
  s='Why in the first six minutes?';d=self.obj(s);d.update(evidence='mixed',time_range=[0,360],time_quote='first six minutes')
  self.assertIsNotNone(self.go(d,s)[0]);self.assertEqual(k.explicit_range(s)['bounds'],[0,360])
 def test_numeric_range_not_first_event(self):
  s='Why in the first six minutes?';d=self.obj(s);d.update(time_range=[0,360],time_quote='first six minutes',relation='first_event');self.assertIsNone(self.go(d,s)[0])
 def test_missing_range_rejected(self):
  s='In the first six minutes what happens?';self.assertIsNone(self.go(self.obj(s),s)[0])
 def test_invented_24_seconds_rejected(self):
  s='Which tool in the first magic?';d=self.obj(s);d.update(relation='first_event',time_range=[0,24],time_quote='first magic');self.assertIsNone(self.go(d,s)[0])
 def test_first_magic_explicit_unsupported(self):
  s='Which tool in the first magic?';d=self.obj(s);d['relation']='first_event';parsed,error=self.go(d,s);self.assertIsNotNone(parsed);self.assertEqual(k.fallback_reason(parsed),'unsupported_relation_first_event')
 def test_unknown_order_no_imposed_sequence(self):
  s='In which order do these happen?';d=self.obj(s);d['relation']='known_sequence';self.assertIsNone(self.go(d,s)[0])
 def test_known_order_needs_two_events(self):
  s='What happens after the man stands?';d=self.obj(s);d['relation']='known_sequence';self.assertIsNone(self.go(d,s)[0])
 def test_no_option_as_question_evidence(self):
  s='What is visible?';d=self.obj('one');self.assertIsNone(self.go(d,s)[0])
 def test_spatial_not_temporal(self):
  s='In what order in a single scene?';d=self.obj(s);d['relation']='unknown_order';self.assertIsNone(self.go(d,s)[0]);d['relation']='spatial_order';self.assertIsNotNone(self.go(d,s)[0])
 def test_all_explicit_items_kept(self):
  s='In which order?\n(a) cup\n(b) plate\n(c) straw\n(d) snack';d=self.obj(s);d['relation']='unknown_order';self.assertIsNone(self.go(d,s)[0])
 def test_all_explicit_items_valid_without_order_assignment(self):
  s='In which order?\n(a) cup\n(b) plate\n(c) straw\n(d) snack';d=self.obj('cup');d['relation']='unknown_order';d['events']=[dict(target=x,action='',question_quote=x) for x in ['cup','plate','straw','snack']];self.assertIsNotNone(self.go(d,s)[0])
 def test_all_options_in_prompt(self):
  text=k.parser_prompt(self.row('Q?'));self.assertIn('D.four',text);self.assertIn('never known_sequence',text)
 def test_speech_not_turned_visual(self):self.assertEqual(k.fallback_reason(dict(evidence='speech',relation='independent')),'unsupported_evidence_speech')
 def test_no_labels_in_parse_summary(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('must not read')):
   with self.assertRaises(ValueError):r.ready(dict(parses={}),['1'])
 def test_unparseable_is_not_neutral(self):self.assertIsNone(k.parse('C',self.row('Q?'))[0])
 def test_number_ranges(self):
  self.assertEqual(k.explicit_range('From 2 to 5 minutes')['bounds'],[120,300]);self.assertIsNone(k.explicit_range('At the beginning'))

class Temporal(unittest.TestCase):
 def test_seq_requires_strict_before(self):
  a=np.array([[1.,0.],[0.,0.],[0.,1.]])
  x=k.temporal_gates(a,np.array([0.,2.,9.]),'known_sequence');self.assertEqual(x[0,0],1);self.assertEqual(x[2,1],1)
 def test_reversed_order_suppressed(self):
  a=np.array([[0.,1.],[0.,0.],[1.,0.]])
  self.assertEqual(float(k.temporal_gates(a,np.array([0.,2.,9.]),'known_sequence').max()),0)
 def test_unknown_order_not_gated(self):
  a=np.array([[0.,1.],[1.,0.]])
  np.testing.assert_array_equal(a,k.temporal_gates(a,np.array([0.,9.]),'unknown_order'))
 def test_duplicate_time_rejected(self):
  with self.assertRaises(ValueError):k.temporal_gates(np.ones((2,1)),np.array([0.,0.]),'independent')

class Selection(unittest.TestCase):
 def setUp(self):
  self.ids=list(range(0,15000,25));self.f=np.random.default_rng(3).normal(size=(len(self.ids),768)).astype('float32');self.v=np.random.default_rng(4).normal(size=(3,768)).astype('float32')
  self.d=dict(evidence='visual',relation='unknown_order',events=[{} for _ in range(3)],time_range=None)
 def go(self):return k.select_windows(self.ids,self.f,self.v,25.,15000,self.d)
 def test_budget_and_leaf_assignment(self):
  out=self.go();self.assertLessEqual(len(out['indices']),32);self.assertEqual(len(out['windows']),6);self.assertEqual(set(out['assigned_leaf_windows']),{'0','1','2'});self.assertFalse(out['condition_satisfied'])
 def test_nonoverlapping(self):
  ws=self.go()['windows']
  for j,a in enumerate(ws):
   for b in ws[j+1:]:self.assertLessEqual(min(a['end'],b['end'])-max(a['start'],b['start']),1e-6)
 def test_explicit_range_on_windows_not_global_anchors(self):
  self.d['time_range']=[0,360];out=self.go();self.assertTrue(all(0<=w['start']<w['end']<=360 for w in out['windows']));self.assertGreater(out['indices'][-1]/25,360)
 def test_short_range_stops_without_expansion(self):
  self.d['time_range']=[0,24]
  with self.assertRaises(ValueError):self.go()
 def test_flat_leaf_stops(self):
  self.f[:]=0
  with self.assertRaises(ValueError):self.go()
 def test_known_predecessor_outside_range_not_counted(self):
  self.f[:]=0;self.f[500,0]=1;self.f[100,1]=1;self.v=np.zeros((2,768),dtype='float32');self.v[0,0]=1;self.v[1,1]=1;self.d.update(relation='known_sequence',events=[{},{}],time_range=[0,360])
  with self.assertRaises(ValueError):self.go()
 def test_deterministic(self):self.assertEqual(self.go(),self.go())
 def test_scope_restored_on_failure(self):
  before=q.qa.OUT;fn=q.qa.validate
  with self.assertRaises(RuntimeError):
   with q.scope():raise RuntimeError()
  self.assertEqual(q.qa.OUT,before);self.assertIs(q.qa.validate,fn)
 def test_labels_not_read_before_QA_complete(self):
  with patch.object(q,'context',return_value=dict(units={})),patch.object(Path,'read_text',side_effect=AssertionError('no labels')):
   with self.assertRaises(ValueError):q.summarize({})

if __name__=='__main__':unittest.main()
