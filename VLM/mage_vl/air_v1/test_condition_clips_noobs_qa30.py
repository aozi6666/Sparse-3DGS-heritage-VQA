import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import air_protocol as p
import condition_clips as old
import condition_clips_noobs_qa30 as n

class Ablation(unittest.TestCase):
 def setUp(self):
  self.row=dict(question='Which event?',bodies=['one','two','three','four']);self.m=dict(fps=25.,frame_count=10000)
  self.ids=list(range(0,10000,25));self.f=np.random.default_rng(5).normal(size=(len(self.ids),768)).astype('float32');self.v=np.ones(768,dtype='float32').tolist()
 def test_observation_not_used(self):
  tv=dict(texts=[p.query(self.row),'bad automatic caption'],vectors=[self.v,self.v]);a=n.plan_one(self.row,self.m,self.ids,self.f,tv)
  tv['texts'][1]='CHANGED';tv['vectors'][1]=(-np.ones(768)).tolist();b=n.plan_one(self.row,self.m,self.ids,self.f,tv)
  self.assertEqual(a,b);self.assertEqual(a['queries'],[p.query(self.row)])
 def test_explicit_items_all_kept(self):
  row=dict(self.row,question='Order?\n(a) cup\n(b) plate\n(c) straw\n(d) snacks')
  texts=old.query_texts(p.query(row),row['question'],None);a=n.plan_one(row,self.m,self.ids,self.f,dict(texts=texts,vectors=[self.v]*5))
  self.assertEqual(len(a['queries']),5);self.assertLessEqual(len(a['indices']),32)
 def test_missing_vector_stops_without_encoding(self):
  with self.assertRaises(ValueError):n.plan_one(self.row,self.m,self.ids,self.f,dict(texts=['wrong'],vectors=[self.v]))
 def test_nan_vector_stops(self):
  v=list(self.v);v[0]=float('nan')
  with self.assertRaises(ValueError):n.plan_one(self.row,self.m,self.ids,self.f,dict(texts=[p.query(self.row)],vectors=[v]))
 def test_scope_restored_on_failure(self):
  out,validate=n.qa.OUT,n.qa.validate
  with self.assertRaises(RuntimeError):
   with n.runtime_scope():
    self.assertEqual(n.qa.OUT,n.OUT);raise RuntimeError()
  self.assertEqual(n.qa.OUT,out);self.assertIs(n.qa.validate,validate)
 def test_labels_not_read_early(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('no labels')):
   with self.assertRaises(ValueError):n.summarize(dict(units={}))
 def fixture(self):
  src=dict(index='1',indices=[1,2],raw_text='A',pred='A',prompt_sha='x',tensor_sha='a'*64,baseline_tensor_sha='b'*64,baseline_pred='B',baseline_record_sha='old',input_tokens=99,visual_tokens=32,grid_thw=[[1,2,2]],native_equal=True)
  src=p.seal(src);c=dict(signature='new',rows={'1':{}},plans={'1':dict(reuse_v2=True,indices=[1,2])},source=dict(units={'1':src}))
  u=p.unseal(src);u.update(signature='new',source_reuse_record=src['record_sha256'],new_calls=0,generated_tokens=0,inference_s=0.,preprocess_s=0.,peak_gib=0.);return c,u
 def test_identical_inputs_reuse_source(self):
  c,u=self.fixture();n.validate(c,p.seal(u),{})
 def test_reuse_tensor_drift_stops(self):
  c,u=self.fixture();u['tensor_sha']='x'*64
  with self.assertRaises(ValueError):n.validate(c,p.seal(u),{})
 def test_reuse_changed_frames_stops(self):
  c,u=self.fixture();c['plans']['1']['indices']=[3,4]
  with self.assertRaises(ValueError):n.validate(c,p.seal(u),{})
 def test_reuse_cannot_charge_generation(self):
  c,u=self.fixture();u['new_calls']=1
  with self.assertRaises(ValueError):n.validate(c,p.seal(u),{})
if __name__=='__main__':unittest.main()
