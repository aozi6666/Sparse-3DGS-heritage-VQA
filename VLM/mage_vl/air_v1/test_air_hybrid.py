"""只测固定候选的关键契约；临时夹具不进入实验数据。"""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import air_hybrid as h
from air_protocol import seal,digest

class Selection(unittest.TestCase):
 def setUp(self):self.base=[i*100 for i in range(32)]
 def choose(self,scores,sim=None):return h.choose(self.base,scores,sim or {i:0.5 for i in scores},30.)
 def test_no_positive_is_exact_uniform(self):
  v=self.choose({10:None,30:3,40:2});self.assertEqual(v['indices'],self.base);self.assertEqual(len(v['anchors']),16);self.assertEqual(v['local'],[])
 def test_single_positive_never_collapses_video(self):
  v=self.choose({15:4});self.assertEqual(len(v['indices']),32);self.assertIn(15,v['indices']);self.assertTrue(set(v['anchors']).issubset(v['indices']))
 def test_max_local_and_no_duplicates(self):
  v=self.choose({i:5 for i in range(1,3000,7)});self.assertEqual(len(v['local']),16);self.assertEqual(len(set(v['indices'])),32);self.assertEqual(v['indices'],sorted(v['indices']))
 def test_anchor_overlap_not_counted_as_new_evidence(self):
  anchors=self.choose({})['anchors'];v=self.choose({i:5 for i in anchors});self.assertEqual(v['local'],[]);self.assertEqual(v['indices'],self.base)
 def test_round_robin_does_not_exhaust_first_cluster(self):
  v=self.choose({1:5,2:5,3:5,1001:4,1002:4});self.assertEqual(v['local'][:2],[1,1001]);self.assertEqual(v['local'][2],3)
 def test_deterministic_order_independent_of_dict_insertion(self):
  scores={i:4+i%2 for i in range(1,2900,29)};self.assertEqual(self.choose(scores),self.choose(dict(reversed(list(scores.items())))))
 def test_invalid_never_promoted(self):
  v=self.choose({11:None,12:3,13:1,14:4});self.assertEqual(v['local'],[14])
 def test_exact_two_second_cluster_boundary(self):
  v=self.choose({1:5,61:4,122:4});self.assertEqual(v['clusters'],[[1,61],[122]])
 def test_bad_baseline_and_fps(self):
  for base,fps in [(self.base[:-1],30.),(list(reversed(self.base)),30.),(self.base,0.)]:
   with self.assertRaises(ValueError):h.choose(base,{}, {},fps)

class Records(unittest.TestCase):
 def build(self):
  plan={'indices':list(range(32)),'reuse_arm':'uniform32'};old={'record_sha256':'old','raw_text':'B'}
  c={'signature':'sig','plans':{'1':plan},'source':{'units':{'1:uniform32':old}}}
  u=dict(signature='sig',index='1',plan_sha256=digest(plan),indices=plan['indices'],pred='B',raw_text='B',native_equal=True,tensor_sha256='a'*64,input_tokens=200,visual_tokens=100,inference_s=0.,preprocess_s=1.,peak_gib=0.,origin='reused_uniform32',source_record='old',new_calls=0,generated_tokens=0)
  return c,seal(u)
 def test_reused_record_and_corruption(self):
  c,u=self.build();h.validate_record(c,u,{})
  with self.assertRaises(ValueError):h.validate_record(c,dict(u,raw_text='A'),{})
 def test_duplicate_and_wrong_source_rejected(self):
  c,u=self.build()
  with self.assertRaises(ValueError):h.validate_record(c,u,{'1':u})
  u.pop('record_sha256');u['source_record']='wrong'
  with self.assertRaises(ValueError):h.validate_record(c,seal(u),{})
 def test_partial_summary_does_not_read_labels(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('GT read too early')):
   with self.assertRaises(ValueError):h.summarize({'units':{}})
 def test_tensor_hash_detects_pixel_time_and_dtype_changes(self):
  import torch
  a={'pixel_values':torch.ones(2,3),'patch_positions':torch.zeros(2,3,dtype=torch.int64)};ha=h.tensor_sha(a)
  self.assertEqual(ha,h.tensor_sha({k:v.clone() for k,v in a.items()}))
  for k in a:
   b={n:v.clone() for n,v in a.items()};b[k][0,0]+=1;self.assertNotEqual(ha,h.tensor_sha(b))

if __name__=='__main__':unittest.main()
