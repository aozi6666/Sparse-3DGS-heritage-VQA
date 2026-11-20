"""契约测试，不加载模型权重，不生成评测答案。"""
import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import condition_clips_v2 as k
import condition_clips_v2_run as r
import condition_clips_v2_eval as ev
import air_protocol as p

class Tokenizer:
    eos_token_id=99
    all_special_ids=[99]
    def encode(self,ch,add_special_tokens=False):return [ord(ch)]
    def decode(self,ids):return ''.join(chr(i) for i in ids)

class Codes(unittest.TestCase):
 def setUp(self):self.a,self.e=k.atom_tokens(Tokenizer());self.cs=['target_identity','first_event']
 def encode(self,s):return [self.a[ch] for ch in s]+[self.e]
 def test_unknown_is_not_sufficient(self):
  d=k.parse_codes(self.encode('UNUN'),self.cs,['G','0'],self.a,self.e);self.assertEqual(d[0]['state'],'unknown');self.assertIsNone(d[1]['unit'])
 def test_supported_is_unverified(self):
  d=k.parse_codes(self.encode('E0PG'),self.cs,['G','0'],self.a,self.e);self.assertTrue(all(x['semantically_verified'] is False for x in d))
 def test_joint_unit(self):self.assertEqual(k.parse_codes(self.encode('EJUN'),self.cs,['G','J'],self.a,self.e)[0]['unit'],'J')
 def test_unseen_ref_rejected(self):
  with self.assertRaises(ValueError):k.parse_codes(self.encode('E5UN'),self.cs,['G','0'],self.a,self.e)
 def test_unknown_cannot_cite(self):
  with self.assertRaises(ValueError):k.parse_codes(self.encode('U0UN'),self.cs,['G','0'],self.a,self.e)
 def test_missing_eos_rejected(self):
  with self.assertRaises(ValueError):k.parse_codes(self.encode('UNUN')[:-1],self.cs,['G'],self.a,self.e)
 def test_length_not_relaxed(self):
  with self.assertRaises(ValueError):k.parse_codes(self.encode('UN'),self.cs,['G'],self.a,self.e)
 def test_arbitrary_status_rejected(self):
  ids=self.encode('UNUN');ids[0]=10000
  with self.assertRaises(ValueError):k.parse_codes(ids,self.cs,['G'],self.a,self.e)
 def test_every_masked_path_parses(self):
  prefixes=[[]]
  for step in range(5):prefixes=[pre+[i] for pre in prefixes for i in k.allowed_next(pre,2,['G','0'],self.a,self.e)]
  for ids in prefixes:k.parse_codes(ids,self.cs,['G','0'],self.a,self.e)
  self.assertEqual(len(prefixes),25)
 def test_no_early_eos(self):
  self.assertNotIn(self.e,k.allowed_next([],2,['G'],self.a,self.e));self.assertEqual(k.allowed_next([self.a['U']],2,['G'],self.a,self.e),[self.a['N']])
 def test_bad_atom_stops(self):
  t=Tokenizer();t.encode=lambda ch,**kw:[1,2]
  with self.assertRaises(ValueError):k.atom_tokens(t)

class Evidence(unittest.TestCase):
 def setUp(self):self.w=[dict(id=0,start=0,end=12,indices=[0,100,200,300]),dict(id=1,start=12,end=24,indices=[300,400,500,600])]
 def test_time_mapping_is_program_owned(self):
  u=k.evidence_units([0,900],self.w,25);self.assertEqual(u['0']['seconds'],[0,4,8,12]);self.assertEqual(u['0']['frames'],[0,100,200,300])
 def test_joint_union(self):
  u=k.evidence_units([0,900],self.w,25);self.assertEqual(u['J']['frames'],[0,100,200,300,400,500,600,900])
 def test_global_never_continuous(self):self.assertIn('NOT continuous',k.evidence_units([0,900],[],25)['G']['role'])
 def test_first_global_claim_flagged(self):
  c=[dict(condition='first_event',state='model_claimed_sufficient',unit='G')];self.assertTrue(k.structural_issues(c,k.evidence_units([0,900],self.w,25),self.w,25,10000))
 def test_partial_not_misreported_sufficient(self):
  c=[dict(condition='first_event',state='partial',unit='G')];self.assertEqual(k.structural_issues(c,{},[],25,10000),[])
 def test_original_options_and_meaning_kept(self):
  row=dict(question='Which is first?',bodies=['one','two','three','four'])
  s=k.prompt(row,['first_event'],{'G':{}},'codes',observation='Fact')
  self.assertIn('D.four',s);self.assertIn('earlier relevant events',s);self.assertIn('U=unknown',s)
 def test_caption_not_json(self):
  s=k.prompt(dict(question='Q?',bodies=['a','b','c','d']),['target_identity'],{'G':{}},'observation');self.assertIn('No JSON',s)

class Gates(unittest.TestCase):
 def units(self):return {i:dict(observation_flags=[k.observation_flags('A visible object.',False) for _ in range(8)]) for i in k.SMOKE}
 def test_gate_does_not_certify_semantics(self):self.assertIn('NOT semantic',k.smoke_gate(self.units())['scope'])
 def test_empty_stops_expansion(self):
  u=self.units();u[k.SMOKE[0]]['observation_flags'][0]['empty']=True
  with self.assertRaises(ValueError):k.smoke_gate(u)
 def test_repetition_detected(self):self.assertTrue(k.observation_flags(('The same object is visible here. '*5),True)['repetition_loop'])
 def test_cap_rate_stops_expansion(self):
  u=self.units()
  for f in u[k.SMOKE[0]]['observation_flags']:f['at_token_cap']=True
  with self.assertRaises(ValueError):k.smoke_gate(u)
 def test_offline_reference_not_read_before_completion(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('should not read')):
   with self.assertRaises(ValueError):ev.summarize(dict(protocol=dict(indices=['379']),units={}), 'diag30')
 def test_read_call_identity_stops(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'call.json';p.write_once(path,p.seal(dict(signature='wrong',index='1',task='global_codes',prompt_sha='x',indices=[0],evidence_units={})))
   with self.assertRaises(ValueError):r.read_call(path,dict(signature='right'),'1','global_codes','p',[0],{},[],{},99)
 def test_immutable_results_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'x.json';p.write_once(path,dict(x=1))
   with self.assertRaises(ValueError):p.write_once(path,dict(x=2))
   self.assertEqual(json.loads(path.read_text()),dict(x=1))

if __name__=='__main__':unittest.main()
