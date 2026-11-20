import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import condition_clips_qa30 as q
import air_protocol as p

class Contracts(unittest.TestCase):
 def fixture(self):
  row=dict(question='Which?',bodies=['one','two','three','four'])
  base=dict(pred='A',raw_text='A',record_sha256='old',indices=[0,1])
  c=dict(signature='sig',rows={'1':row},bases={'1':base},plans={'1':dict(indices=[0,1])})
  u=dict(signature='sig',index='1',indices=[0,1],baseline_record_sha='old',baseline_pred='A',raw_text='A',pred='A',prompt_sha=p.digest(p.prompt(row)),native_equal=True,tensor_sha='a'*64,baseline_tensor_sha='b'*64,new_calls=0,generated_tokens=0,inference_s=0.,preprocess_s=1.,peak_gib=0.)
  return c,u
 def test_reuse_valid(self):
  c,u=self.fixture();q.validate(c,p.seal(u),{})
 def test_changed_frame_cannot_reuse(self):
  c,u=self.fixture();c['plans']['1']['indices']=[0,2];u['indices']=[0,2]
  with self.assertRaises(ValueError):q.validate(c,p.seal(u),{})
 def test_wrong_source_rejected(self):
  c,u=self.fixture();u['baseline_record_sha']='wrong'
  with self.assertRaises(ValueError):q.validate(c,p.seal(u),{})
 def test_duplicate_stops(self):
  c,u=self.fixture()
  with self.assertRaises(ValueError):q.validate(c,p.seal(u),{'1':u})
 def test_qa_prompt_only_original(self):
  c,u=self.fixture();self.assertEqual(p.prompt(c['rows']['1']),p.ANSWER_HEADER+'Which?\nA.one\nB.two\nC.three\nD.four')
 def test_no_gt_before_completion(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('no read')):
   with self.assertRaises(ValueError):q.summarize(dict(units={}))
 def test_nan_cost_rejected(self):
  c,u=self.fixture();u['preprocess_s']=float('nan')
  with self.assertRaises(ValueError):q.validate(c,p.seal(u),{})
if __name__=='__main__':unittest.main()
