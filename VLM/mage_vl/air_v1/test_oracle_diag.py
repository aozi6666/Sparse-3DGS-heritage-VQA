"""临时单元夹具只验证工程契约，不作为评测数据。"""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import oracle_diag as d
from air_protocol import seal,digest,sha,write_once,append,read_journal

class Inputs(unittest.TestCase):
 def sample(self):return dict(version='oracle_evidence_diag_v1',rows=[dict(index=i,frame_indices=list(range(32))) for i in d.IDS])
 def test_fixed12_and_32frames(self):self.assertEqual(d.validate_inputs(self.sample())['rows'][0]['index'],'379')
 def test_answer_explanation_and_subtitle_rejected(self):
  for key in ['gt','answer','explanation','subtitle']:
   obj=self.sample();obj['rows'][0][key]='DO NOT SEND'
   with self.assertRaises(ValueError):d.validate_inputs(obj)
 def test_duplicate_unsorted_and_wrong_count_rejected(self):
  for ix in [list(range(31)),list(reversed(range(32))),[0]*32]:
   obj=self.sample();obj['rows'][0]['frame_indices']=ix
   with self.assertRaises(ValueError):d.validate_inputs(obj)
 def test_changing_question_set_rejected(self):
  obj=self.sample();obj['rows'][0]['index']='1'
  with self.assertRaises(ValueError):d.validate_inputs(obj)
 def test_original_prompt_ignores_review_and_label_fields(self):
  row=dict(question='Original question?',bodies=[' one',' two',' three',' four'])
  expected=d.p.prompt(row);row.update(gt='A',evidence_note='SECRET EXPLANATION',review={'status':'explicit'})
  self.assertEqual(expected,d.p.prompt(row));self.assertNotIn('SECRET',expected)

class Records(unittest.TestCase):
 def build(self):
  idx=d.IDS[0];row=dict(question='Q?',bodies=[' a',' b',' c',' d'])
  base=dict(visual=dict(n_tok=100),raw_text='B',record_sha256='source',indices=list(range(32)))
  c=dict(signature='sig',indices={idx:list(range(32))},rows={idx:row},baselines={idx:{'uniform32':base}})
  u=dict(signature='sig',index=idx,indices=list(range(32)),prompt_sha256=digest(d.p.prompt(row)),raw_text='B',pred='B',native_equal=True,tensor_sha256='a'*64,visual_tokens=100,input_tokens=200,reuse_arm='uniform32',source_record='source',new_calls=0,generated_tokens=0,inference_s=0.,preprocess_s=1.,peak_gib=0.)
  return c,seal(u)
 def test_valid_reuse_and_resume(self):
  c,u=self.build();d.validate_unit(c,u,{})
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'units.jsonl';append(p,u);v=read_journal(p)[0];d.validate_unit(c,v,{})
 def test_corruption_duplicate_and_wrong_prompt_stop(self):
  c,u=self.build()
  with self.assertRaises(ValueError):d.validate_unit(c,dict(u,raw_text='A'),{})
  with self.assertRaises(ValueError):d.validate_unit(c,u,{u['index']:u})
  u.pop('record_sha256');u['prompt_sha256']='changed'
  with self.assertRaises(ValueError):d.validate_unit(c,seal(u),{})
 def test_changed_visual_budget_stops(self):
  c,u=self.build();u.pop('record_sha256');u['visual_tokens']=99
  with self.assertRaises(ValueError):d.validate_unit(c,seal(u),{})
 def test_partial_does_not_read_GT(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('GT read')):
   with self.assertRaises(ValueError):d.summarize(dict(units={}))
 def test_run_requires_user_flag(self):
  with patch.object(d,'context',return_value=dict(reviews={},units={})),patch.object(d.p,'disk_guard',return_value={}),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}),patch('sys.argv',['oracle_diag.py','run']):
   with self.assertRaises(RuntimeError):d.main()

class Summary(unittest.TestCase):
 def test_complete_summary_separates_evidence_quality(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);old=root/'old';out=root/'diag';lp=old/'01_manifests/evaluation_labels.json'
   write_once(lp,{i:'A' for i in d.IDS});write_once(lp.parent/'dataset_lock.json',dict(labels_sha256=sha(lp)))
   reviews={i:dict(status='explicit',group=g) for g,ids in d.GROUPS.items() for i in ids}
   reviews['626']['status']='uncertain';reviews['486']['status']='uncertain';reviews['704']['status']='visual_insufficient'
   units={i:dict(pred='A',new_calls=1,inference_s=.1,preprocess_s=.2,visual_tokens=100) for i in d.IDS}
   c=dict(units=units,signature='test',reviews=reviews,baselines={i:{a:dict(pred='B') for a in d.ARMS[:3]} for i in d.IDS})
   with patch.object(d,'OUT',out),patch.object(d.p,'OLD',old),patch('builtins.print'):d.summarize(c)
   s=json.loads((out/'results/summary.json').read_text())
   self.assertEqual(s['explicit_evidence']['N'],9);self.assertEqual(s['uncertain_evidence']['N'],2);self.assertEqual(s['visual_insufficient']['N'],1)
   self.assertEqual(s['by_group']['all_three_wrong']['N'],4);self.assertEqual(s['overall']['methods']['evidence32']['correct'],12)

if __name__=='__main__':unittest.main()
