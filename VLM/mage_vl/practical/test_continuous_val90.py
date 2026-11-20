"""CPU模拟检查：禁止补跑基线、恢复身份、缺字幕回退，不加载真实模型。"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import subprocess
import sys
from pathlib import Path
import continuous_val90 as v
from practical_common import signature
from subtitle_common import sealed

class ValRecords(unittest.TestCase):
    def setUp(self):
        self.row=dict(index='1',video_id='video',question='Q',bodies=['a','b','c','d'])
        self.e=dict(aligned=dict(text='original',text_tokens=1),continuous=dict(text='continuous',text_tokens=2))
        self.visual=dict(n_tok=100)
        self.source=dict(raw_text='A',pred='A',input_sha256='a'*64,prompt_sha256=signature(v.query(self.row,'no_sub',self.e)),
            visual=self.visual,signature='old',record_sha256='b'*64,path='/old/units.jsonl')
        self.ctx=dict(signature='new',old=dict(units={'1:'+v.OLD_ARMS[0]:dict(visual=self.visual)}))
    def record(self,arm='no_sub',origin='reused_old'):
        proof=dict(source_signature='old',source_record_sha256='b'*64,source_path='/old/units.jsonl',input_sha256_equal=True,visual_equal=True,tensor_equal=True)
        return sealed(dict(signature='new',index='1',video_id='video',arm=arm,origin=origin,raw_text='A',pred='A',
            input_sha256='a'*64,prompt_sha256=signature(v.query(self.row,arm,self.e)),evidence_sha256=signature(self.e),
            visual=self.visual,input_tokens=200,subtitle_tokens=0 if arm=='no_sub' else 2,
            checks=dict(native_input_equal=True),new_answer_calls=int(origin=='generated'),generated_tokens=1 if origin=='generated' else None,
            reuse_proof=proof if origin=='reused_old' else None,inference_s=0,preprocess_s=1,peak_allocated_gib=0))
    def mutate(self,u,**kw):
        x=deepcopy(u);x.pop('record_sha256');x.update(kw);return sealed(x)
    def validate(self,u,existing=None):
        source=self.source if u['arm']=='no_sub' else None
        with patch.object(v,'old_source',return_value=source):return v.validate_record(self.ctx,self.row,self.e,u,existing or {})
    def test_baseline_reuse(self):self.assertEqual(self.validate(self.record()),'1:no_sub')
    def test_no_baseline_regeneration(self):
        with self.assertRaises(ValueError):self.validate(self.record(origin='generated'))
    def test_reuse_prefix_drift(self):
        with self.assertRaises(ValueError):self.validate(self.mutate(self.record(),input_sha256='c'*64))
    def test_no_changed_reuse_answer(self):
        with self.assertRaises(ValueError):self.validate(self.mutate(self.record(),raw_text='B',pred='B'))
    def test_no_fake_tensor_proof(self):
        u=self.record();u=self.mutate(u,reuse_proof=dict(u['reuse_proof'],tensor_equal=False))
        with self.assertRaises(ValueError):self.validate(u)
    def test_continuous_generation(self):
        base=self.record();self.validate(self.record('continuous','generated'),{'1:no_sub':base})
    def test_missing_baseline(self):
        with self.assertRaises(ValueError):self.validate(self.record('continuous','generated'))
    def test_visual_drift(self):
        with self.assertRaises(ValueError):self.validate(self.mutate(self.record(),visual=dict(n_tok=99)))
    def test_corruption_duplicate_nan(self):
        u=self.record();bad=deepcopy(u);bad['pred']='D'
        for x,existing in [(bad,{}),(u,{'1:no_sub':u}),(self.mutate(u,preprocess_s=float('nan')),{})]:
            with self.assertRaises(ValueError):self.validate(x,existing)
    def test_missing_subtitle_exact_fallback(self):
        self.e['continuous']=dict(text='',text_tokens=0)
        b=self.record();u=self.record('continuous','reused_current_no_sub')
        u=self.mutate(u,subtitle_tokens=0,reuse_proof=dict(no_sub_record_sha256=b['record_sha256'],tensor_equal=True))
        self.validate(u,{'1:no_sub':b})
        with self.assertRaises(ValueError):self.validate(self.mutate(u,raw_text='D',pred='D'),{'1:no_sub':b})

class Guards(unittest.TestCase):
    def test_gpu_cli_attributes(self):
        import ast,inspect
        tree=ast.parse(inspect.getsource(v.run_gpu))
        names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args'}
        self.assertLessEqual(names,{'min_free','budget'})
    def test_no_run_without_user_flag(self):
        with patch.object(v,'preflight'),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}):
            with self.assertRaises(RuntimeError):v.run(None,{})
    def test_incomplete_no_labels(self):
        with self.assertRaises(ValueError):v.summarize(dict(expected={'x'},units={}))
    def test_forbidden_scope(self):
        for arg in ['--stage','--scope']:
            r=subprocess.run([sys.executable,'-B',str(Path(v.__file__)),'init',arg,'test'],capture_output=True)
            self.assertEqual(r.returncode,2)
    def test_import_no_gpu_decode_libs(self):
        r=subprocess.run([sys.executable,'-B','-c',"import continuous_val90,sys;assert not any(n in sys.modules for n in ['torch','transformers','decord'])"],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
