"""最终评测的CPU检查：配对恢复、缺字幕回退、全量拒绝与完整后汇总。"""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import unittest
import continuous_test225 as t
from practical_common import signature,write_once,sha
from subtitle_common import sealed

def fixture(index='1'):
    r=dict(index=index,video_id='v',question='Q?',bodies=['a','b','c','d'],duration='medium',task_type='action')
    a=dict(text='original',text_tokens=1,frame_indices=list(range(32)))
    return r,dict(aligned=a,continuous=dict(text='continuous',text_tokens=2))

def record(r,e,arm='no_sub',raw='A'):
    return sealed(dict(signature='sig',index=r['index'],video_id=r['video_id'],arm=arm,origin='generated',raw_text=raw,pred=raw,
        prompt_sha256=signature(t.query(r,arm,e)),evidence_sha256=signature(e),subtitle_tokens=0 if arm=='no_sub' else e[arm]['text_tokens'],
        input_sha256='a'*64,input_tokens=5000,visual=dict(grid_thw=[[1,24,24]]*32,n_tok=4608,patch_time_range=[0.,31.],pv_shape=[18432,768]),
        checks=dict(native_input_equal=True),new_answer_calls=1,generated_tokens=1,reuse_proof=None,inference_s=1.,preprocess_s=1.,peak_allocated_gib=10.))

def alter(u,**kw):
    x=deepcopy(u);x.pop('record_sha256');x.update(kw);return sealed(x)

class Records(unittest.TestCase):
    def setUp(self):self.r,self.e=fixture();self.ctx=dict(signature='sig');self.b=record(self.r,self.e)
    def check(self,u,existing=None):return t.validate_record(self.ctx,self.r,self.e,u,existing or {})
    def test_fresh_three_arms(self):
        units={}
        for arm in t.ARMS:
            u=record(self.r,self.e,arm);k=self.check(u,units);units[k]=u
        self.assertEqual(len(units),3)
    def test_no_historical_reuse(self):
        with self.assertRaises(ValueError):self.check(alter(self.b,origin='reused_old',new_answer_calls=0))
    def test_no_missing_baseline(self):
        with self.assertRaises(ValueError):self.check(record(self.r,self.e,'aligned'))
    def test_duplicate_and_corruption(self):
        with self.assertRaises(ValueError):self.check(self.b,{'1:no_sub':self.b})
        x=dict(self.b,pred='D')
        with self.assertRaises(ValueError):self.check(x)
    def test_input_evidence_and_native_checks(self):
        for kw in [dict(prompt_sha256='x'),dict(evidence_sha256='x'),dict(checks=dict(native_input_equal=False)),dict(input_tokens=1),dict(preprocess_s=float('nan'))]:
            with self.assertRaises(ValueError):self.check(alter(self.b,**kw))
    def test_visual_drift(self):
        v=dict(self.b['visual'],n_tok=4607)
        with self.assertRaises(ValueError):self.check(alter(record(self.r,self.e,'continuous'),visual=v),{'1:no_sub':self.b})
    def test_invalid_answer_retained(self):
        self.check(alter(self.b,pred='',raw_text=''))
    def test_missing_subtitle_fallback(self):
        self.e['continuous']=dict(text='',text_tokens=0)
        u=record(self.r,self.e,'continuous')
        u=alter(u,origin='reused_current_no_sub',new_answer_calls=0,generated_tokens=None,inference_s=0.,peak_allocated_gib=0.,
            reuse_proof=dict(no_sub_record_sha256=self.b['record_sha256'],tensor_equal=True))
        self.check(u,{'1:no_sub':self.b})
        with self.assertRaises(ValueError):self.check(alter(u,raw_text='D',pred='D'),{'1:no_sub':self.b})

class Guards(unittest.TestCase):
    def test_partition(self):
        rows=[dict(index=str(i),video_id=str(i//3),duration='medium') for i in range(225)]
        with patch.object(t,'stage_rows',return_value=[]):t.check_partition(rows)
        with patch.object(t,'stage_rows',return_value=[rows[0]]):
            with self.assertRaises(ValueError):t.check_partition(rows)
        with self.assertRaises(ValueError):t.check_partition(rows[:-1])
    def test_cli_scope_rejected(self):
        for option in ['--stage','--scope']:
            r=subprocess.run([sys.executable,'-B',t.__file__,'init',option,'full'],capture_output=True)
            self.assertEqual(r.returncode,2)
    def test_no_gpu_imports(self):
        r=subprocess.run([sys.executable,'-B','-c',"import continuous_test225,sys;assert not any(n in sys.modules for n in ['torch','transformers','decord'])"],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_run_requires_flag(self):
        with patch.object(t,'preflight'),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}):
            with self.assertRaises(RuntimeError):t.run(None,{})
    def test_gpu_cli_arguments(self):
        tree=ast.parse(inspect.getsource(t.run_gpu))
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args'}
        self.assertLessEqual(attrs,{'min_free','budget'})
    def test_partial_summary_never_reads_labels(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read labels')):
            with self.assertRaises(ValueError):t.summarize(dict(expected={'x'},units={}))
    def test_full_paired_summary_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifests=root/'01_manifests';labels={str(i):'A' for i in range(225)}
            write_once(manifests/'evaluation_labels.json',labels)
            write_once(manifests/'dataset_lock.json',dict(labels_sha256=sha(manifests/'evaluation_labels.json')))
            rows=[];units={};evidence={}
            for i in range(225):
                r,e=fixture(str(i));r['video_id']=str(i//3);r['duration']='medium' if i<120 else 'long'
                e['aligned']['status']='used';e['continuous']['omitted_neighbor_ids']=[]
                rows.append(r);evidence[r['video_id']]=e
                for arm in t.ARMS:
                    raw='B' if i<30 else 'A'
                    if arm=='continuous' and i<10:raw='A'
                    if arm=='continuous' and 30<=i<35:raw='B'
                    units[t.key(r['index'],arm)]=record(r,e,arm,raw)
            store=root/'cache';write_once(store/'units.jsonl',{})
            ctx=dict(expected=set(units),units=units,rows=rows,signature='sig',tokens=2048,protocol=dict(policy='test'),stage_dir=root/'stage',store=store)
            with patch.object(t,'ROOT',root),patch.object(t,'load_alignment',side_effect=lambda c,v:evidence[v]),patch('builtins.print'):
                t.summarize(ctx)
            s=json.loads((root/'stage/summary.json').read_text())
            self.assertTrue(s['audit']['passed']);self.assertEqual(s['N'],225)
            self.assertEqual(s['overall']['methods']['continuous']['correct'],200)
            self.assertEqual(s['overall']['methods']['continuous']['vs']['no_sub']['net'],5)
            self.assertTrue((root/'07_archive/continuous_test225_fixed2048_v1.json').exists())

if __name__=='__main__':unittest.main()
