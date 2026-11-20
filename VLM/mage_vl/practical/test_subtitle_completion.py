#!/usr/bin/env python3
"""预算/short入口CPU测试；无真实解码、模型加载或API请求。"""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from practical_common import signature,write_once,append_line
from subtitle_common import sealed
import subtitle_completion_common as c
import subtitle_completion_run as run
import subtitle_completion_analyze as stats


def row(index='1'):
    return dict(index=index,video_id='v',duration='medium',question='Q?',bodies=[' a',' b',' c',' d'])


def evidence(text='hello'):
    original=dict(text='hello',text_tokens=1,frame_indices=list(range(32)))
    return dict(aligned=original,continuous=dict(text=text,text_tokens=1))


def source(r,e):
    return dict(raw_text='A',pred='A',input_sha256='a'*64,prompt_sha256=signature(c.query(r,'continuous',e)),
        visual=dict(grid_thw=[[1,18,32]]*32,patch_time_range=[0.0,31.0],n_tok=4608,pv_shape=[18432,768]),
        signature='old',record_sha256='b'*64,path='/old/units.jsonl')


def record(r,e,origin='reused_old',arm='continuous'):
    old=source(r,e)
    proof=dict(source_signature='old',source_record_sha256='b'*64,source_path='/old/units.jsonl',
        input_sha256_equal=True,visual_equal=True,tensor_equal=True) if origin=='reused_old' else None
    return sealed(dict(signature='new',index=r['index'],video_id=r['video_id'],arm=arm,origin=origin,raw_text='A',pred='A',
        generated_tokens=1 if origin=='generated' else None,new_answer_calls=int(origin=='generated'),reuse_proof=proof,
        prompt_sha256=signature(c.query(r,arm,e)),input_sha256='a'*64,input_tokens=5000,evidence_sha256=signature(e),
        visual=old['visual'],subtitle_tokens=0 if arm=='no_sub' else 1,
        checks=dict(native_input_equal=True,old_2048_input_equal=True,old_2048_input_sha256='a'*64),
        inference_s=1.0 if origin=='generated' else 0.0,preprocess_s=1.0,peak_allocated_gib=10.0 if origin=='generated' else 0.0))


def changed(item,**kw):
    item=deepcopy(item); item.pop('record_sha256'); item.update(kw); return sealed(item)


class Records(unittest.TestCase):
    def setUp(self):
        self.r=row(); self.e=evidence(); self.ctx=dict(scope='budget',signature='new',arms=('continuous',),changed=set())
        self.patch=patch.object(c,'old_source',return_value=source(self.r,self.e)); self.patch.start()
    def tearDown(self): self.patch.stop()
    def test_verified_reuse(self):
        self.assertEqual(c.validate_record(self.ctx,self.r,self.e,record(self.r,self.e),{}),'1:continuous')
    def test_new_generation_only_changed(self):
        self.ctx['changed']={'1'}
        c.validate_record(self.ctx,self.r,self.e,record(self.r,self.e,'generated'),{})
    def test_unchanged_cannot_generate(self):
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,record(self.r,self.e,'generated'),{})
    def test_changed_cannot_reuse(self):
        self.ctx['changed']={'1'}
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,record(self.r,self.e),{})
    def test_prefix_drift_rejected(self):
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,changed(record(self.r,self.e),input_sha256='c'*64),{})
    def test_visual_drift_rejected(self):
        visual=deepcopy(source(self.r,self.e)['visual']); visual['n_tok']=4609
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,changed(record(self.r,self.e),visual=visual),{})
    def test_no_fake_tensor_proof(self):
        item=record(self.r,self.e); proof=dict(item['reuse_proof'],tensor_equal=False)
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,changed(item,reuse_proof=proof),{})
    def test_wrong_old_answer_rejected(self):
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,changed(record(self.r,self.e),raw_text='B',pred='B'),{})
    def test_corruption_and_duplicate(self):
        item=record(self.r,self.e); corrupt=deepcopy(item); corrupt['pred']='D'
        for candidate,existing in [(corrupt,{}),(item,{'1:continuous':item})]:
            with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,candidate,existing)
    def test_cost_nonfinite_rejected(self):
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,changed(record(self.r,self.e),preprocess_s=float('nan')),{})
    def test_short_pair_needs_baseline(self):
        self.ctx.update(scope='short',arms=c.ARMS)
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,self.e,record(self.r,self.e,'generated'),{})
    def test_short_missing_reuses_current_no_sub(self):
        self.ctx.update(scope='short',arms=c.ARMS)
        e=evidence(''); e['aligned']['text']=''; e['aligned']['text_tokens']=0; e['continuous']['text_tokens']=0
        base=record(self.r,e,'generated','no_sub'); existing={'1:no_sub':base}
        item=record(self.r,e,'generated','continuous')
        item=changed(item,origin='reused_current_no_sub',generated_tokens=None,new_answer_calls=0,
            subtitle_tokens=0,inference_s=0.0,peak_allocated_gib=0.0,reuse_proof=dict(no_sub_record_sha256=base['record_sha256'],tensor_equal=True))
        c.validate_record(self.ctx,self.r,e,item,existing)
        with self.assertRaises(ValueError): c.validate_record(self.ctx,self.r,e,changed(item,raw_text='D',pred='D'),existing)


class Guards(unittest.TestCase):
    def test_actual_tensor_fields_all_compared(self):
        fake=SimpleNamespace(equal=lambda a,b:a==b)
        left=dict(input_ids=[1,2],pixel_values=[10],patch_positions=[0])
        self.assertTrue(run.equal_inputs(left,deepcopy(left),fake))
        for field in left:
            right=deepcopy(left); right[field]=[99]
            self.assertFalse(run.equal_inputs(left,right,fake))
        self.assertFalse(run.equal_inputs(left,dict(input_ids=[1,2]),fake))
    def test_user_run_guard(self):
        with patch.object(run,'preflight'),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}):
            with self.assertRaises(RuntimeError): run.run(None,{})
    def test_complete_stage_skips_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'cache').mkdir()
            ctx=dict(signature='s',protocol={},scope='budget',rows=[row()],arms=('continuous',),
                store=root/'cache/s',stage_dir=root/'stage',expected={'1:continuous'},units={'1:continuous':{}})
            args=SimpleNamespace(scope='budget')
            with patch.object(run,'ROOT',root),patch.object(run,'preflight'),patch.object(run,'context',return_value=ctx), \
                    patch.object(run,'run_gpu') as gpu,patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'1'}):
                run.run(args,ctx); gpu.assert_not_called()
    def test_partial_summary_does_not_read_gt(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read GT')):
            with self.assertRaises(ValueError): stats.summarize(None,dict(expected={'x'},units={}))
    def test_no_test_or_full_cli(self):
        for scope in ['test','full','val']:
            with patch.object(sys,'argv',['subtitle_completion_run.py','preflight','--scope',scope]),patch.object(run,'context') as ctx,patch('sys.stderr',new=io.StringIO()):
                with self.assertRaises(SystemExit): run.main()
                ctx.assert_not_called()
    def test_imports_no_gpu_libraries(self):
        code='import subtitle_completion_run,subtitle_completion_analyze,sys; assert "torch" not in sys.modules and "transformers" not in sys.modules and "decord" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,check=True)


class Freeze(unittest.TestCase):
    def test_freeze_once_and_refuse_reselection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); stage=root/'budget3072_dev180'; stage.mkdir()
            write_once(stage/'summary.json',dict(signature='sig',audit=dict(passed=True,origins=dict(generated=24,reused_old=156))))
            write_once(root/'budget3072_candidate.json',{'sig':'sig'}); write_once(root/'short219_sources.json',{'sources':{}})
            ctx=dict(signature='sig',stage_dir=stage,expected={'x'},units={'x':{}},protocol={'scripts':{'x':'a'}})
            args=SimpleNamespace(scope='budget',tokens=2048)
            with patch.object(stats,'RESULTS',root),patch.object(stats,'context',return_value=ctx),patch('sys.stdout',new=io.StringIO()):
                stats.freeze_short(args); before=(root/'selected_short_config.json').read_bytes(); stats.freeze_short(args)
                self.assertEqual((root/'selected_short_config.json').read_bytes(),before)
                args.tokens=3072
                with self.assertRaises(ValueError): stats.freeze_short(args)
                self.assertEqual((root/'selected_short_config.json').read_bytes(),before)
    def test_refuse_freeze_before_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); write_once(root/'summary.json',dict(signature='s',audit=dict(passed=False)))
            ctx=dict(signature='s',stage_dir=root,expected={'x'},units={})
            with patch.object(stats,'context',return_value=ctx),self.assertRaises(ValueError): stats.freeze_short(SimpleNamespace(tokens=2048))
    def test_short_context_blocked_without_frozen_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); stage=root/'parent'; stage.mkdir(); store=root/'cache'; store.mkdir()
            write_once(stage/'summary.json',dict(N=180,audit=dict(passed=True)))
            (store/'units.jsonl').write_text('{}\n')
            parent=dict(signature=c.PARENT_SIG,rows=[row()],units={'1':{}},stage_dir=stage,store=store,protocol={},old={'count_tokens':len,'cues':{}})
            with patch.object(c,'RESULTS',root),patch.object(c,'parent_context',return_value=parent):
                with self.assertRaisesRegex(ValueError,'freeze-short'): c.context(SimpleNamespace(scope='short'))


class Reports(unittest.TestCase):
    def test_full_budget_negative_result_and_reuse_count(self):
        rows=[]; units={}; parent=dict(units={},old=dict(units={},base=dict(units={}))); labels={}
        for i in range(180):
            r=row(str(i)); r['video_id']=f'v{i//3}'; r['duration']='medium' if i<96 else 'long'; rows.append(r)
            labels[str(i)]='A'; parent['units'][str(i)]={'pred':'A'}; parent['old']['units'][str(i)]={'pred':'A'}
            parent['old']['base']['units'][f'{i}:frames:p0']={'semantic_pred':'A'}
            units[c.key(str(i),'continuous')]=dict(index=str(i),pred='B' if i==0 else 'A',origin='generated' if i<24 else 'reused_old',
                new_answer_calls=int(i<24),visual={'n_tok':4608},subtitle_tokens=2000,input_tokens=7000,inference_s=1.0 if i<24 else 0.0,
                preprocess_s=0.1,peak_allocated_gib=10.0 if i<24 else 0.0,record_sha256='a'*64)
        ctx=dict(scope='budget',signature='s',tokens=3072,rows=rows,arms=('continuous',),units=units,expected=set(units),parent=parent,changed={str(i) for i in range(24)})
        report,details=stats.build_report(None,ctx,labels)
        self.assertTrue(report['audit']['passed']); self.assertEqual(report['audit']['origins'],dict(generated=24,reused_old=156))
        self.assertEqual(report['overall']['methods']['continuous_3072']['vs']['continuous_2048']['net'],-1)
        self.assertEqual(report['cost']['continuous']['new_generation_calls'],24); self.assertEqual(len(details),180)
        units[c.key('1','continuous')]['pred']=''
        report,_=stats.build_report(None,ctx,labels)
        self.assertFalse(report['audit']['passed']); self.assertEqual(report['overall']['N'],180)
        self.assertEqual(report['overall']['methods']['continuous_3072']['invalid'],1)


if __name__=='__main__': unittest.main(verbosity=2)
