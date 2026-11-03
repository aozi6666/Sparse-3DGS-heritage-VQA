#!/usr/bin/env python3
"""验证配对日志、细粒度恢复、缺字幕回退与离线统计；不加载模型或解码视频。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from practical_common import signature, prompt, append_line
from subtitle_common import sealed, subtitle_prompt
import subtitle_holdout as h
import subtitle_holdout_analyze as a


def row(index='1',video='video1',duration='medium'):
    return dict(index=index,video_id=video,duration=duration,task_type='test',
        question='What is shown?',bodies=[' one',' two',' three',' four'])


def alignment(status='used'):
    return dict(evidence=dict(frame_indices=list(range(32)),frame_seconds=list(range(32)),
        status=status,text='hello' if status=='used' else '',text_tokens=1 if status=='used' else 0,
        frame_cue_ids=[[1]]*32 if status=='used' else [[]]*32),
        native_inputs_equal={'plain':True,'subtitle':True})


def record(r,arm,base=None,status='used',pred='A'):
    evidence=alignment(status)['evidence']; fallback=arm==h.ARMS[1] and status!='used'
    query=prompt(r,0) if arm==h.ARMS[0] else subtitle_prompt(r,evidence['text'])
    plain_hash=signature([1,2,3]); input_hash=signature([1,2,3,4]) if arm==h.ARMS[1] and not fallback else plain_hash
    item=dict(signature='sig',index=r['index'],video_id=r['video_id'],arm=arm,origin_stage='val',
        origin='reused_current_no_sub' if fallback else 'generated',
        pred=base['pred'] if fallback else pred,raw_text=base['raw_text'] if fallback else pred,
        generated_tokens=None if fallback else 1,subtitle_status=status,evidence_sha256=signature(evidence),
        subtitle_tokens=evidence['text_tokens'] if arm==h.ARMS[1] else 0,
        prompt_sha256=signature(query),input_sha256=input_hash,input_tokens=5000 if input_hash==plain_hash else 5001,
        base_input_sha256=plain_hash,baseline_record_sha256=base['record_sha256'] if base else None,
        visual=dict(grid_thw=[[1,18,32]]*32,pv_shape=[18432,768],patch_time_range=[0.0,31.0],n_tok=4608),
        visual_shared=True,native_inputs_equal={'plain':True,'subtitle':True},
        cost=dict(inference_s=0.0 if fallback else 1.0,preprocess_s=0.0,
            peak_allocated_gib=0.0 if fallback else 10.0,new_answer_calls=0 if fallback else 1))
    return sealed(item)


def changed(item,**changes):
    item=deepcopy(item); item.pop('record_sha256'); item.update(changes)
    return sealed(item)


class Records(unittest.TestCase):
    def setUp(self):
        self.r=row(); self.base=record(self.r,h.ARMS[0]); self.key=h.unit_key('1',h.ARMS[0])

    def test_plain_and_sub_pair(self):
        self.assertEqual(h.validate_record(self.base,'sig',self.r,alignment(),{}),self.key)
        sub=record(self.r,h.ARMS[1],self.base,pred='B')
        self.assertEqual(h.validate_record(sub,'sig',self.r,alignment(),{self.key:self.base}),h.unit_key('1',h.ARMS[1]))

    def test_missing_empty_nohit_fallback(self):
        for status in ['missing','empty','no_hit']:
            base=record(self.r,h.ARMS[0],status=status)
            sub=record(self.r,h.ARMS[1],base,status=status)
            h.validate_record(sub,'sig',self.r,alignment(status),{self.key:base})

    def test_seal_corruption(self):
        item=deepcopy(self.base); item['raw_text']='C'
        with self.assertRaises(ValueError):
            h.validate_record(item,'sig',self.r,alignment(),{})

    def test_missing_baseline(self):
        with self.assertRaises(ValueError):
            h.validate_record(record(self.r,h.ARMS[1],self.base),'sig',self.r,alignment(),{})

    def test_foreign_baseline_reference(self):
        sub=changed(record(self.r,h.ARMS[1],self.base),baseline_record_sha256='0'*64)
        with self.assertRaises(ValueError):
            h.validate_record(sub,'sig',self.r,alignment(),{self.key:self.base})

    def test_visual_drift(self):
        sub=record(self.r,h.ARMS[1],self.base); visual=deepcopy(sub['visual']); visual['n_tok']+=1
        with self.assertRaises(ValueError):
            h.validate_record(changed(sub,visual=visual),'sig',self.r,alignment(),{self.key:self.base})

    def test_fallback_cannot_change_answer(self):
        base=record(self.r,h.ARMS[0],status='missing')
        sub=changed(record(self.r,h.ARMS[1],base,status='missing'),pred='B',raw_text='B')
        with self.assertRaises(ValueError):
            h.validate_record(sub,'sig',self.r,alignment('missing'),{self.key:base})

    def test_reject_cost_nan_and_fake_call(self):
        for changes in [{'inference_s':float('nan')},{'new_answer_calls':0}]:
            cost=dict(self.base['cost'],**changes)
            with self.assertRaises(ValueError):
                h.validate_record(changed(self.base,cost=cost),'sig',self.r,alignment(),{})

    def test_reject_prompt_output_or_native_audit_drift(self):
        for changes in [dict(prompt_sha256='x'*64),dict(pred='D'),dict(native_inputs_equal={'plain':False,'subtitle':True}),dict(origin_stage='test')]:
            with self.assertRaises(ValueError):
                h.validate_record(changed(self.base,**changes),'sig',self.r,alignment(),{})

    def test_invalid_answer_is_recorded_for_denominator(self):
        item=changed(self.base,pred='',raw_text='No response')
        h.validate_record(item,'sig',self.r,alignment(),{})

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            h.validate_record(self.base,'sig',self.r,alignment(),{self.key:self.base})

    def test_resume_between_arms_and_tail_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'units.jsonl'
            with path.open('w') as f:
                append_line(f,self.base)
            units=h.load_units(path,'sig',[self.r],lambda _:alignment())
            self.assertEqual(list(units),[self.key])
            with path.open('a') as f:
                append_line(f,record(self.r,h.ARMS[1],self.base))
            self.assertEqual(len(h.load_units(path,'sig',[self.r],lambda _:alignment())),2)
            with path.open('a') as f:
                f.write('{"partial":')
            before=path.read_bytes()
            with self.assertRaises(ValueError):
                h.load_units(path,'sig',[self.r],lambda _:alignment())
            self.assertEqual(path.read_bytes(),before)

    def test_foreign_index_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'units.jsonl'; path.write_text(json.dumps(self.base)+'\n')
            with self.assertRaises(ValueError):
                h.load_units(path,'sig',[row('2')],lambda _:alignment())


class Reporting(unittest.TestCase):
    def test_rescue_harm_quadrants(self):
        rows=[row(str(i),f'v{i//3}') for i in range(6)]; units={}; labels={r['index']:'A' for r in rows}
        for r,plain,sub in zip(rows,['A','B','A','B','B','A'],['A','A','B','B','A','A']):
            units[h.unit_key(r['index'],h.ARMS[0])]={'pred':plain}
            units[h.unit_key(r['index'],h.ARMS[1])]={'pred':sub}
        report=a.group_metrics(rows,units,labels)
        self.assertEqual(report['methods'][h.ARMS[1]]['quadrants'],dict(both_ok=2,rescued=2,harmed=1,both_wrong=1))
        self.assertEqual(report['methods'][h.ARMS[1]]['vs'][h.ARMS[0]]['net'],1)

    def test_empty_group(self):
        self.assertEqual(a.group_metrics([],{},{}),dict(N=0,videos=0,methods={}))

    def test_partial_summary_does_not_read_labels(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read GT')):
            with self.assertRaises(ValueError):
                a.summarize(dict(expected={'x'},units={}))

    def test_build_report_invalid_denominator_and_fallback_cost(self):
        rows=[row('1'),row('2','video2','long')]; units={}; evidence={}
        for r,status,answer in zip(rows,['used','missing'],['','A']):
            base=record(r,h.ARMS[0],status=status,pred=answer)
            if not answer:
                base=changed(base,raw_text='No response')
            sub=record(r,h.ARMS[1],base,status=status,pred='B')
            for u in [base,sub]:
                units[h.unit_key(r['index'],u['arm'])]=u
            evidence[r['video_id']]=alignment(status)
        ctx=dict(rows=rows,units=units,expected=set(units),signature='sig',
            evidence_loader=evidence.__getitem__,store=Path('/not_written'))
        report,details=a.build_report(ctx,{'1':'B','2':'A'})
        self.assertFalse(report['audit']['passed'])
        self.assertEqual(report['overall']['N'],2)
        self.assertEqual(report['overall']['methods'][h.ARMS[1]]['correct'],2)
        self.assertEqual(report['cost'][h.ARMS[1]]['new_answer_calls'],1)
        self.assertEqual(report['cost'][h.ARMS[1]]['logical_inference_s'],2)
        self.assertEqual(len(details),2)


class Guards(unittest.TestCase):
    def test_partition_overlap(self):
        rows=[row(str(i),f'v{i//3}') for i in range(90)]
        h.validate_partition(rows,[row('100','dev')],[row('101','test')])
        with self.assertRaises(ValueError):
            h.validate_partition(rows,[row('100','v0')],[row('101','test')])
        with self.assertRaises(ValueError):
            h.validate_partition(rows[:89],[],[])

    def test_run_requires_user_switch_before_any_write(self):
        with patch.object(h,'preflight'),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}):
            with self.assertRaises(RuntimeError):
                h.run(None,{})

    def test_test_and_full_are_rejected_before_context(self):
        for stage in ['test','full']:
            with patch.object(sys,'argv',['subtitle_holdout.py','preflight','--stage',stage]),patch.object(h,'make_context') as context:
                with self.assertRaises(SystemExit):
                    h.main()
                context.assert_not_called()

    def test_import_does_not_load_gpu_libraries(self):
        code='import subtitle_holdout, subtitle_holdout_analyze,sys; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,check=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
