#!/usr/bin/env python3
"""纯 CPU 合成单元测试：不导入 Torch，不加载模型，不接触真实测试标签。"""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from practical_common import (LETTERS,SLOT,permutation,remap,prompt,position_sensitive,
    aggregate,softmax,write_once,read_lines,append_line,make_splits,jobs_for,
    ballot_key,load_ballots,signature,parse_answer)
from practical_vote import CaptureTokenizer
from practical_analyze import predictions,freeze,summarize


def row(index='0'):
    return dict(index=index,video_id='v'+index,question='What happens?',
                bodies=[' red',' green',' blue',' yellow'],no_permute=False)


def ballot(item,k,display='A'):
    p=softmax([3,2,1,0])
    return dict(signature='test',index=item['index'],backend='frames',permutation_id=k,
        prompt_sha256=signature(prompt(item,k)),display_pred=display,
        semantic_pred=LETTERS[permutation(k)[LETTERS.index(display)]] if display else '',
        plain_logits=[3,2,1,0],semantic_p=remap(p,k))


class Tests(unittest.TestCase):
    def test_four_rotations_are_balanced(self):
        for original in range(4):
            self.assertEqual(sorted(permutation(k).index(original) for k in range(4)),list(range(4)))
        with self.assertRaises(ValueError):permutation(4)

    def test_semantic_mapping_not_letter_vote(self):
        item=row()
        # 同一语义 C 在四次分别显示为 C/B/A/D，映射后四票应全部为 C。
        units=[ballot(item,k,LETTERS[permutation(k).index(2)]) for k in range(4)]
        self.assertEqual([u['semantic_pred'] for u in units],['C']*4)
        self.assertEqual(aggregate(units,'vote'),'C')

    def test_probability_inverse_mapping(self):
        original=[.1,.2,.3,.4]
        for k in range(4):
            display=[original[i] for i in permutation(k)]
            self.assertEqual(remap(display,k),original)

    def test_identity_prompt_exact(self):
        item=row()
        self.assertTrue(prompt(item,0).endswith('A. red\nB. green\nC. blue\nD. yellow'))
        self.assertTrue(prompt(item,1).endswith('A. green\nB. blue\nC. yellow\nD. red'))

    def test_unsafe_reference_and_safe_ordinary_text(self):
        for text in ['All of the above','Both A and B','A and C are correct','Option B is right','以上都对']:
            item=row();item['bodies'][0]=' '+text
            self.assertTrue(position_sensitive(item),text)
        for text in ['a red door','Both people turn around','The three kinds are of the same number.']:
            item=row();item['bodies'][0]=' '+text
            self.assertFalse(position_sensitive(item),text)

    def test_tie_prefers_original_not_gt(self):
        units=[dict(permutation_id=i,semantic_pred=c,semantic_p=[.1,.2,.3,.4]) for i,c in enumerate('ABAB')]
        self.assertEqual(aggregate(units,'vote'),'A')
        units[0]['semantic_pred']='B'
        self.assertEqual(aggregate(units,'vote'),'B')

    def test_invalid_votes_are_not_forced_legal(self):
        units=[dict(permutation_id=i,semantic_pred='',semantic_p=[.9,.04,.03,.03]) for i in range(4)]
        self.assertEqual(aggregate(units,'vote'),'')
        self.assertEqual(aggregate(units,'mean'),'')
        units[1]['semantic_pred']='C'
        self.assertEqual(aggregate(units,'vote'),'C')

    def test_free_generation_parser_and_unscorable_prefix(self):
        self.assertEqual(parse_answer('The answer is B.'),'B')
        self.assertEqual(parse_answer('A.'),'A')
        self.assertEqual(parse_answer('Because it happens.'),'')
        units=[dict(permutation_id=0,semantic_pred='B',semantic_p=[.99,.01,0,0],score_eligible=False)]
        self.assertEqual(aggregate(units,'mean'),'B')

    def test_resume_single_unit_and_reject_duplicates(self):
        item=row()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'units.jsonl'
            with path.open('a') as f:append_line(f,ballot(item,0))
            loaded=load_ballots(path,'test',[item])
            self.assertEqual(set(loaded),{ballot_key('0','frames',0)})
            self.assertIsNone(predictions([item],loaded,'frames_vote'))
            with path.open('a') as f:append_line(f,ballot(item,0))
            with self.assertRaises(ValueError):load_ballots(path,'test',[item])

    def test_reject_partial_tail_and_changed_config(self):
        item=row()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'units.jsonl'
            path.write_text(json.dumps(ballot(item,0)))
            with self.assertRaises(ValueError):read_lines(path)
            path.write_text(json.dumps(ballot(item,0))+'\n')
            with self.assertRaises(ValueError):load_ballots(path,'different',[item])

    def test_reject_probability_or_prompt_corruption(self):
        item=row()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'units.jsonl'
            value=ballot(item,1);value['semantic_p'][0]=.123
            path.write_text(json.dumps(value)+'\n')
            with self.assertRaises(ValueError):load_ballots(path,'test',[item])
            value=ballot(item,1);value['prompt_sha256']='wrong'
            path.write_text(json.dumps(value)+'\n')
            with self.assertRaises(ValueError):load_ballots(path,'test',[item])

    def test_write_once_idempotent_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'lock.json'
            write_once(path,{'a':1});write_once(path,{'a':1})
            with self.assertRaises(ValueError):write_once(path,{'a':2})
            self.assertEqual(json.loads(path.read_text()),{'a':1})

    def test_video_disjoint_stratified_splits(self):
        rows=[]
        for duration,count in [('short',73),('medium',86),('long',79)]:
            for v in range(count):
                rows.extend(dict(video_id=f'{duration}_{v}',duration=duration,index=f'{duration}_{v}_{q}') for q in range(3))
        s=make_splits(rows)
        self.assertEqual([len(s[k]) for k in ['short_dev','dev','val','test','dev30']],[73,60,30,75,30])
        for a,b in [('short_dev','dev'),('dev','val'),('val','test'),('dev','test')]:
            self.assertFalse(set(s[a])&set(s[b]))
        self.assertTrue(set(s['dev30']).issubset(s['dev']))
        self.assertEqual(s,make_splits(list(reversed(rows))))

    def test_fallback_jobs_and_frozen_method(self):
        item=row()
        self.assertEqual(len(jobs_for(item)),5)
        self.assertEqual(len(jobs_for(item,True)),8)
        self.assertEqual(len(jobs_for(item,True,'frames_identity')),2)
        item['no_permute']=True
        self.assertEqual(len(jobs_for(item,True)),2)

    def test_capture_preserves_text_and_delegation(self):
        class Fake:
            special='value'
            def __call__(self,text,**kwargs):return {'text':text,'kwargs':kwargs}
        proxy=CaptureTokenizer(Fake())
        text=['visual tags '+SLOT+' assistant suffix']
        result=proxy(text,padding=True)
        self.assertEqual(proxy.text,text)
        self.assertEqual(result['text'],text)
        self.assertEqual(proxy.special,'value')

    def test_no_torch_imported(self):
        self.assertNotIn('torch',sys.modules)

    def test_freeze_before_test_and_reject_after_test_opened(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=root/'01_manifests'
            write_once(m/'inputs.json',{'rows':[dict(index='9',video_id='heldout',duration='long')]})
            write_once(m/'splits.json',{'test':['heldout']})
            stage=root/'val';stage.mkdir()
            write_once(stage/'summary.json',{'signature':'s','methods':{'frames_vote':{}}})
            ctx=dict(stage_dir=stage,signature='s',tag='tag',protocol={},units={})
            args=SimpleNamespace(stage='val',root=root,method='frames_vote')
            with patch('builtins.print'):
                freeze(args,ctx)
            self.assertEqual(json.loads((m/'frozen_tag.json').read_text())['method'],'frames_vote')
            ctx['units']={'any':{'index':'9'}}
            with self.assertRaises(ValueError):freeze(args,ctx)

    @unittest.skipUnless(importlib.util.find_spec('numpy') is not None,'汇总集成测试需要现有 NumPy')
    def test_offline_summary_end_to_end_with_synthetic_ballots(self):
        from practical_common import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=root/'01_manifests'
            write_once(m/'evaluation_labels.json',{'0':'A','1':'C'})
            write_once(m/'dataset_lock.json',{'labels_sha256':sha(m/'evaluation_labels.json')})
            rows=[row('0'),row('1')]
            for r in rows:r['duration']='short'
            units={}
            for item in rows:
                for backend,k in jobs_for(item):
                    display=('A' if item['index']=='0' else 'B') if backend=='codec' else LETTERS[permutation(k).index(2)]
                    value=ballot(item,k,display);value['backend']=backend
                    value.update(score_eligible=True,n_tok=128,inference_s=1.0,preprocess_s=.1,
                        peak_allocated_gib=10,frames_max_pixels=150000,frames_min_pixels=3136,
                        grid_thw=[[1,2,2]]*32,audit=dict(direct_input_equal=True,legacy_equal=True))
                    units[ballot_key(item['index'],backend,k)]=value
            ctx=dict(signature='test',rows=rows,units=units,expected=set(units),stage_dir=root/'smoke',chosen=None)
            args=SimpleNamespace(root=root,stage='smoke',frames=32,codec_vote=False)
            with patch('builtins.print'):
                summarize(args,ctx)
                summarize(args,ctx)  # 重复汇总幂等，不覆盖不同结果。
            report=json.loads((root/'smoke/summary.json').read_text())
            self.assertTrue(report['audit']['smoke_pass'])
            result=report['methods']['frames_vote']
            self.assertEqual(result['correct'],1)
            self.assertEqual(result['vs']['codec_identity']['rescue'],1)
            self.assertEqual(result['vs']['codec_identity']['harm'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
