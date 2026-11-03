"""字幕路线CPU测试：测试证据边界和恢复/统计行为，不加载Mage或解码视频。"""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from practical_common import SLOT, prompt, signature, append_line, write_once, sha
from subtitle_common import (parse_srt, clean_text, subtitle_source, subtitle_prompt,
    make_alignment, CaptureVideoProcessor, sealed, load_units, MAX_SUB_TOKENS)
from subtitle_analyze import build_report, summarize
import subtitle_analyze
import practical_subtitles


SOURCE=dict(status='available',path='video.srt',bytes=1,sha256='fixture',cues=1,nonempty_cues=1)
ROW=dict(index='1',video_id='video',question='What is shown?',bodies=[' One.',' Two.',' Three.',' Four.'],
         duration='long',task_type='Counting Problem')


def alignment(source=SOURCE,cues=None):
    cues=cues if cues is not None else [dict(cue_id=1,start_ms=0,end_ms=32000,text='spoken evidence')]
    return make_alignment('video',source,cues,list(range(32)),[float(i) for i in range(32)],lambda s:len(s.split()))


def fixture(evidence=None, stage='smoke', prediction='B'):
    e=evidence if evidence is not None else alignment()
    old=dict(semantic_pred='A',raw_text='A',prefix_sha256='plain',
        grid_thw=[[1,18,32]]*32,pv_shape=[18432,768],patch_time_range=[0.,31.],n_tok=4608)
    base={'1:frames:p0':old,'1:codec:p0':deepcopy(old)}
    used=e['status']=='used'
    r=sealed(dict(signature='sig',index='1',video_id='video',origin_stage=stage,
        origin='generated_with_subtitle' if used else 'reused_frames32_'+e['status'],
        pred=prediction if used else 'A',raw_text=prediction if used else 'A',subtitle_status=e['status'],
        evidence_sha256=signature(e),prompt_sha256=signature(subtitle_prompt(ROW,e['text'])),
        input_sha256='input',input_tokens=5000,subtitle_tokens=e['text_tokens'],generated_tokens=1 if used else None,
        visual={k:old[k] for k in ['grid_thw','pv_shape','patch_time_range','n_tok']},
        checks=dict(visual_unchanged=True,base_prefix_sha256='plain',native_input_equal=stage=='smoke',no_sub_answer_equal=stage=='smoke'),
        cost=dict(new_answer_calls=int(used),extra_smoke_control_calls=int(stage=='smoke'),
            answer_s=1. if used else 0.,control_s=1. if stage=='smoke' else 0.,preprocess_s=1.,peak_allocated_gib=10. if used else 0.)))
    return e,base,r


class Subtitles(unittest.TestCase):
    def test_plain_and_styled_srt(self):
        s='\ufeff1\r\n00:00:01,200 --> 00:00:02,300\r\n<font color="red">Wood &amp; earth.</font>\r\n\r\n2\r\n00:00:03.01 --> 00:00:04.1\r\nPlain text.\r\n'
        cues=parse_srt(s)
        self.assertEqual((cues[0]['start_ms'],cues[0]['end_ms'],cues[0]['text']),(1200,2300,'Wood & earth.'))
        self.assertEqual((cues[1]['start_ms'],cues[1]['end_ms']),(3010,4100))
        self.assertEqual(cues[1]['text'],'Plain text.')

    def test_style_and_line_breaks(self):
        self.assertEqual(clean_text(r'{\an8}<i>First</i><br/>second\Nthird'),'First second third')

    def test_real_srt_blank_lines_inside_caption_are_not_cue_boundaries(self):
        text='1\n00:00:00,433 --> 00:00:03,703\n<font color="white">\nFirst line\n\nsecond line.</font>\n\n2\n00:00:03,703 --> 00:00:06,606\n<font>Third\n\nfourth.</font>\n'
        cues=parse_srt(text)
        self.assertEqual([c['text'] for c in cues],['First line second line.','Third fourth.'])
        self.assertEqual(len(cues),2)

    def test_optional_numeric_index_and_numeric_caption_preserved(self):
        cues=parse_srt('00:00:00,000 --> 00:00:01,000\n42\n\n00:00:01,000 --> 00:00:02,000\n43\n')
        self.assertEqual([c['text'] for c in cues],['42','43'])

    def test_invalid_time_or_block_rejected(self):
        for text in ['bad content','1\n00:00:03,000 --> 00:00:02,000\nOops',
                     '1\n00:99:00,000 --> 01:00:00,000\nOops',
                     '1\n00:00:01,000 --> BROKEN\nOops']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_srt(text)

    def test_empty_and_missing_are_explicit(self):
        self.assertEqual(parse_srt(' \n'),[])
        with tempfile.TemporaryDirectory() as tmp, patch('subtitle_common.DATA',Path(tmp)):
            source,cues=subtitle_source('video')
            self.assertEqual((source['status'],cues),('missing',[]))
            (Path(tmp)/'subtitle').mkdir()
            (Path(tmp)/'subtitle/video.srt').write_text('')
            self.assertEqual(subtitle_source('video')[0]['status'],'empty')

    def test_bad_encoding_does_not_fall_back_silently(self):
        with tempfile.TemporaryDirectory() as tmp, patch('subtitle_common.DATA',Path(tmp)):
            (Path(tmp)/'subtitle').mkdir()
            (Path(tmp)/'subtitle/video.srt').write_bytes(b'\xff\xfe')
            with self.assertRaisesRegex(ValueError,'UTF-8'):
                subtitle_source('video')

    def test_reserved_tokens_rejected(self):
        for text in [SLOT,'<|im_end|>','&lt;|im_end|&gt;','\x00']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                clean_text(text)

    def test_half_open_seconds_boundary_and_overlap(self):
        cues=[dict(cue_id=1,start_ms=0,end_ms=1000,text='one'),
              dict(cue_id=2,start_ms=1000,end_ms=2000,text='two'),
              dict(cue_id=3,start_ms=500,end_ms=1500,text='overlap')]
        e=alignment(cues=cues)
        self.assertEqual(e['frame_cue_ids'][0],[1])
        self.assertEqual(e['frame_cue_ids'][1],[2,3])
        self.assertEqual(e['frame_cue_ids'][2],[])
        self.assertEqual([c['cue_id'] for c in e['selected_cues']],[1,3,2])

    def test_same_cue_once_but_repeated_phrase_at_other_time_kept(self):
        e=alignment(cues=[dict(cue_id=1,start_ms=0,end_ms=10000,text='same'),
                         dict(cue_id=2,start_ms=20000,end_ms=32000,text='same')])
        self.assertEqual(e['text'],'same\nsame')
        self.assertEqual(len(e['selected_cues']),2)

    def test_milliseconds_not_raw_frame_indices(self):
        cues=[dict(cue_id=1,start_ms=1000,end_ms=1100,text='correct time')]
        e=make_alignment('video',SOURCE,cues,[i*30 for i in range(32)],
                         [float(i) for i in range(32)],lambda s:len(s))
        self.assertEqual(e['frame_cue_ids'][1],[1])
        self.assertEqual(e['frame_indices'][1],30)

    def test_no_hit_and_no_future_leak(self):
        e=alignment(cues=[dict(cue_id=1,start_ms=32000,end_ms=33000,text='not sampled')])
        self.assertEqual((e['status'],e['text']),('no_hit',''))

    def test_budget_overflow_stops_without_truncation(self):
        with self.assertRaisesRegex(ValueError,'不截断'):
            make_alignment('video',SOURCE,[dict(cue_id=1,start_ms=0,end_ms=32000,text='x')],
                list(range(32)),list(map(float,range(32))),lambda s:MAX_SUB_TOKENS+1)

    def test_bad_frame_metadata_rejected(self):
        for indices,times in [(list(range(31)),list(map(float,range(31)))),
                              ([0]*32,list(map(float,range(32)))),
                              (list(range(32)),[float('nan')]*32)]:
            with self.assertRaises(ValueError):
                make_alignment('video',SOURCE,[],indices,times,len)

    def test_no_sub_prompt_is_exact_and_only_subtitles_added(self):
        self.assertEqual(subtitle_prompt(ROW,''),prompt(ROW,0))
        text=subtitle_prompt(ROW,'evidence')
        self.assertIn("This video's subtitles are listed below:\nevidence\n",text)
        self.assertEqual(text.split('Select the best answer',1)[1],prompt(ROW,0).split('Select the best answer',1)[1])

    def test_capture_processor_forwards_temporary_frame_override(self):
        class Dummy:
            fixed_num_frames=None
            def __call__(self,**kwargs):
                return dict(frame_timestamps=[[float(i) for i in range(self.fixed_num_frames)]])
        native=Dummy(); wrapped=CaptureVideoProcessor(native)
        saved=wrapped.fixed_num_frames
        wrapped.fixed_num_frames=32
        result=wrapped()
        self.assertEqual(native.fixed_num_frames,32)
        self.assertEqual(wrapped.timestamps,result['frame_timestamps'])
        wrapped.fixed_num_frames=saved
        self.assertIsNone(native.fixed_num_frames)

    def test_fine_grained_resume(self):
        e,base,record=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'units.jsonl'
            self.assertEqual(load_units(p,'sig',[ROW],base,lambda _:e),{})
            with p.open('a') as f: append_line(f,record)
            self.assertEqual(set(load_units(p,'sig',[ROW],base,lambda _:e)),{'1'})

    def test_duplicate_partial_and_changed_signature_rejected(self):
        e,base,r=fixture()
        for contents in [json.dumps(r)+'\n'+json.dumps(r)+'\n',json.dumps(r),
                         json.dumps(sealed({**{k:v for k,v in r.items() if k!='record_sha256'},'signature':'other'}))+'\n']:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'units.jsonl'; p.write_text(contents)
                with self.assertRaises(ValueError): load_units(p,'sig',[ROW],base,lambda _:e)

    def test_tampered_prompt_evidence_or_baseline_rejected(self):
        e,base,r=fixture()
        for field,value in [('prompt_sha256','bad'),('evidence_sha256','bad'),('pred','C')]:
            changed={k:v for k,v in r.items() if k!='record_sha256'}; changed[field]=value
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'units.jsonl'; p.write_text(json.dumps(sealed(changed))+'\n')
                with self.assertRaises(ValueError): load_units(p,'sig',[ROW],base,lambda _:e)

    def test_changed_visual_or_prefix_refuses_resume(self):
        e,base,r=fixture()
        for change in ['visual','checks']:
            changed=deepcopy({k:v for k,v in r.items() if k!='record_sha256'})
            if change=='visual':
                changed['visual']['n_tok']=999
            else:
                changed['checks']['base_prefix_sha256']='different'
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'units.jsonl'; p.write_text(json.dumps(sealed(changed))+'\n')
                with self.assertRaises(ValueError): load_units(p,'sig',[ROW],base,lambda _:e)

    def test_no_sub_reuse_is_input_based_and_checked(self):
        source={**SOURCE,'status':'missing','cues':0,'nonempty_cues':0}
        e,base,r=fixture(alignment(source,[]))
        self.assertEqual(r['origin'],'reused_frames32_missing')
        self.assertEqual(r['cost']['new_answer_calls'],0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'units.jsonl'; p.write_text(json.dumps(r)+'\n')
            self.assertEqual(load_units(p,'sig',[ROW],base,lambda _:e)['1']['pred'],'A')

    def test_no_gpu_framework_imported(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)

    def test_incomplete_checkpoint_refuses_summary_before_gt_read(self):
        with self.assertRaisesRegex(ValueError,'尚未完成'):
            summarize(SimpleNamespace(),dict(expected={'1'},units={}))

    def test_dev_requires_real_smoke_completion_and_same_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'smoke'/practical_subtitles.TAG
            folder.mkdir(parents=True)
            summary=dict(signature='sig',stage='smoke',N=1,audit=dict(passed=True))
            (folder/'summary.json').write_text(json.dumps(summary))
            ctx=dict(signature='sig',units={})
            with patch.object(practical_subtitles,'RESULTS',Path(tmp)), \
                 patch.object(practical_subtitles,'stage_rows',return_value=[ROW]), \
                 patch.object(practical_subtitles.importlib.util,'find_spec',return_value=object()):
                with self.assertRaises(ValueError):
                    practical_subtitles.preflight(SimpleNamespace(stage='dev30'),ctx)
                ctx['units']={'1':object()}
                practical_subtitles.preflight(SimpleNamespace(stage='dev30'),ctx)
                summary['signature']='other'
                (folder/'summary.json').write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    practical_subtitles.preflight(SimpleNamespace(stage='dev30'),ctx)

    def test_run_requires_user_confirmation_before_any_gpu_or_writes(self):
        with patch.object(practical_subtitles,'preflight'), \
             patch.dict(os.environ,{'PRACTICAL_STAGE_CONFIRMED':'0'}), \
             patch.object(practical_subtitles,'run_gpu') as gpu:
            with self.assertRaisesRegex(RuntimeError,'用户'):
                practical_subtitles.run(SimpleNamespace(),{})
            gpu.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'CPU统计需要numpy')
    def test_summary_pairs_rescue_and_preserves_idempotence(self):
        e,base,r=fixture(prediction='B')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); labels=root/'01_manifests/evaluation_labels.json'
            write_once(labels,{'1':'B'})
            write_once(labels.parent/'dataset_lock.json',dict(labels_sha256=sha(labels)))
            ctx=dict(rows=[ROW],units={'1':r},base=dict(units=base),signature='sig',expected={'1'},
                     evidence_loader=lambda _:e,stage_dir=root/'result',store=root/'store')
            args=SimpleNamespace(stage='smoke')
            with patch.object(subtitle_analyze,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                summarize(args,ctx); summarize(args,ctx)
            report=json.loads((ctx['stage_dir']/'summary.json').read_text())
            self.assertTrue(report['audit']['passed'])
            self.assertEqual(report['overall']['methods']['frames32_aligned_sub']['vs']['frames32_no_sub']['rescue'],1)
            self.assertEqual(report['cost']['new_answer_calls'],1)
            self.assertEqual(report['groups']['missing']['N'],0)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'CPU统计需要numpy')
    def test_fallback_count_and_smoke_must_exercise_subtitles(self):
        e,base,r=fixture(alignment({**SOURCE,'status':'missing'},[]))
        ctx=dict(rows=[ROW],units={'1':r},base=dict(units=base),signature='sig',expected={'1'},
                 evidence_loader=lambda _:e,store=Path('/synthetic'))
        report,_=build_report(SimpleNamespace(stage='smoke'),ctx,{'1':'A'})
        self.assertFalse(report['audit']['passed'])
        self.assertEqual(report['cost']['logical_answer_calls'],1)
        self.assertEqual(report['cost']['new_answer_calls'],0)
        self.assertEqual(report['cost']['logical_visual_tokens'],4608)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'CPU统计需要numpy')
    def test_invalid_answer_kept_in_denominator_and_blocks_smoke(self):
        e,base,r=fixture(prediction='')
        ctx=dict(rows=[ROW],units={'1':r},base=dict(units=base),signature='sig',expected={'1'},
                 evidence_loader=lambda _:e,store=Path('/synthetic'))
        report,details=build_report(SimpleNamespace(stage='smoke'),ctx,{'1':'B'})
        self.assertEqual(report['N'],1)
        self.assertEqual(report['overall']['methods']['frames32_aligned_sub']['invalid'],1)
        self.assertFalse(report['audit']['passed'])
        self.assertEqual(len(details),1)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'CPU统计需要numpy')
    def test_new90_subset_without_cherry_picking(self):
        e,base,r=fixture(stage='dev60')
        ctx=dict(rows=[ROW],units={'1':r},base=dict(units=base),signature='sig',expected={'1'},
                 evidence_loader=lambda _:e,store=Path('/synthetic'))
        with patch.object(subtitle_analyze,'stage_rows',return_value=[]):
            report,_=build_report(SimpleNamespace(stage='dev60'),ctx,{'1':'B'})
        self.assertEqual(report['groups']['new90']['N'],1)
        self.assertEqual(report['groups']['new90']['methods']['frames32_aligned_sub']['correct'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
