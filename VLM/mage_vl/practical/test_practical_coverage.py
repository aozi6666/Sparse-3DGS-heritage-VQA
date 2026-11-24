"""路线 B 的 CPU 审计测试；不导入 Torch，不读取模型/视频。"""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import practical_coverage
from practical_coverage import audit_units, summarize
from practical_common import ballot_key, sha, write_once


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(index='1', video_id='v1', duration='long')]
        self.base={}
        for backend in ['codec','frames']:
            self.base[ballot_key('1',backend,0)] = dict(display_pred='A',semantic_pred='A',
                prefix_sha256='old',grid_thw=[[1,18,32]]*32,pv_shape=[18432,768],
                patch_time_range=[0.,100.],n_tok=4608,frames_max_pixels=150000,
                frames_min_pixels=3136,audit=dict(direct_input_equal=True),
                inference_s=1.,preprocess_s=2.,peak_allocated_gib=10.)
        self.units=deepcopy(self.base)
        self.frame=self.units['1:frames:p0']
        self.frame.update(grid_thw=[[1,18,32]]*48,pv_shape=[27648,768],n_tok=6912)

    def check(self):
        return audit_units(self.rows,self.units,self.base,48,smoke=True)['passed']

    def test_valid_48_and_64(self):
        self.assertTrue(self.check())
        self.frame.update(grid_thw=[[1,18,32]]*64,pv_shape=[36864,768],n_tok=9216)
        self.assertTrue(audit_units(self.rows,self.units,self.base,64,True)['passed'])

    def test_silently_capped_frames_rejected(self):
        self.units['1:frames:p0']=deepcopy(self.base['1:frames:p0'])
        self.assertFalse(self.check())

    def test_control_drift_rejected(self):
        self.units['1:codec:p0']['semantic_pred']='B'
        self.assertFalse(self.check())

    def test_wrong_tokens_rejected(self):
        self.frame['n_tok']-=1
        self.assertFalse(self.check())

    def test_time_drift_rejected(self):
        self.frame['patch_time_range']=[0.,99.]
        self.assertFalse(self.check())

    def test_pixel_budget_rejected(self):
        self.frame['frames_max_pixels']=300000
        self.assertFalse(self.check())

    def test_native_path_failure_rejected(self):
        self.frame['audit']['direct_input_equal']=False
        self.assertFalse(self.check())

    def test_invalid_answer_rejected(self):
        self.frame['display_pred']=''
        self.assertFalse(self.check())

    @unittest.skipUnless(importlib.util.find_spec('numpy'), '离线区间计算需要 numpy')
    def test_paired_summary_and_idempotence(self):
        # 端到端验证对照确实取历史 32 帧，救回/误伤方向正确，重汇总不覆盖异值。
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            labels=root/'01_manifests/evaluation_labels.json'
            write_once(labels, {'1':'A'})
            write_once(labels.parent/'dataset_lock.json', {'labels_sha256':sha(labels)})
            self.frame['semantic_pred']=self.frame['display_pred']='B'
            ctx=dict(expected=set(self.units),units=self.units,rows=self.rows,signature='synthetic',
                     stage_dir=root/'summary')
            base=dict(units=self.base)
            args=SimpleNamespace(frames=48,stage='smoke')
            with patch.object(practical_coverage,'ROOT',root), contextlib.redirect_stdout(io.StringIO()):
                summarize(args,ctx,base)
                summarize(args,ctx,base)
            report=json.loads((ctx['stage_dir']/'summary.json').read_text())
            self.assertTrue(report['audit']['passed'])
            self.assertEqual(report['methods']['frames32']['correct'],1)
            self.assertEqual(report['methods']['frames48']['correct'],0)
            self.assertEqual(report['methods']['frames48']['vs']['frames32']['harm'],1)
            self.assertEqual(report['methods']['frames48']['cost']['visual_tokens'],6912)

    def test_incomplete_checkpoint_refuses_summary(self):
        ctx=dict(expected=set(self.units), units={'1:codec:p0':self.units['1:codec:p0']})
        with self.assertRaisesRegex(ValueError,'尚未完成'):
            summarize(SimpleNamespace(),ctx,{})


if __name__=='__main__':
    unittest.main()
