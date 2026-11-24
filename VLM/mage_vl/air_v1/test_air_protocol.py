"""只用临时合成记录核验断点身份；不是新增评测数据集。"""
from copy import deepcopy
from dataclasses import asdict
import json,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import air_protocol as p
import air_run
import air_analyze
from air_selection import AIRConfig,uniform_indices

class Journals(unittest.TestCase):
    def test_seal_corruption_and_partial_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'u.jsonl';u=p.seal(dict(index='1',raw_text='A'))
            p.append(path,u);self.assertEqual(p.read_journal(path),[u])
            with self.assertRaises(ValueError):p.unseal(dict(u,raw_text='B'))
            with path.open('a') as f:f.write('{')
            with self.assertRaises(ValueError):p.read_journal(path)
    def test_write_once_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.json';p.write_once(path,{'a':1});p.write_once(path,{'a':1})
            with self.assertRaises(ValueError):p.write_once(path,{'a':2})
            self.assertEqual(json.loads(path.read_text()),{'a':1})
    def test_clip_chunks_no_option_d_loss(self):
        ids=list(range(161));chunks=p.text_chunks(ids,49406,49407)
        self.assertEqual(len(chunks),3);self.assertTrue(all(len(c)<=77 for c in chunks))
        self.assertEqual([i for c in chunks for i in c[1:-1]],ids)
        self.assertTrue(all(c[0]==49406 and c[-1]==49407 for c in chunks))

class UnitIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.row=dict(index='1',video_id='v',question='Question?',bodies=[' a',' b',' c',' d'])
        self.ctx=dict(signature='sig',store=self.root,lookup={'1':self.row},protocol=dict(selection=asdict(AIRConfig())))
        p.write_once(self.root/'video_metadata/v.json',dict(signature='sig',video_id='v',fps=30.,frame_count=1000))
        self.indices=uniform_indices(1000);self.seconds=[round(i/30,6) for i in self.indices]
        p.write_once(self.root/'native_audits/1.json',dict(signature='sig',index='1',input_sha256='a'*64,indices=self.indices,seconds=self.seconds,tensor_equal=True))
        self.base=p.seal(dict(signature='sig',index='1',video_id='v',arm='uniform32',origin='generated',new_calls=1,reuse_record=None,
            prompt_sha256=p.digest(p.prompt(self.row)),input_sha256='a'*64,input_tokens=5000,raw_text='A',pred='A',
            indices=self.indices,seconds=self.seconds,visual=dict(grid_thw=[[1,24,24]]*32,n_tok=4608,patch_time_range=[0.,999.]),
            native_uniform_equal=True,selection_sha256=None,inference_s=1.,preprocess_s=1.,peak_gib=11.,generated_tokens=1))
    def change(self,u,**kw):
        x=deepcopy(u);x.pop('record_sha256');x.update(kw);return p.seal(x)
    def test_uniform_native_identity(self):self.assertEqual(p.validate_unit(self.ctx,self.base,{}),'1:uniform32')
    def test_time_axis_not_dense(self):
        with self.assertRaises(ValueError):p.validate_unit(self.ctx,self.change(self.base,seconds=list(range(32))),{})
    def test_original_uniform_indices_required(self):
        with self.assertRaises(ValueError):p.validate_unit(self.ctx,self.change(self.base,indices=list(range(32))),{})
    def test_duplicate_and_visual_and_generated_length(self):
        for candidate,existing in [(self.base,{'1:uniform32':self.base}),(self.change(self.base,generated_tokens=33),{}),(self.change(self.base,visual=dict(self.base['visual'],patch_time_range=[0.,31.])),{})]:
            with self.assertRaises(ValueError):p.validate_unit(self.ctx,candidate,existing)
    def selection(self,indices,fallback):
        path=self.root/'selections/1.json';p.write_once(path,p.seal(dict(signature='sig',index='1',total_frames=1000,fps=30.,config=asdict(AIRConfig()),final_indices=indices,fallback_reason=fallback)))
        return path
    def test_exact_fallback_same_answer(self):
        s=self.selection(self.indices,'no_positive')
        u=self.change(self.base,arm='air_cap32',origin='fallback_uniform',new_calls=0,reuse_record=self.base['record_sha256'],selection_sha256=p.sha(s),generated_tokens=None,inference_s=0.,peak_gib=0.)
        p.validate_unit(self.ctx,u,{'1:uniform32':self.base})
        with self.assertRaises(ValueError):p.validate_unit(self.ctx,self.change(u,pred='B',raw_text='B'),{'1:uniform32':self.base})
    def test_partial_positive_chronological_original_positions(self):
        s=self.selection([100,300],None)
        u=self.change(self.base,arm='air_cap32',indices=[100,300],seconds=[round(100/30,6),10.],visual=dict(grid_thw=[[1,24,24]]*2,n_tok=288,patch_time_range=[100.,300.]),selection_sha256=p.sha(s))
        p.validate_unit(self.ctx,u,{'1:uniform32':self.base})
        with self.assertRaises(ValueError):p.validate_unit(self.ctx,u,{})
    def test_invalid_qa_not_discarded(self):p.validate_unit(self.ctx,self.change(self.base,raw_text='unparseable',pred=''),{})

class Guards(unittest.TestCase):
    def test_no_unapproved_gpu(self):
        with patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}):
            with self.assertRaises(RuntimeError):air_run.run(SimpleNamespace())
    def test_partial_summary_does_not_read_labels(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('no GT')):
            with self.assertRaises(ValueError):air_analyze.summarize(dict(expected={'1:uniform32'},units={}))
    def test_cli_forbids_test_full(self):
        for stage in ['val','test','full']:
            r=subprocess.run([sys.executable,'-B',air_run.__file__,'run','--stage',stage],capture_output=True)
            self.assertEqual(r.returncode,2)
    def test_import_no_runtime_models(self):
        r=subprocess.run([sys.executable,'-B','-c',"import air_runtime,air_protocol,sys;assert not any(n in sys.modules for n in ['torch','transformers','decord'])"],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
