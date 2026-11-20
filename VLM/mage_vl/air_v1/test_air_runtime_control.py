"""Exercise actual selection closure with cached arrays/scores; no GPU/decoder."""
import ast
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import air_runtime as runtime
import air_analyze
from air_protocol import digest,seal,unseal,write_once,sha,query,ANALYSIS
from air_selection import AIRConfig,select_frames,parse_score,uniform_indices


def extract_nested(name,namespace):
    """Compile the real nested function, not a duplicate implementation."""
    for symbol,value in vars(runtime).items():namespace.setdefault(symbol,value)
    tree=ast.parse(inspect.getsource(runtime.run_gpu))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    node=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,'<actual-air-runtime-selection>','exec'),namespace)
    return namespace[name]


class RuntimeSelection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.row=dict(index='219',video_id='video',question='Question?',bodies=['a','b','c','d'])
        self.hit=dict(count=200,fps=30.0)
        self.ctx=dict(signature='sig',store=self.root,protocol=dict(selection=asdict(AIRConfig())))
        self.indices=list(range(0,200,10))+[199]
        self.features=np.ones((len(self.indices),768),dtype=np.float32)/np.sqrt(768.)
        for i in range(200):
            self.write_score(i)
            # Every possible LDS extra is already cached. Real read_feature
            # validates these; encoding/decoding remains forbidden below.
            p=self.root/'clip_features'/self.row['video_id']/f'extra_{i}.npz'
            p.parent.mkdir(parents=True,exist_ok=True)
            meta=dict(signature='sig',video_id=self.row['video_id'],kind='extra',index=i)
            np.savez(p,indices=np.asarray([i],dtype=np.int64),
                features=np.ones((1,768),dtype=np.float16)/np.float16(np.sqrt(768.)),
                meta=np.asarray(json.dumps(meta,sort_keys=True)),wall_s=np.asarray(.1))
        def forbidden(*a,**kw):raise AssertionError('No decoding/GPU work allowed in this test')
        self.namespace=dict(store=self.root,ctx=self.ctx,np=np,json=json,time=time,
            select_frames=select_frames,AIRConfig=AIRConfig,parse_score=parse_score,
            coarse=lambda row,hit:(self.indices,self.features),text_vector=lambda row:np.ones(768,dtype=np.float32)/np.sqrt(768.),
            unseal=unseal,seal=seal,write_once=write_once,digest=digest,sha=sha,query=query,ANALYSIS=ANALYSIS,
            tqdm=lambda items,**kw:items,check_space=lambda:None,raw_frames=forbidden,generate=forbidden,
            encode_images=forbidden,save_feature=forbidden,read_feature=forbidden)
        self.namespace['read_feature']=extract_nested('read_feature',self.namespace)
        self.selection=extract_nested('selection',self.namespace)
    def tearDown(self):self.temp.cleanup()
    def write_score(self,index,raw='Score: 5\nReasoning: Cached CPU control.',**kw):
        record=dict(signature='sig',index=self.row['index'],video_id=self.row['video_id'],frame_index=index,
            prompt_sha256=digest(ANALYSIS.format(query=query(self.row))),input_sha256='a'*64,
            raw_text=raw,score=parse_score(raw),inference_s=1.,preprocess_s=0.,peak_gib=10.,input_tokens=200,
            generated_tokens=10,visual=dict(n_tok=144))
        record.update(kw)
        p=self.root/'scores'/self.row['index']/f'{index}.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(seal(record)))
    def test_cached_frames_drive_real_selection_without_inference(self):
        result=self.selection(self.row,self.hit)
        unseal(result)
        self.assertTrue(result['final_indices']);self.assertLessEqual(len(result['final_indices']),32)
        self.assertEqual(result['final_indices'],sorted(set(result['final_indices'])))
        self.assertIsNone(result['fallback_reason'])
        self.assertTrue((self.root/'selections/219.json').is_file())
        self.assertTrue(list((self.root/'rounds/219').glob('*.json')))
    def test_final_selection_cache_skips_all_expensive_callbacks(self):
        first=self.selection(self.row,self.hit)
        self.namespace['coarse']=lambda *a: (_ for _ in ()).throw(AssertionError('must reuse final selection'))
        second=self.selection(self.row,self.hit)
        self.assertEqual(first,second)
    def test_round_interruption_replays_cached_scores(self):
        real_write=write_once
        def interrupted(path,obj):
            real_write(path,obj)
            if Path(path).parent.name=='219' and Path(path).parent.parent.name=='rounds':
                raise RuntimeError('interrupted after sealed round checkpoint')
        self.namespace['write_once']=interrupted
        with self.assertRaises(RuntimeError):self.selection(self.row,self.hit)
        self.assertFalse((self.root/'selections/219.json').exists())
        self.namespace['write_once']=real_write
        resumed=self.selection(self.row,self.hit)
        self.assertTrue(resumed['final_indices'])
        self.assertEqual(resumed,self.selection(self.row,self.hit))
    def test_bad_score_source_stops_before_generate(self):
        for i in range(200):self.write_score(i,prompt_sha256='bad')
        with self.assertRaises(ValueError):self.selection(self.row,self.hit)
        self.assertFalse((self.root/'selections/219.json').exists())
    def test_bad_score_parse_stops_before_generate(self):
        for i in range(200):self.write_score(i,score=2)
        with self.assertRaises(ValueError):self.selection(self.row,self.hit)
    def test_final_selection_tamper_stops(self):
        self.selection(self.row,self.hit);p=self.root/'selections/219.json'
        obj=json.loads(p.read_text());obj['final_indices']=[0];p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):self.selection(self.row,self.hit)
    def test_final_selection_wrong_signature_stops(self):
        self.selection(self.row,self.hit);p=self.root/'selections/219.json'
        obj=json.loads(p.read_text());obj.pop('record_sha256');obj['signature']='other';p.write_text(json.dumps(seal(obj)))
        with self.assertRaises(ValueError):self.selection(self.row,self.hit)
    def test_final_selection_video_identity_drift_stops(self):
        self.selection(self.row,self.hit)
        for hit in [dict(count=201,fps=30.),dict(count=200,fps=29.97)]:
            with self.assertRaises(ValueError):self.selection(self.row,hit)
    def test_final_selection_config_drift_stops(self):
        self.selection(self.row,self.hit);p=self.root/'selections/219.json'
        obj=json.loads(p.read_text());obj.pop('record_sha256');obj['config']['max_iterations']=99
        p.write_text(json.dumps(seal(obj)))
        with self.assertRaises(ValueError):self.selection(self.row,self.hit)
    def test_final_selection_score_trace_drift_stops(self):
        self.selection(self.row,self.hit);p=self.root/'selections/219.json'
        obj=json.loads(p.read_text());obj.pop('record_sha256')
        r=obj['rounds'][0];r['score_records'][str(r['candidates'][0])]['score']=1
        p.write_text(json.dumps(seal(obj)))
        with self.assertRaises(ValueError):self.selection(self.row,self.hit)
    def test_invalid_score_remains_explicit_and_uses_uniform_fallback(self):
        for i in range(200):self.write_score(i,raw='No valid scoring field')
        result=self.selection(self.row,self.hit)
        self.assertTrue(result['fallback_reason']);self.assertTrue(result['invalid_score_indices'])
        self.assertEqual(result['final_indices'],uniform_indices(200,32))
    def test_qA_option_D_is_in_actual_scoring_prompt(self):
        self.assertIn('D.d',ANALYSIS.format(query=query(self.row)))


class AnalysisGuards(unittest.TestCase):
    def test_partial_summary_does_not_open_labels(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read label/source files')):
            with self.assertRaises(ValueError):air_analyze.summarize(dict(expected={'219:uniform32','219:air_cap32'},units={}))


if __name__=='__main__':unittest.main()
