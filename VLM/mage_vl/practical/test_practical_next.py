#!/usr/bin/env python3
"""连续补句与DeepSeek工程的纯CPU/模拟网络测试，不加载模型、不调用真实API。"""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock
import urllib.error

from practical_common import signature,append_line
import continuous_common as c
import continuous_mage as m
import deepseek_review as d
import deepseek_analyze as api_stats
from subtitle_common import sealed


def cue(i,start,end,text=None): return dict(cue_id=i,start_ms=start,end_ms=end,text=text or f'cue{i}')
def original(anchors,status='used'):
    return dict(selected_cues=anchors,status=status,frame_seconds=list(range(32)),frame_indices=list(range(32)))
def answer(**kwargs):
    obj=dict(answer='B',supported=True,complete=True,subtitle_quotes=[dict(block_id='S1',quote='hello world')],frame_ids=['F0'],windows=[],notes='evidence')
    obj.update(kwargs); return obj


class Continuous(unittest.TestCase):
    def test_timestamp(self): self.assertEqual(c.timestamp(3661002),'01:01:01.002')
    def test_neighbors_and_time(self):
        cues=[cue(1,0,1000),cue(2,1000,2000),cue(3,2000,3000),cue(4,3000,4000)]
        e=c.continuous_evidence(original([cues[1]]),cues,len,2048)
        self.assertEqual(e['added_cue_ids'],[1,3]); self.assertEqual(e['blocks'][0]['cue_ids'],[1,2,3])
        self.assertIn('00:00:00.000 --> 00:00:03.000',e['text'])
    def test_gap_not_filled(self):
        cues=[cue(1,0,1000),cue(2,10000,11000)]
        e=c.continuous_evidence(original([cues[1]]),cues,len)
        self.assertEqual(e['added_cue_ids'],[])
    def test_overlapping_cues_kept(self):
        cues=[cue(1,0,4000),cue(2,2000,5000),cue(3,3000,6000)]
        e=c.continuous_evidence(original(cues[:2]),cues,len)
        self.assertEqual(e['blocks'][0]['cue_ids'],[1,2,3])
    def test_empty_nohit_missing_exact(self):
        for status in ['empty','no_hit','missing']:
            e=c.continuous_evidence(original([],status),[cue(1,0,1000)],len)
            self.assertEqual(e['text'],''); self.assertEqual(e['status'],status)
    def test_budget_retains_anchors(self):
        cues=[cue(i,i*1000,(i+1)*1000,'x'*30) for i in range(1,12)]
        e=c.continuous_evidence(original([cues[2],cues[8]]),cues,len,150)
        ids={i for b in e['blocks'] for i in b['cue_ids']}
        self.assertTrue({3,9}.issubset(ids)); self.assertLessEqual(e['text_tokens'],150)
        self.assertTrue(e['omitted_neighbor_ids'])
    def test_anchor_overflow_stops(self):
        q=cue(1,0,1000,'x'*100)
        with self.assertRaises(ValueError): c.continuous_evidence(original([q]),[q],len,20)
    def test_no_gt_argument(self):
        import inspect
        self.assertNotIn('label',str(inspect.signature(c.continuous_evidence)))
    def test_mage_run_guard(self):
        with patch.object(m,'preflight'),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'0'}),self.assertRaises(RuntimeError): m.run(None,{})


class APIEvidence(unittest.TestCase):
    def setUp(self):
        self.e=dict(blocks=[dict(id='S1',text='hello world and more')]); self.frames=[dict(id='F0')]
    def validate(self,obj): return d.validate_answer(json.dumps(obj),self.e,self.frames,20)
    def test_valid(self): self.assertEqual(self.validate(answer())[1],[])
    def test_fabricated_quote(self): self.assertTrue(self.validate(answer(subtitle_quotes=[dict(block_id='S1',quote='false phrase')]))[1])
    def test_bad_id_type(self): self.assertTrue(self.validate(answer(subtitle_quotes=[dict(block_id=[],quote='hello')]))[1])
    def test_unknown_frame(self): self.assertTrue(self.validate(answer(frame_ids=['F999']))[1])
    def test_out_of_range_and_long_windows(self):
        for w in [dict(start=-1,end=2),dict(start=1,end=10),dict(start=19,end=21),dict(start=1,end=float('nan'))]:
            self.assertTrue(self.validate(answer(windows=[w]))[1])
    def test_no_citation_supported(self): self.assertTrue(self.validate(answer(frame_ids=[],subtitle_quotes=[]))[1])
    def test_abstention(self): self.assertEqual(self.validate(answer(answer='INSUFFICIENT',supported=False,complete=False,frame_ids=[],subtitle_quotes=[]))[1],[])
    def test_invalid_json(self): self.assertTrue(d.validate_answer(' ',self.e,self.frames,20)[1])
    def test_override_requires_verified_complete(self):
        row={'task_type':'Object Reasoning'}; ind=answer()
        self.assertTrue(d.allow_override(ind,[],answer(),[],row,False))
        for v in [answer(complete=False),answer(supported=False),answer(answer='A')]:
            self.assertFalse(d.allow_override(ind,[],v,[],row,False))
    def test_temporal_needs_dense_citation(self):
        row={'task_type':'Counting Problem'}
        self.assertFalse(d.allow_override(answer(),[],answer(),[],row,False))
        self.assertFalse(d.allow_override(answer(),[],answer(),[],row,True,{'F9'}))
        self.assertTrue(d.allow_override(answer(),[],answer(),[],row,True,{'F0'}))
    def test_independent_prompt_contains_no_mage_answer(self):
        row=dict(question='Q',bodies=['one','two','three','four'],task_type='Object Reasoning',gt='C',mage_answer='D')
        p=d.request_payload(row,self.e,[],20)
        query=json.loads(p['messages'][1]['content'][0]['text'])
        self.assertNotIn('gt',query); self.assertNotIn('mage_answer',query); self.assertNotIn('verification_target',query)
    def test_api_run_guard(self):
        with patch.object(d,'preflight'),patch.dict('os.environ',{'DEEPSEEK_API_CONFIRMED':'0'}),self.assertRaises(RuntimeError): d.run(None,{})


class APICache(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.store=Path(self.temp.name)
        self.ctx=dict(store=self.store,signature='sig'); self.args=SimpleNamespace(max_calls=3,max_total_tokens=1000,retry_uncertain=False)
        self.payload={'model':d.MODEL,'messages':[]}; self.secret='TEST_SECRET_NEVER_SAVE'
    def tearDown(self): self.temp.cleanup()
    def fake_response(self):
        raw=json.dumps(dict(choices=[dict(message=dict(content='{}'),finish_reason='stop')],usage=dict(total_tokens=10))).encode()
        response=MagicMock(); response.__enter__.return_value.read.return_value=raw
        opener=MagicMock(); opener.open.return_value=response
        return opener
    def test_response_saved_once_no_duplicate_network(self):
        opener=self.fake_response()
        with patch.object(d.urllib.request,'build_opener',return_value=opener):
            first=d.api_call(self.ctx,'1_independent',self.payload,self.args,self.secret)
            second=d.api_call(self.ctx,'1_independent',self.payload,self.args,self.secret)
        self.assertEqual(first,second); self.assertEqual(opener.open.call_count,1)
        for p in self.store.rglob('*'):
            if p.is_file(): self.assertNotIn(self.secret,p.read_text())
    def test_request_drift_rejected(self):
        with patch.object(d.urllib.request,'build_opener',return_value=self.fake_response()): d.api_call(self.ctx,'1_independent',self.payload,self.args,self.secret)
        with self.assertRaises(ValueError): d.api_call(self.ctx,'1_independent',{'different':1},self.args,self.secret)
    def test_pending_request_requires_ack(self):
        with (self.store/'attempts.jsonl').open('w') as f: append_line(f,dict(event='start',attempt=1,key='x'))
        with patch.object(d.urllib.request,'build_opener') as network,self.assertRaises(RuntimeError): d.api_call(self.ctx,'x',self.payload,self.args,self.secret)
        network.assert_not_called()
    def test_persistent_call_limit(self):
        self.args.max_calls=1
        with (self.store/'attempts.jsonl').open('w') as f: append_line(f,dict(event='start',attempt=1,key='old'))
        with patch.object(d.urllib.request,'build_opener') as network,self.assertRaises(ValueError): d.api_call(self.ctx,'x',self.payload,self.args,self.secret)
        network.assert_not_called()
    def test_cached_token_stopline(self):
        (self.store/'responses').mkdir(); (self.store/'responses/old.json').write_text(json.dumps(dict(response=dict(usage=dict(total_tokens=1000)))))
        with patch.object(d.urllib.request,'build_opener') as network,self.assertRaises(ValueError): d.api_call(self.ctx,'x',self.payload,self.args,self.secret)
        network.assert_not_called()
    def test_http_failure_not_automatically_retried(self):
        opener=MagicMock(); opener.open.side_effect=urllib.error.HTTPError('url',401,'unauthorized',{},None)
        with patch.object(d.urllib.request,'build_opener',return_value=opener),self.assertRaises(RuntimeError): d.api_call(self.ctx,'x',self.payload,self.args,self.secret)
        self.assertEqual(opener.open.call_count,1)
    def test_timeout_keeps_uncertain_attempt(self):
        opener=MagicMock(); opener.open.side_effect=TimeoutError()
        with patch.object(d.urllib.request,'build_opener',return_value=opener),self.assertRaises(RuntimeError): d.api_call(self.ctx,'x',self.payload,self.args,self.secret)
        with patch.object(d.urllib.request,'build_opener') as network,self.assertRaises(RuntimeError): d.api_call(self.ctx,'x',self.payload,self.args,self.secret)
        network.assert_not_called()


class Scope(unittest.TestCase):
    def test_manual_decord_install_without_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'__init__.py'; p.write_text('source')
            with patch.object(d.importlib.util,'find_spec',return_value=SimpleNamespace(origin=str(p))), \
                    patch.object(d.importlib.metadata,'version',side_effect=d.importlib.metadata.PackageNotFoundError('decord')):
                identity=d.decoder_identity()
            self.assertIsNone(identity['version']); self.assertEqual(len(identity['files']['__init__.py']),64)
    def test_import_no_model_libraries(self):
        code='import continuous_common,continuous_mage,deepseek_review,sys; assert "torch" not in sys.modules and "transformers" not in sys.modules'
        subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,check=True)
    def test_pilot_uses_predictions_only(self):
        rows=[dict(index=str(i),duration='medium' if i<10 else 'long') for i in range(20)]
        old={r['index']:{'pred':'A'} for r in rows}; base={f'{r["index"]}:frames:p0':{'semantic_pred':'A'} for r in rows}
        old['0']['pred']='B'; old['19']['pred']='B'
        selected=d.pilot_rows(rows,old,base)
        self.assertEqual(len(selected),14); self.assertTrue({'0','19'}.issubset({r['index'] for r in selected}))
        self.assertEqual(selected,d.pilot_rows(rows,old,base))
    def test_no_test_full_scope(self):
        self.assertNotIn('test',c.STAGES); self.assertNotIn('full',c.STAGES)
        self.assertNotIn('test',d.STAGES); self.assertNotIn('full',d.STAGES)


class APIPipeline(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('tqdm'),'requires existing tqdm; do not install')
    def test_two_calls_then_resume_without_network_or_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'cache').mkdir(); store=root/'cache/review'
            row=dict(index='1',video_id='v',duration='medium',question='Q',bodies=['a','b','c','d'],task_type='Object Reasoning')
            e=dict(blocks=[dict(id='S1',text='hello world')],original=dict(frame_seconds=list(range(32)),frame_indices=list(range(32))))
            ctx=dict(signature='sig',protocol={},store=store,rows=[row],units={},expected={'1'},stage_dir=root/'results',
                continuous=dict(evidence={'v':e},old=dict(base=dict(units={'1:frames:p0':dict(semantic_pred='A')}))))
            response=dict(choices=[dict(message=dict(content=json.dumps(answer())),finish_reason='stop')],usage=dict(total_tokens=10))
            fake=MagicMock(); fake.__enter__.return_value.read.return_value=json.dumps(response).encode()
            opener=MagicMock(); opener.open.return_value=fake
            args=SimpleNamespace(stage='smoke',min_free=10,budget=1,max_calls=6,max_total_tokens=1000,retry_uncertain=False)
            with patch.object(d,'ROOT',root),patch.object(d,'preflight'),patch.object(d,'context',return_value=ctx), \
                    patch.object(d,'disk_guard'),patch.object(d,'extract_images',return_value=([],[dict(id='F0',dense=False)],20,False)) as decode, \
                    patch.object(d.urllib.request,'build_opener',return_value=opener), \
                    patch.dict('os.environ',{'DEEPSEEK_API_CONFIRMED':'1','DEEPSEEK_API_KEY':'TEST_SECRET'}), \
                    patch('sys.stdout',new=io.StringIO()):
                d.run(args,ctx)
                self.assertEqual(opener.open.call_count,2); self.assertTrue(ctx['units']['1']['overridden'])
                self.assertEqual(ctx['units']['1']['system_pred'],'B')
                manifest=root/'01_manifests'; manifest.mkdir()
                labels=manifest/'evaluation_labels.json'; labels.write_text(json.dumps({'1':'B'}))
                (manifest/'dataset_lock.json').write_text(json.dumps(dict(labels_sha256=d.sha(labels))))
                with patch.object(api_stats,'ROOT',root):
                    api_stats.summarize(args,ctx)
                    first=(ctx['stage_dir']/'summary.json').read_bytes()
                    # 模拟后续pilot产生其他题的请求，不得改变已完成smoke的汇总。
                    with (store/'attempts.jsonl').open('a') as journal:
                        append_line(journal,dict(event='start',attempt=99,key='999_independent'))
                    api_stats.summarize(args,ctx)
                    self.assertEqual((ctx['stage_dir']/'summary.json').read_bytes(),first)
                decoded=decode.call_count
                d.run(args,ctx)
                self.assertEqual(opener.open.call_count,2); self.assertEqual(decode.call_count,decoded)


if __name__=='__main__': unittest.main(verbosity=2)
