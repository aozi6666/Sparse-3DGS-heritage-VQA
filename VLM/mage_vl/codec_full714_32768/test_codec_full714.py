"""纯文本、假缓存和假GPU模型的CPU测试；不访问真实视频/模型/API。"""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import codec_full714 as m


class ProtocolTests(unittest.TestCase):
    def test_anchors_uniform_endpoints_and_subset(self):
        ids=list(range(0,1000,3));s=m.anchor_subset(ids)
        self.assertEqual(len(s),32);self.assertEqual(s[0],ids[0]);self.assertEqual(s[-1],ids[-1]);self.assertEqual(s,sorted(set(s)));self.assertTrue(set(s)<=set(ids))
    def test_small_source_not_fabricated_to32(self):
        self.assertEqual(m.anchor_subset([7]),[7]);self.assertEqual(m.anchor_subset([1,4,10]),[1,4,10])
    def test_bad_ids_stop(self):
        for ids in [[],[2,1],[1,1],[-1,3]]:
            with self.assertRaises(ValueError):m.anchor_subset(ids)
    def test_halfopen_dedup_and_real_seconds(self):
        cues=[dict(cue_id=1,start_ms=1000,end_ms=2000,text='one'),dict(cue_id=2,start_ms=2000,end_ms=3000,text='two')]
        e=m.align([10,15,20],10,cues,dict(status='available'),len)
        self.assertEqual([c['cue_id'] for c in e['selected_cues']],[1,2]);self.assertEqual(e['frame_cue_ids'],[[1],[1],[2]])
    def test_missing_and_nohit(self):
        for source,status in [(dict(status='missing'),'missing'),(dict(status='available'),'no_hit')]:
            e=m.align([0],25,[],source,len);self.assertEqual(e['status'],status);self.assertEqual(e['text'],'')
    def test_no_fps_fallback(self):
        for fps in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):m.align([0],fps,[],dict(status='missing'),len)
    def test_missing_meta_fps_rejected(self):
        with self.assertRaises(ValueError):m.validate_meta(dict(video='v',total_frames=100),'v')
    def test_evidence_caps_no_visual_change(self):
        ids=list(range(100));original=list(ids)
        cues=[dict(cue_id=1,start_ms=0,end_ms=10000,text='words')]
        e=m.evidence(ids,10,cues,dict(status='available'),len)
        self.assertEqual(len(e['codec_complete32768']['subtitle_anchors']),100)
        self.assertEqual(len(e['codec_anchor32_2048']['subtitle_anchors']),32)
        self.assertEqual(ids,original)
    def test_cache_key_known(self):
        path='/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/data/VideoMME/video/dphq5X-rMew.mp4'
        self.assertEqual(m.cache_name(path),'dphq5X-rMew_791fce2c0b3f8215713e92e0389a7520')
    def test_no_budget_refill_for_cap(self):
        cues=[dict(cue_id=i+1,start_ms=i*1000,end_ms=(i+1)*1000,text='word'+str(i)) for i in range(80)]
        e=m.evidence(list(range(800)),10,cues,dict(status='available'),len)
        self.assertEqual(e['codec_complete32768']['budget'],32768)
        self.assertEqual(e['codec_anchor32_2048']['budget'],2048)


class ReadonlyTests(unittest.TestCase):
    def test_cache_read_restore_and_wrong_config(self):
        with tempfile.TemporaryDirectory() as d:
            fp=Path(d)/'meta.json';fp.write_text('{}')
            v=dict(video_path='video',cache_dir=d,cache_files_sha256={str(fp):m.sha(fp)},fps=25.)
            calls=[]
            orig=lambda *args: (_ for _ in ()).throw(RuntimeError('must not generate'))
            mod=types.SimpleNamespace(process_codec_video=orig,_cache_dir_for=lambda video,cfg:Path(d),_load_codec_result=lambda p:(calls.append(str(p)) or dict(fps=25.,images=[],src_positions=[])))
            cfg=types.SimpleNamespace(**m.CODEC)
            with patch.object(m.importlib,'import_module',return_value=mod):
                with m.readonly_codec(object(),dict(videos={'v':v})):
                    mod.process_codec_video('video',cfg);mod.process_codec_video('video',cfg)
                    wrong=copy.copy(cfg);wrong.patch=14
                    with self.assertRaises(ValueError):mod.process_codec_video('video',wrong)
                self.assertIs(mod.process_codec_video,orig)
            self.assertEqual(len(calls),1);self.assertEqual(fp.read_text(),'{}')
    def test_changed_cache_rejected_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as d:
            fp=Path(d)/'x';fp.write_text('data')
            v=dict(video_path='v',cache_dir=d,cache_files_sha256={str(fp):'wrong'},fps=25.)
            mod=types.SimpleNamespace(process_codec_video=None,_cache_dir_for=lambda *a:Path(d))
            with patch.object(m.importlib,'import_module',return_value=mod),m.readonly_codec(object(),dict(videos={'v':v})):
                with self.assertRaises(ValueError):mod.process_codec_video('v',types.SimpleNamespace(**m.CODEC))
            self.assertEqual(fp.read_text(),'data')


class PipelineTests(unittest.TestCase):
    def setup_fake(self,d,answer='A'):
        import torch,transformers
        self.torch=torch;self.calls=0;self.loads=0
        outer=self
        rows=[dict(index=str(i),video_id='v'+str(i//3),duration=['short','medium','long'][i//3],question='q',bodies=[' a',' b',' c',' d']) for i in range(9)]
        for i,r in enumerate(rows):r.update(stage=['val90','short219','test225'][i//3],role='synthetic')
        visual=dict(grid_thw=[[1,2,2]],pv_shape=[4,768],patch_time_range=[0.,0.],n_tok=1)
        variants={}
        for video in ['v0','v1','v2']:
            variants[video]=dict(evidence={a:dict(continuous=dict(text='full' if a=='codec_complete32768' else ('full' if video=='v0' else 'control'),text_tokens=4)) for a in m.ARMS[1:]})
        refs={r['index']:dict(codec_old_visual=visual,codec_old=dict(semantic_pred='A'),frames_continuous2048=dict(pred='A')) for r in rows}
        rt=dict(torch=torch.__version__,transformers=transformers.__version__,gpu='fake',dtype='torch.bfloat16')
        self.c=dict(scope='full714',probe_mode=True,signature='mock',protocol=dict(runtime=rt,policy='mock'),bundle=dict(rows=rows,videos=variants,references=refs),units={})
        class FakeInputs:
            def __init__(self,c):self.proc=types.SimpleNamespace(tokenizer=types.SimpleNamespace(decode=lambda t,**kw:answer))
            def make(self,row):
                out={}
                for a in m.ARMS:
                    code=1 if a=='codec_no_sub' else 2 if a=='codec_complete32768' or row['video_id']=='v0' else 3
                    out[a]=dict(input_ids=torch.tensor([[code]]),attention_mask=torch.tensor([[1]]),pixel_values=torch.zeros(4,768),image_grid_thw=torch.tensor([[1,2,2]]),patch_positions=torch.zeros(4,3,dtype=torch.int64))
                return out
        self.FakeInputs=FakeInputs
        for r in rows:
            vals=FakeInputs(self.c).make(r)
            inputs={a:dict(tensor_sha256=m.common.tensor_digest(x),input_sha256=m.signature(x['input_ids'].tolist()),input_tokens=1,visual=visual) for a,x in vals.items()}
            m.atomic(Path(d)/'input_audits'/f'{r["index"]}.json',m.common.sealed(dict(signature='mock',passed=True,inputs=inputs)))
        class Model:
            device='cpu';dtype=torch.bfloat16
            def eval(self):return self
            def generate(self,**kw):outer.calls+=1;return torch.cat([kw['input_ids'],torch.tensor([[65]])],dim=1)
        def load(*a,**kw):self.loads+=1;return Model()
        self.load=load
    def patched(self,d):
        s=contextlib.ExitStack();s.enter_context(patch.object(m,'OUT',Path(d)));s.enter_context(patch.object(m,'guard',lambda:{}))
        s.enter_context(patch.object(m,'Inputs',self.FakeInputs));s.enter_context(patch.object(m,'readonly_codec',lambda *a:contextlib.nullcontext()))
        s.enter_context(patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'1'}))
        for n,f in [('is_available',lambda:True),('get_device_name',lambda _: 'fake'),('manual_seed_all',lambda _:None),('synchronize',lambda:None),('reset_peak_memory_stats',lambda:None),('max_memory_allocated',lambda:123)]:s.enter_context(patch.object(self.torch.cuda,n,f))
        s.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=self.load))
        return s
    def test_27_units_24_calls_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_fake(d)
            with self.patched(d):
                m.run(self.c);self.assertEqual((self.calls,len(self.c['units'])),(24,27))
                m.run(self.c);self.assertEqual((self.calls,self.loads),(24,1))
                root=Path(d)/'old';folder=root/'01_manifests';folder.mkdir(parents=True)
                labels=folder/'evaluation_labels.json';labels.write_text(json.dumps({str(i):'A' for i in range(9)}));(folder/'dataset_lock.json').write_text(json.dumps(dict(labels_sha256=m.sha(labels))))
                with patch.object(m,'ROOT',root):m.summarize(self.c);m.summarize(self.c)
                report=json.loads((Path(d)/'results/full714/summary.json').read_text());self.assertTrue(report['audit']['passed'])
                self.assertEqual(report['cost']['new_calls'],24)
    def test_control_mismatch_preserved_and_resume_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_fake(d,answer='B')
            with self.patched(d):
                with self.assertRaisesRegex(ValueError,'控制答案未复现'):m.run(self.c)
                self.assertEqual(self.calls,1)
                self.assertTrue((Path(d)/'units/0_codec_no_sub.json').exists())
                with self.assertRaisesRegex(ValueError,'已有控制'):m.run(self.c)
                self.assertEqual(self.loads,1)

    def test_new_scope_without_historical_controls(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_fake(d)
            for ref in self.c['bundle']['references'].values():ref['codec_old']=None;ref['codec_old_visual']=None
            with self.patched(d):
                m.run(self.c)
            self.assertTrue(all(u['baseline_answer_equal'] is None for u in self.c['units'].values()))

    def test_seed_development_after_tensor_check(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_fake(d)
            for r in self.c['bundle']['rows'][:3]:r['stage']='dev180'
            with self.patched(d):m.run(self.c)
            old=copy.deepcopy(self.c)
            # 旧开发阶段仍用13312名字，不能把新路线名直接当作旧文件名。
            migrated={}
            for (idx,arm),u in old['units'].items():
                old_arm=m.development_arm(arm)
                v=m.common.sealed(dict(m.common.unseal(u),arm=old_arm))
                migrated[(idx,old_arm)]=v
                if old_arm!=arm:m.atomic(Path(d)/'units'/f'{idx}_{old_arm}.json',v)
            old['units']=migrated
            new=copy.deepcopy(self.c);new['signature']='new';new['units']={}
            newdir=Path(d)/'new'
            for i in range(3):
                audit=json.loads((Path(d)/'input_audits'/f'{i}.json').read_text());audit=m.common.unseal(audit);audit['signature']='new'
                m.atomic(newdir/'input_audits'/f'{i}.json',m.common.sealed(audit))
            with patch.object(m,'OUT',newdir),patch.object(m,'DEV_OUT',Path(d)),patch.object(m,'context_development',return_value=old):
                m.seed_development(new);m.seed_development(new)
            self.assertEqual(len(new['units']),9)
            self.assertTrue(all(u['new_calls']==0 and u['origin']=='reuse_prior_development' for u in new['units'].values()))
            self.assertTrue(all(u['arm']==arm for (_,arm),u in new['units'].items()))
            for (idx,arm),u in new['units'].items():
                self.assertEqual(Path(u['reuse']['source_path']).name,f'{idx}_{m.development_arm(arm)}.json')
            self.assertEqual(self.calls,24)


class CacheBuildTests(unittest.TestCase):
    def test_no_encoding_even_if_user_confirmed(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);old=out/'old_cache';old.mkdir();(old/'sentinel').write_text('keep')
            row=dict(video_id='video',video_path='/fake/video.mp4')
            with patch.object(m,'OUT',out),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'1'}),patch.object(m.subprocess,'run') as tool:
                with self.assertRaisesRegex(ValueError,'禁止重新编码'):m.build_cache(row,{})
                tool.assert_not_called()
            self.assertEqual((old/'sentinel').read_text(),'keep')

    def test_incomplete_final_cache_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);row=dict(video_id='video',video_path='/fake/video.mp4');target=out/'cache/codec'/m.cache_name(row['video_path']);target.mkdir(parents=True);(target/'sentinel').write_text('keep')
            with patch.object(m,'OUT',out),patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'1'}),patch.object(m.subprocess,'run') as tool:
                with self.assertRaises(ValueError):m.build_cache(row,{})
                tool.assert_not_called()
            self.assertEqual((target/'sentinel').read_text(),'keep')

    def test_overflow_blocks_before_gpu_and_preserves_diagnostic(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cache=out/'old';cache.mkdir();(cache/'meta.json').write_text('{}')
            row=dict(index='1',stage='val90',video_id='video',video_path='/fake/video.mp4')
            source=dict(status='available')
            plan=dict(signature='mock',rows=[row],catalog={'video':str(cache)},srt_sources={'video':source})
            tokenizer=types.SimpleNamespace(encode=lambda *a,**kw:types.SimpleNamespace(ids=[]))
            with patch.object(m,'OUT',out),patch.object(m,'guard',lambda:{}),patch.object(m,'plan_context',return_value=plan),patch.object(m,'validate_meta',return_value=(25.,100)),patch.object(m,'subtitle_source',return_value=(source,[])),patch.object(m,'continuous_evidence',return_value=dict(text_tokens=32769)),patch('numpy.load',return_value=np.array([[0,0,0],[1,0,0]])),patch('tokenizers.Tokenizer.from_file',return_value=tokenizer):
                with self.assertRaisesRegex(ValueError,'预算/完整性'):m.prepare_cache()
            reports=list((out/'preflight').glob('budget_*.json'));self.assertEqual(len(reports),1)
            self.assertEqual(json.loads(reports[0].read_text())['blocked'][0]['full_tokens'],32769)
            self.assertFalse((out/'protocol.json').exists())


class MigrationTests(unittest.TestCase):
    def sample(self):
        primary=dict(budget=13312,continuous=dict(budget=13312,text='same exact body',text_tokens=7,omitted_neighbor_ids=[]),subtitle_anchors=[0,5])
        old=dict(video_path='v.mp4',cache_dir='/existing/cache',fps=25.,total_frames=100,source_indices=[0,5],cache_files_sha256={'x':'sha'},canvas_files=['c.jpg'],source_video_identity={'bytes':1},evidence={m.LEGACY_PRIMARY:primary,'codec_anchor32_2048':{'text':'control'}})
        new=copy.deepcopy(old);new['evidence'][m.PRIMARY]=new['evidence'].pop(m.LEGACY_PRIMARY)
        new['evidence'][m.PRIMARY]['budget']=32768;new['evidence'][m.PRIMARY]['continuous']['budget']=32768
        return old,new
    def check(self,old,new):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'video_records';p.mkdir();(p/'v.json').write_text(json.dumps(m.common.sealed(dict(video=old))))
            with patch.object(m,'LEGACY_OUT',Path(d)):return m.check_legacy_video('v',new)
    def test_budget_name_only_change_passes(self):
        self.assertTrue(self.check(*self.sample()))
    def test_actual_text_change_rejected(self):
        old,new=self.sample();new['evidence'][m.PRIMARY]['continuous']['text']='changed'
        with self.assertRaisesRegex(ValueError,'实际字幕'):self.check(old,new)
    def test_cache_or_time_change_rejected(self):
        old,new=self.sample();new['source_indices']=[0,6]
        with self.assertRaisesRegex(ValueError,'缓存/时间来源'):self.check(old,new)
    def test_control_change_rejected(self):
        old,new=self.sample();new['evidence']['codec_anchor32_2048']['text']='changed'
        with self.assertRaisesRegex(ValueError,'2048对照'):self.check(old,new)
    def test_missing_legacy_is_new_not_unchanged(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'LEGACY_OUT',Path(d)):
            self.assertFalse(m.check_legacy_video('new',{}))
    def test_old_primary_mapping_and_reject_unknown(self):
        self.assertEqual(m.development_arm(m.PRIMARY),'codec_complete13312')
        self.assertEqual(m.development_arm('codec_no_sub'),'codec_no_sub')
        with self.assertRaises(ValueError):m.development_arm('typo')


class ResourceProbeTests(unittest.TestCase):
    def probe(self,d,peak):
        out=Path(d)
        rows=[dict(index=str(i),video_id='v'+str(i//3)) for i in range(9)]
        for r in rows:
            n=100 if r['video_id']=='v2' else 10
            m.atomic(out/'input_audits'/f'{r["index"]}.json',dict(inputs={a:dict(input_tokens=n) for a in m.ARMS}))
        c=dict(signature='mock',protocol=dict(probe_indices=['0','1','2']),bundle=dict(rows=rows),units={})
        observed=[]
        def fake_run(s):
            observed.extend(r['index'] for r in s['bundle']['rows'])
            s['units']={(r['index'],a):dict(pred='A',peak_gib=peak) for r in s['bundle']['rows'] for a in m.ARMS}
        with patch.object(m,'OUT',out),patch.object(m,'run',side_effect=fake_run),patch('torch.cuda.get_device_properties',return_value=types.SimpleNamespace(total_memory=24*1024**3)):
            m.resource_probe(c)
        return observed,json.loads((out/'resource_gate.json').read_text())
    def test_longest_actual_input_and_original_probe_both_included(self):
        with tempfile.TemporaryDirectory() as d:
            seen,gate=self.probe(d,13.)
            self.assertEqual(set(seen),set('012678'))
            self.assertTrue(gate['passed']);self.assertEqual(gate['longest_audited_video'],'v2')
    def test_memory_gate_failure_preserves_report(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'显存余量'):self.probe(d,23.)
            self.assertFalse(json.loads((Path(d)/'resource_gate.json').read_text())['passed'])


if __name__=='__main__':unittest.main()
