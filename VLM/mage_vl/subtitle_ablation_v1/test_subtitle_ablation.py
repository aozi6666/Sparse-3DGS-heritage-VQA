"""仅合成文本/CPU Tensor与假模型；禁止真实模型、视频与API调用。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subtitle_ablation as m


def fixture():
    a=dict(cue_id=1,start_ms=1000,end_ms=2000,text='anchor')
    n=dict(cue_id=2,start_ms=2000,end_ms=3000,text='neighbor')
    aligned=dict(selected_cues=[a],text='anchor',frame_indices=list(range(32)),frame_seconds=[float(i) for i in range(32)])
    full=dict(blocks=[dict(id='S1',cue_ids=[1,2],start_ms=1000,end_ms=3000,text='anchor neighbor')],added_cue_ids=[2],budget=2048,text='S1 [00:00:01.000 --> 00:00:03.000] anchor neighbor',text_tokens=47)
    return aligned,full


class TextTests(unittest.TestCase):
    def test_time_removal_preserves_exact_body(self):
        a,f=fixture();f['blocks'][0]['text']='字幕 S8 [00:00:01.000 --> 00:00:02.000] literal'
        r=m.ablate(a,f,len)
        self.assertEqual(r['no_time']['text'],f['blocks'][0]['text'])
        self.assertEqual(r['no_time']['cue_ids'],[1,2])

    def test_neighbor_removal_keeps_anchor_updates_bounds(self):
        a,f=fixture();r=m.ablate(a,f,len)
        self.assertEqual(r['no_neighbors']['text'],'S1 [00:00:01.000 --> 00:00:02.000] anchor')
        self.assertEqual(r['no_neighbors']['cue_ids'],[1])

    def test_no_refill_of_saved_budget(self):
        a,f=fixture();r=m.ablate(a,f,len)
        self.assertNotIn('neighbor',r['no_neighbors']['text'])
        self.assertEqual(f['added_cue_ids'],[2])

    def test_absent_subtitle_exact_empty(self):
        r=m.ablate(dict(selected_cues=[]),dict(blocks=[],added_cue_ids=[]),len)
        self.assertEqual([r[x]['text'] for x in m.ARMS],['',''])

    def test_duplicate_ids_rejected(self):
        a,f=fixture();f['blocks'][0]['cue_ids']=[1,2,1]
        with self.assertRaises(ValueError):m.ablate(a,f,len)

    def test_dropped_anchor_rejected(self):
        a,f=fixture();f['blocks'][0]['cue_ids']=[2]
        with self.assertRaises(ValueError):m.ablate(a,f,len)

    def test_unrecorded_neighbor_rejected(self):
        a,f=fixture();f['added_cue_ids']=[]
        with self.assertRaises(ValueError):m.ablate(a,f,len)

    def test_cap_stops_no_truncation(self):
        a,f=fixture()
        with self.assertRaises(ValueError):m.ablate(a,f,lambda _:2049)

    def test_no_neighbors_identity_when_none_added(self):
        a,f=fixture();f['added_cue_ids']=[];f['blocks'][0].update(cue_ids=[1],text='anchor',end_ms=2000)
        f['text']='S1 [00:00:01.000 --> 00:00:02.000] anchor'
        self.assertEqual(m.ablate(a,f,len)['no_neighbors']['text'],f['text'])

    def test_multiple_blocks_no_cross_block_merge(self):
        a,f=fixture();second=dict(cue_id=7,start_ms=7000,end_ms=8000,text='second')
        a['selected_cues'].append(second)
        f['blocks'].append(dict(id='S7',cue_ids=[7],start_ms=7000,end_ms=8000,text='second'))
        r=m.ablate(a,f,len)
        self.assertEqual(r['no_time']['text'],'anchor neighbor\nsecond')
        self.assertEqual(r['no_neighbors']['text'].count('\n'),1)


class PersistenceTests(unittest.TestCase):
    def test_atomic_idempotent_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';m.atomic_json(p,{'a':1});m.atomic_json(p,{'a':1})
            with self.assertRaises(ValueError):m.atomic_json(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})
            self.assertEqual(list(Path(d).glob('.own_*')),[])

    def test_seal_detects_corruption(self):
        u=m.sealed({'a':[1,2]});self.assertEqual(m.unseal(u),{'a':[1,2]})
        u['a'][0]=3
        with self.assertRaises(ValueError):m.unseal(u)

    def test_torn_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{')
            with self.assertRaises(json.JSONDecodeError):m.atomic_json(p,{'x':1})
            self.assertEqual(p.read_text(),'{')


class MockPipelineTests(unittest.TestCase):
    def setup_fake(self):
        import torch
        self.torch=torch
        self.loads=0;self.generated=0;self.decodes=0
        outer=self
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return [ord(x) for x in text]
            def __call__(self,text,padding=True,return_tensors='pt'):
                t=torch.tensor([[ord(x) for x in s] for s in text]);return dict(input_ids=t,attention_mask=torch.ones_like(t))
            def decode(self,t,**kwargs):return ''.join(chr(x) for x in t.tolist())
        class VP:
            fixed_num_frames=32;max_frames=32;target_fps=None;max_pixels=150000;min_pixels=3136
            def __call__(self,videos,**kwargs):
                outer.decodes+=1
                return dict(frame_timestamps=[[float(i) for i in range(32)]],pixel_values=torch.zeros(128,768),image_grid_thw=torch.tensor([[1,2,2]]*32),patch_positions=torch.tensor([[i,0,0] for i in range(32) for _ in range(4)]))
        class Processor:
            def __init__(self):self.video_processor=VP();self.image_processor=VP();self.tokenizer=Tokenizer();self.spatial_merge_size=2
            def apply_chat_template(self,messages,**kwargs):return messages[0]['content'][1]['text']
            def __call__(self,text,videos,**kwargs):
                v=self.video_processor(videos=videos);v.pop('frame_timestamps');return dict(v,**self.tokenizer(['<visual>\n'+x for x in text],return_tensors='pt',padding=True))
        class Model:
            device='cpu';dtype=torch.bfloat16
            def eval(self):return self
            def generate(self,**kwargs):
                outer.generated+=1
                ids=kwargs['input_ids'];return torch.cat([ids,torch.tensor([[ord('A')]])],dim=1)
        self.processor=Processor()
        def loader(*a,**kw):self.loads+=1;return Model()
        self.loader=loader
        a,f=fixture();f['text_tokens']=len(f['text'])
        variant=m.ablate(a,f,len)
        a2=copy.deepcopy(a);f2=copy.deepcopy(f);f2['added_cue_ids']=[];f2['blocks'][0].update(cue_ids=[1],text='anchor',end_ms=2000)
        f2['text']='S1 [00:00:01.000 --> 00:00:02.000] anchor';f2['text_tokens']=len(f2['text'])
        evidence={'v1':dict(aligned=a,continuous=f,variants=variant),'v2':dict(aligned=a2,continuous=f2,variants=m.ablate(a2,f2,len))}
        stat=Path(__file__).stat()
        rows=[dict(index=str(i),video_id='v1' if i<2 else 'v2',stage='dev180',role='development',duration='medium',task_type='test',question='question'+str(i),bodies=[' one',' two',' three',' four'],video_path=str(Path(__file__)),video_identity=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)) for i in range(3)]
        old={}
        for r in rows:
            e=evidence[r['video_id']];old[r['index']]={}
            for arm in m.BASES:
                text=m.subtitle_prompt(r,'' if arm=='no_sub' else e[arm]['text'])
                val=self.processor(text=[text],videos=[r['video_path']])
                old[r['index']][arm]=dict(pred='A',raw_text='A',input_sha256=m.signature(val['input_ids'].tolist()),prompt_sha256=m.signature(text),visual=m.visual_info(val,2),input_tokens=val['input_ids'].shape[1],subtitle_tokens=0 if arm=='no_sub' else len(e[arm]['text']),source_record_sha256='source'+arm,inference_s=1.)
        self.decodes=0
        rt=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,gpu='test GPU',dtype='torch.bfloat16')
        self.c=dict(signature='mock',protocol=dict(runtime=rt,caveat='mock'),rows=rows,units={},bundle=dict(rows=rows,evidence=evidence,old=old,roles={'dev180':'development'}))

    def patches(self,d):
        import contextlib
        stack=contextlib.ExitStack()
        for name,value in [('OUT',Path(d)),('guard',lambda:{})]:stack.enter_context(patch.object(m,name,value))
        stack.enter_context(patch.dict('os.environ',{'PRACTICAL_STAGE_CONFIRMED':'1'}))
        for n,v in [('is_available',lambda:True),('get_device_name',lambda _: 'test GPU'),('manual_seed_all',lambda _:None),('synchronize',lambda:None),('reset_peak_memory_stats',lambda:None),('max_memory_allocated',lambda:1234)]:
            stack.enter_context(patch.object(self.torch.cuda,n,v))
        stack.enter_context(patch('transformers.AutoProcessor.from_pretrained',return_value=self.processor))
        stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=self.loader))
        return stack

    def test_full_mock_workflow_and_resume_without_models(self):
        self.setup_fake()
        with tempfile.TemporaryDirectory() as d,self.patches(d):
            m.run(self.c,'smoke3')
            self.assertEqual((self.loads,self.generated,self.decodes),(1,4,2))
            self.assertEqual(len(self.c['units']),6)
            self.assertEqual(sum(u['new_calls'] for u in self.c['units'].values()),4)
            m.run(self.c,'smoke3');self.assertEqual((self.loads,self.generated),(1,4))
            bad=copy.deepcopy(next(iter(self.c['units'].values())));bad['source_audit']['passed']=False
            with self.assertRaises(ValueError):m.validate_unit('mock',self.c['bundle'],{r['index']:r for r in self.c['rows']},bad)

    def test_interruption_after_checkpoint_no_duplicate_generation(self):
        self.setup_fake()
        with tempfile.TemporaryDirectory() as d,self.patches(d):
            real=m.atomic_json;fired=[False]
            def interrupt(path,obj):
                real(path,obj)
                if Path(path).parent.name=='units' and not fired[0]:fired[0]=True;raise RuntimeError('synthetic interrupt')
            with patch.object(m,'atomic_json',interrupt),self.assertRaisesRegex(RuntimeError,'synthetic'):
                m.run(self.c,'smoke3')
            self.c['units']={}
            for f in (Path(d)/'units').glob('*.json'):
                u=json.loads(f.read_text());self.c['units'][(u['index'],u['arm'])]=u
            m.run(self.c,'smoke3')
            self.assertEqual(self.generated,4)

    def test_source_mismatch_stops_before_generation(self):
        self.setup_fake();self.c['bundle']['old']['0']['continuous']['input_sha256']='wrong'
        with tempfile.TemporaryDirectory() as d,self.patches(d),self.assertRaisesRegex(ValueError,'身份未复现'):
            m.run(self.c,'smoke3')
        self.assertEqual(self.generated,0)

    def test_summarize_incomplete_does_not_read_labels(self):
        self.setup_fake()
        with self.assertRaisesRegex(ValueError,'尚未完整'):m.summarize(self.c,'smoke3')

    def test_summary_and_csv_are_idempotent(self):
        self.setup_fake()
        with tempfile.TemporaryDirectory() as d,self.patches(d):
            m.run(self.c,'smoke3')
            root=Path(d)/'old';folder=root/'01_manifests';folder.mkdir(parents=True)
            labels=folder/'evaluation_labels.json';labels.write_text(json.dumps({'0':'A','1':'B','2':'A'}))
            (folder/'dataset_lock.json').write_text(json.dumps(dict(labels_sha256=m.sha(labels))))
            with patch.object(m,'ROOT',root):
                m.summarize(self.c,'smoke3');m.summarize(self.c,'smoke3')
            s=json.loads((Path(d)/'results/smoke3/summary.json').read_text())
            self.assertTrue(s['audit']['passed'])
            self.assertEqual(s['new_cost']['calls'],4)
            self.assertEqual(s['new_cost']['reused'],2)
            self.assertEqual(len(json.loads((Path(d)/'results/smoke3/predictions.json').read_text())),3)


if __name__=='__main__':unittest.main()
