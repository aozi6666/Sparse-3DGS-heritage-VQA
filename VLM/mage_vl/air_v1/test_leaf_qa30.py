"""CPU契约/模拟生成回归；合成测试夹具不作为研究数据集或准确率结果。"""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import air_protocol as p
import condition_clips as old
import leaf_qa30_protocol as k
import leaf_qa30_run as r
import leaf_qa30_analyze as a
from leaf_probe import write_once


IDS=list(old.IDS)+['401','355','424','296','406','275','389','259','347','543','669','579','506','516','596','672','549','524']


def fixture(audited=True,uniform_overrides=None):
    groups=dict(candidate=['325','450','424','669','389','625','484'],anchor_only=['265','549','454','406'])
    groups['fallback']=[i for i in IDS if i not in sum(groups.values(),[])]
    rows={i:dict(index=i,video_id='video'+i,duration='medium' if n%2 else 'long',question='Which fixture object?',bodies=['one','two','three','four']) for n,i in enumerate(IDS)}
    meta={i:dict(fps=25.,frame_count=1000) for i in IDS};oldplans={i:dict(indices=list(range(32)),fps=25.,count=1000) for i in IDS}
    locs={i:dict(index=i,group=next(g for g,ids in groups.items() if i in ids),original_noobs_indices=oldplans[i]['indices'],status='localized_candidates_only',
                 result=dict(indices=list(range(1,33)))) for i in IDS}
    plans=k.build_plans(IDS,groups,locs,oldplans,meta);bases={};units={}
    for i in IDS:
        letter=(uniform_overrides or {}).get(i,'A')
        bases[i]=p.seal(dict(index=i,pred=letter,raw_text=letter,indices=list(range(32))))
        fp=dict(tensor_sha=p.digest(['old',i]),baseline_tensor_sha=p.digest(['uniform',i]),input_tokens=100,visual_tokens=32,grid_thw=[[1,2,2]]*32)
        units[i]=p.seal(dict(index=i,indices=oldplans[i]['indices'],prompt_sha=p.digest(p.prompt(rows[i])),raw_text='A',pred='A',inference_s=.5,**fp))
    c=dict(signature='testsig',protocol=dict(indices=IDS,groups=groups),plans=plans,rows=rows,noobs=dict(units=units,bases=bases),audits={},generations={},units={})
    if audited:
        for i in IDS:c['audits'][i]=audit(c,i)
    return c,locs,oldplans,meta


def audit(c,i):
    return p.seal(dict(signature=c['signature'],index=i,indices=c['plans'][i]['old_noobs_indices'],**k.source_refs(c,i),
                       prompt_sha=p.digest(p.prompt(c['rows'][i])),input={n:c['noobs']['units'][i][n] for n in k.FP_FIELDS},native_equal=True,preprocess_s=1.))


def generation(c,i,answer='A'):
    count=len(c['plans'][i]['indices'])
    return p.seal(dict(signature=c['signature'],index=i,indices=c['plans'][i]['indices'],**k.source_refs(c,i),audit_record_sha=c['audits'][i]['record_sha256'],
                       prompt_sha=p.digest(p.prompt(c['rows'][i])),input=dict(tensor_sha=p.digest(['new',i]),baseline_tensor_sha=c['audits'][i]['input']['baseline_tensor_sha'],
                       input_tokens=100,visual_tokens=count,grid_thw=[[1,2,2]]*count),native_equal=True,generated_token_ids=[65],raw_text=answer,pred=p.parse_answer(answer),
                       preprocess_s=.2,inference_s=.1,peak_gib=2.))


class Plans(unittest.TestCase):
    def test_exact_7_23_and_484_retained(self):
        c,_,_,_=fixture();self.assertEqual(sum(x['generate'] for x in c['plans'].values()),7);self.assertTrue(c['plans']['484']['generate'])
        self.assertEqual(sum(not x['generate'] for x in c['plans'].values()),23)

    def test_anchor_and_fallback_ignore_probe_indices(self):
        c,_,_,_=fixture()
        for i in c['protocol']['groups']['anchor_only']+c['protocol']['groups']['fallback']:
            self.assertEqual(c['plans'][i]['indices'],list(range(32)))

    def test_missing_candidate_stops_not_silent_drop(self):
        c,l,o,m=fixture();l['484']['result']=None
        with self.assertRaises(ValueError):k.build_plans(IDS,c['protocol']['groups'],l,o,m)

    def test_duplicate_scope_rejected(self):
        c,l,o,m=fixture();c['protocol']['groups']['candidate'][0]='484'
        with self.assertRaises(ValueError):k.build_plans(IDS,c['protocol']['groups'],l,o,m)

    def test_original_noobs_identity_checked(self):
        c,l,o,m=fixture();l['549']['original_noobs_indices']=[100]
        with self.assertRaises(ValueError):k.build_plans(IDS,c['protocol']['groups'],l,o,m)

    def test_frame_budget_not_filled_or_changed(self):
        c,l,o,m=fixture();l['424']['result']['indices']=list(range(1,30));l['389']['result']['indices']=list(range(1,32))
        plans=k.build_plans(IDS,c['protocol']['groups'],l,o,m)
        self.assertEqual(len(plans['424']['indices']),29);self.assertEqual(len(plans['389']['indices']),31)

    def test_duplicate_or_outside_frames_rejected(self):
        for ids in [[1,1],[1000],list(range(33))]:
            c,l,o,m=fixture();l['484']['result']['indices']=ids
            with self.assertRaises(ValueError):k.build_plans(IDS,c['protocol']['groups'],l,o,m)

    def test_identical_candidate_input_reuses_not_regenerates(self):
        c,l,o,m=fixture();l['484']['result']['indices']=list(range(32))
        self.assertFalse(k.build_plans(IDS,c['protocol']['groups'],l,o,m)['484']['generate'])


class Records(unittest.TestCase):
    def setUp(self):self.c=fixture()[0]
    def test_audit_exact_full_input_match(self):k.validate_audit(self.c,self.c['audits']['549'])

    def test_audit_pixel_or_native_hash_mismatch_rejected(self):
        for name in ['tensor_sha','baseline_tensor_sha']:
            u=p.unseal(deepcopy(self.c['audits']['549']));u['input'][name]='f'*64
            with self.assertRaises(ValueError):k.validate_audit(self.c,p.seal(u))

    def test_audit_token_grid_mismatch_rejected(self):
        for name,value in [('input_tokens',101),('grid_thw',[[1,4,4]]*32)]:
            u=p.unseal(deepcopy(self.c['audits']['549']));u['input'][name]=value
            with self.assertRaises(ValueError):k.validate_audit(self.c,p.seal(u))

    def test_prompt_change_rejected(self):
        u=p.unseal(self.c['audits']['549']);u['prompt_sha']='changed'
        with self.assertRaises(ValueError):k.validate_audit(self.c,p.seal(u))

    def test_source_record_change_rejected(self):
        u=p.unseal(self.c['audits']['549']);u['source_noobs_record_sha']='changed'
        with self.assertRaises(ValueError):k.validate_audit(self.c,p.seal(u))

    def test_generation_for_anchor_or_fallback_rejected(self):
        for i in ['549','379']:
            with self.assertRaises(ValueError):k.validate_generation(self.c,generation(self.c,i))

    def test_generation_original_pixels_cannot_be_used_for_new_indices(self):
        g=p.unseal(generation(self.c,'484'));g['input']['tensor_sha']=self.c['audits']['484']['input']['tensor_sha']
        with self.assertRaises(ValueError):k.validate_generation(self.c,p.seal(g))

    def test_generation_frame_prompt_and_baseline_changed_rejected(self):
        for name,value in [('indices',[99]),('prompt_sha','changed'),('audit_record_sha','changed')]:
            g=p.unseal(generation(self.c,'484'));g[name]=value
            with self.assertRaises(ValueError):k.validate_generation(self.c,p.seal(g))

    def test_generated_length_limit(self):
        for tokens in [[],[65]*33,[True],[1.5]]:
            g=p.unseal(generation(self.c,'484'));g['generated_token_ids']=tokens
            with self.assertRaises(ValueError):k.validate_generation(self.c,p.seal(g))

    def test_invalid_answer_retained_not_repaired(self):
        g=generation(self.c,'484','No usable answer');k.validate_generation(self.c,g)
        u=k.make_unit(self.c,'484',g);self.assertEqual(u['pred'],'');self.assertEqual(u['new_calls'],1)

    def test_reused_raw_and_metadata_unchanged(self):
        u=k.make_unit(self.c,'549');old=self.c['noobs']['units']['549']
        self.assertEqual(u['raw_text'],old['raw_text']);self.assertEqual(u['pred'],old['pred']);self.assertEqual(u['input']['tensor_sha'],old['tensor_sha']);self.assertEqual(u['new_calls'],0)

    def test_reused_answer_counter_or_origin_tamper_rejected(self):
        for name,value in [('raw_text','B'),('new_calls',1),('origin','new_generation')]:
            u=p.unseal(k.make_unit(self.c,'549'));u[name]=value
            with self.assertRaises(ValueError):k.validate_unit(self.c,p.seal(u))

    def test_generation_record_tamper_rejected(self):
        g=generation(self.c,'484');u=p.unseal(k.make_unit(self.c,'484',g));u['generation_record_sha']='changed'
        with self.assertRaises(ValueError):k.validate_unit(self.c,p.seal(u),g)

    def test_full_tensor_sha_shape_guard(self):
        g=p.unseal(generation(self.c,'484'));g['input']['tensor_sha']='short'
        with self.assertRaises(ValueError):k.validate_generation(self.c,p.seal(g))

    def test_nonfinite_cost_rejected(self):
        g=p.unseal(generation(self.c,'484'));g['inference_s']=-1
        with self.assertRaises(ValueError):k.validate_generation(self.c,p.seal(g))


class Runtime(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.scope=patch.object(r,'OUT',self.root);self.scope.start();self.addCleanup(self.scope.stop)
        self.guard=patch.object(p,'disk_guard',return_value={});self.guard.start();self.addCleanup(self.guard.stop)
        out=contextlib.redirect_stdout(io.StringIO());out.__enter__();self.addCleanup(out.__exit__,None,None,None)
        err=contextlib.redirect_stderr(io.StringIO());err.__enter__();self.addCleanup(err.__exit__,None,None,None)
    def run_fake(self,c,answers=None):
        calls=[];factories=[]
        class Fake:
            def __init__(self,ctx):factories.append(1)
            def generate(self,ctx,i):calls.append(i);return generation(ctx,i,(answers or {}).get(i,'A'))
        r.run(c,Fake)
        return calls,factories

    def reload(self,c):
        fresh=deepcopy(c);fresh['units']={};fresh['generations']={}
        for i in IDS:
            if r.generation_path(i).exists():fresh['generations'][i]=k.validate_generation(fresh,json.loads(r.generation_path(i).read_text()))
        for i in IDS:
            if r.unit_path(i).exists():fresh['units'][i]=k.validate_unit(fresh,json.loads(r.unit_path(i).read_text()),fresh['generations'].get(i))
        return fresh

    def test_only_seven_new_generations_and_no_model_on_complete(self):
        c=fixture()[0];calls,loads=self.run_fake(c)
        self.assertEqual(set(calls),set(c['protocol']['groups']['candidate']));self.assertEqual(len(calls),7);self.assertEqual(len(loads),1)
        self.assertEqual(len(c['units']),30);self.assertEqual(sum(u['origin']=='reused_noobs' for u in c['units'].values()),23)
        resumed=self.reload(c);calls2,loads2=self.run_fake(resumed);self.assertEqual(calls2,[]);self.assertEqual(loads2,[])

    def test_missing_audits_blocks_before_loading_model(self):
        c=fixture(audited=False)[0]
        with self.assertRaises(ValueError):r.run(c,lambda _:(_ for _ in ()).throw(AssertionError('must not load')))

    def test_crash_after_generation_uses_saved_output_on_resume(self):
        c=fixture()[0];first=next(i for i in IDS if c['plans'][i]['generate']);original=r.materialize;calls=[]
        class Fake:
            def __init__(self,ctx):pass
            def generate(self,ctx,i):calls.append(i);return generation(ctx,i)
        def interrupted(ctx,i):
            if i==first and i in ctx['generations']:raise RuntimeError('simulated interruption after generation save')
            return original(ctx,i)
        with patch.object(r,'materialize',side_effect=interrupted),self.assertRaises(RuntimeError):r.run(c,Fake)
        self.assertTrue(r.generation_path(first).exists());self.assertFalse(r.unit_path(first).exists())
        resumed=self.reload(c);r.run(resumed,Fake)
        self.assertEqual(len(calls),7);self.assertEqual(calls.count(first),1);k.require_complete(resumed)

    def test_all_saved_generations_missing_records_never_load_model(self):
        c=fixture()[0]
        for i in IDS:
            if c['plans'][i]['generate']:
                g=generation(c,i);write_once(r.generation_path(i),g);c['generations'][i]=g
        calls,loads=self.run_fake(c);self.assertEqual(calls,[]);self.assertEqual(loads,[]);k.require_complete(c)

    def test_audit_resume_only_unfinished(self):
        c=fixture(audited=False)[0];calls=[]
        class Fake:
            def __init__(self,ctx):pass
            def inspect(self,ctx,i):calls.append(i);return audit(ctx,i)
        r.audit_inputs(c,Fake);self.assertEqual(len(calls),30)
        r.audit_inputs(c,lambda _:(_ for _ in ()).throw(AssertionError('must skip processor')));self.assertEqual(len(calls),30)

    def test_audit_failure_does_not_save_bad_record(self):
        c=fixture(audited=False)[0]
        class Wrong:
            def __init__(self,ctx):pass
            def inspect(self,ctx,i):
                u=p.unseal(audit(ctx,i));u['input']['tensor_sha']='e'*64;return p.seal(u)
        with self.assertRaises(RuntimeError):r.audit_inputs(c,Wrong)
        self.assertFalse(list((r.OUT/'audits').glob('*.json')))
        self.assertEqual(len(list((r.OUT/'failures').glob('*.json'))),1)

    def test_failed_generation_does_not_fall_back_to_original_answer(self):
        c=fixture()[0]
        class Broken:
            def __init__(self,ctx):pass
            def generate(self,ctx,i):raise RuntimeError('simulated model failure')
        with self.assertRaises(RuntimeError):r.run(c,Broken)
        self.assertEqual(len(c['units']),23);self.assertEqual(c['generations'],{})
        self.assertEqual(len(list((r.OUT/'failures').glob('*.json'))),1)

    def test_invalid_answer_not_retried(self):
        c=fixture()[0];calls,_=self.run_fake(c,{'484':'No usable answer'})
        self.assertEqual(c['units']['484']['pred'],'');self.assertEqual(calls.count('484'),1)
        calls2,_=self.run_fake(self.reload(c));self.assertEqual(calls2,[])


# 使用独立类复用fixture工具，而非重复执行Runtime已有用例。
class Analysis(unittest.TestCase):
    setUp=Runtime.setUp
    run_fake=Runtime.run_fake
    def test_no_label_read_before_complete(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('no file read')):
            with self.assertRaises(ValueError):r.read_labels(dict(units={}))

    def test_score_permission_required_even_when_complete(self):
        c=fixture()[0];self.run_fake(c)
        with patch.dict(r.ACCESS,{'may_score':False}),patch.object(Path,'read_text',side_effect=AssertionError('no labels')):
            with self.assertRaises(ValueError):r.read_labels(c)

    def test_full30_rescue_harm_and_history_separated(self):
        c=fixture(uniform_overrides={'379':'B'})[0]
        labels={i:'A' for i in IDS};labels['484']='B'
        self.run_fake(c,{'484':'B','325':'B'})
        report,details=a.analyze(c,labels,a.legacy_comparison())
        self.assertEqual(report['overall']['N'],30);self.assertEqual(len(details),30)
        self.assertEqual(report['overall']['versus_noobs']['rescue'],1);self.assertEqual(report['overall']['versus_noobs']['harm'],1);self.assertEqual(report['overall']['versus_noobs']['net'],0)
        self.assertEqual(report['overall']['historical_noobs_versus_uniform']['net'],1)
        self.assertEqual(report['overall']['versus_uniform']['net'],1)
        self.assertEqual(report['cohorts']['prior12']['N'],12);self.assertEqual(report['cohorts']['extra18']['N'],18)
        self.assertEqual(report['execution']['changed_inputs_by_cohort'],{'prior12':4,'extra18':3})
        self.assertEqual(report['new_stage_cost']['reused_answers_source_recorded_inference_s'],11.5)
        for name in ['anchor_only','fallback']:self.assertEqual(report['by_scope'][name]['versus_noobs']['net'],0)

    def test_invalid_answer_counted_wrong_with_full_denominator(self):
        c=fixture()[0];self.run_fake(c,{'484':'No usable answer'});labels={i:'A' for i in IDS}
        report,details=a.analyze(c,labels,a.legacy_comparison())
        self.assertFalse(report['audit']['passed']);self.assertEqual(report['overall']['N'],30)
        self.assertEqual(report['overall']['methods']['frozen_leaf7_fallback23']['correct'],29)
        self.assertEqual(report['overall']['versus_noobs']['harm'],1);self.assertEqual(len(details),30)

    def test_missing_label_not_dropped(self):
        c=fixture()[0];self.run_fake(c)
        with self.assertRaises(ValueError):a.analyze(c,{},a.legacy_comparison())


class InputAndAccess(unittest.TestCase):
    def test_full_tensor_fingerprint_detects_pixel_change_at_same_indices(self):
        import torch
        ids=[0,70,299]
        original=dict(input_ids=torch.arange(100).reshape(1,100),image_grid_thw=torch.tensor([[1,2,2]]*3),
                      patch_positions=torch.tensor([[i,h,w] for i in ids for h in range(2) for w in range(2)]),
                      pixel_values=torch.arange(48,dtype=torch.float32).reshape(12,4))
        use={n:x.clone() for n,x in original.items()}
        proc=type('Processor',(),dict(spatial_merge_size=2))()
        first=r.input_fingerprint(original,use,proc,ids)
        use['pixel_values'][0,0]+=1
        second=r.input_fingerprint(original,use,proc,ids)
        self.assertNotEqual(first['tensor_sha'],second['tensor_sha']);self.assertEqual(first['baseline_tensor_sha'],second['baseline_tensor_sha'])

    def test_native_frame_position_mismatch_is_rejected(self):
        import torch
        original=dict(input_ids=torch.arange(100).reshape(1,100),image_grid_thw=torch.tensor([[1,2,2]]),
                      patch_positions=torch.tensor([[0,0,0]]),pixel_values=torch.ones(1,4))
        proc=type('Processor',(),dict(spatial_merge_size=2))()
        with self.assertRaises(ValueError):r.input_fingerprint(original,original,proc,[70])

    def test_guard_blocks_raw_secret_label_and_network_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['.env','deepseek.env','evaluation_labels.json','units.jsonl']:(root/name).write_text('fixture only')
            code='''import sys,socket\nfrom pathlib import Path\nimport leaf_qa30_run as r\nr.access_guard()\nn=0\nfor name in ['.env','deepseek.env','evaluation_labels.json']:\n try:(Path(sys.argv[1])/name).read_text()\n except RuntimeError:n+=1\ntry:socket.getaddrinfo('must-not-resolve.invalid',443)\nexcept RuntimeError:n+=1\nassert n==4,n\nassert (Path(sys.argv[1])/'units.jsonl').read_text()=='fixture only'\n'''
            subprocess.run([sys.executable,'-B','-c',code,tmp],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
