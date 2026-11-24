"""Paired RGB canvas + frozen/re-aligned SRT experiment."""
import argparse, collections, csv, fcntl, hashlib, importlib, json, os, sqlite3, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
V=HERE.parent
P=V/'mage_vl/practical'
LEG=V/'mage_vl/codec_full714_32768'
F=V/'aks_mage_budget_final'
A=V/'eval_results/practical_mage_v1/24_codec_full714_32768_v1'
H=F/'results/eab255d8130df5b9/predictions.jsonl'
sys.path[:0]=[str(LEG),str(P),str(F)]
import codec_full714 as old
from subtitle_common import subtitle_source,subtitle_prompt
from practical_common import MODEL,signature,parse_answer,sha
from continuous_common import continuous_evidence
from render import render
from sampling import plans

def read(p):return json.loads(p.read_text())
def local(p):return Path(p.replace('/VQA/','/VLM/'))
def atom(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
def tids(pos):
    ids=np.unique(pos[...,0]);return ids[ids>=0].astype(int).tolist()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    source=read(A/'plan.json')
    rows=source['rows']
    qmap={(q['video'],q['question']):q for q in read(V/'aks_mage/questions.json')}
    fixed={r['id']:r for r in map(json.loads,H.read_text().splitlines()) if r['budget']==20}
    anchors={r['id']:r['anchors'] for r in map(json.loads,(V/'aks_mage/runs/cd294b5855bcae9c.jsonl').read_text().splitlines())}
    ranks=read(F/'candidates.json')
    refs={r['index']:r for r in csv.DictReader((A/'results/full714/predictions.csv').open())}
    groups=collections.defaultdict(list)
    for row in rows:groups[row['video_id']].append(row)
    assert len(rows)==len(fixed)==len(refs)==714 and len(groups)==238
    dependencies=[Path(__file__),H,A/'plan.json',A/'protocol.json',A/'results/full714/predictions.csv',LEG/'codec_full714.py',P/'continuous_common.py',P/'subtitle_common.py',P/'practical_common.py',F/'sampling.py',F/'render.py',F/'candidates.json',MODEL/'tokenizer.json']
    h=signature({str(p):sha(p) for p in dependencies})[:16]
    out=HERE/'results'/h;out.mkdir(parents=True,exist_ok=True)
    print('Output:',out,flush=True)
    lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(out/'checkpoint.sqlite')
    db.executescript('PRAGMA synchronous=FULL;CREATE TABLE IF NOT EXISTS audit(id TEXT PRIMARY KEY,data TEXT);CREATE TABLE IF NOT EXISTS qa(id TEXT,arm TEXT,data TEXT,PRIMARY KEY(id,arm));')
    audits={i:json.loads(s) for i,s in db.execute('SELECT * FROM audit')}
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    count=lambda s:len(tok.encode(s,add_special_tokens=False).ids)
    with tqdm(total=714,initial=len(audits),desc='CPU subtitle audit') as bar:
        for vid,group in groups.items():
            if all(r['index'] in audits for r in group):continue
            v=read(A/'video_records'/f'{vid}.json')['video']
            directory=local(v['cache_dir'])
            path=directory/'src_patch_position.npy'
            assert sha(path)==next(d for p,d in v['cache_files_sha256'].items() if Path(p).name==path.name)
            pos=np.load(path)
            native=dict(images=v['canvas_files'],src_positions=pos)
            sections=pos.reshape(len(v['canvas_files']),-1,3)
            src,cues=subtitle_source(vid)
            assert src['sha256']==source['srt_sources'][vid]['sha256'] and tids(pos)==v['source_indices']
            frozen=v['evidence'][old.PRIMARY]['continuous']
            def evidence(ids):
                aligned=old.align(ids,v['fps'],cues,src,count)
                text=continuous_evidence(aligned,cues,count,32768)
                assert not text['omitted_neighbor_ids'] and text['text_tokens']<=32768
                ids={c['cue_id'] for c in aligned['selected_cues']}|set(text['added_cue_ids'])
                return text,ids
            b,b_ids=evidence(tids(pos))
            assert b['text']==frozen['text']
            for row in group:
                idx=row['index']
                if idx in audits:continue
                q=qmap[vid,row['question']];k=q['id'];f=fixed[k];ref=refs[idx]
                assert ref['gt']==f['answer'] and ref['video_id']==vid
                layout=plans(native,ranks[k],anchors[k])[20]
                assert list(layout.items())==list(zip(f['slots'],f['frames']))
                times=np.concatenate([np.asarray([layout[i]]) if i in layout else s[:,0] for i,s in enumerate(sections)])
                ids=sorted(set(map(int,times[times>=0])))
                c,c_ids=evidence(ids)
                old_unit=read(A/'units'/f'{idx}_{old.PRIMARY}.json')
                assert signature(subtitle_prompt(row,b['text']))==old_unit['prompt_sha256']
                rec=dict(index=idx,id=k,video=vid,duration=row['duration'],gt=ref['gt'],A=ref[old.PRIMARY],K20=f['pred'],slots=f['slots'],frames=f['frames'],
                         ids_hash=signature(ids),b_hash=signature(b['text']),c_text=c['text'],same=b['text']==c['text'],b_tokens=b['text_tokens'],c_tokens=c['text_tokens'],
                         added=sorted(c_ids-b_ids),removed=sorted(b_ids-c_ids),b_status=b['status'],c_status=c['status'],srt_status=src['status'])
                with db:db.execute('INSERT INTO audit VALUES(?,?)',(idx,json.dumps(rec)))
                audits[idx]=rec;bar.update()
    assert len(audits)==714
    audit_rows=list(audits.values())
    for which,n in [('A',521),('K20',485)]:assert sum(x[which]==x['gt'] for x in audit_rows)==n
    def subsets(x):
        return {name:[v for v in x if name=='overall' or v['duration']==name] for name in ['overall','short','medium','long']}
    atom(out/'audit_summary.json',{d:dict(n=len(g),identical=sum(x['same'] for x in g),added_cues=sum(len(x['added']) for x in g),removed_cues=sum(len(x['removed']) for x in g),
            mean_b_tokens=np.mean([x['b_tokens'] for x in g]),mean_c_tokens=np.mean([x['c_tokens'] for x in g]),zero_c_reasons=dict(collections.Counter(x['c_status'] for x in g if not x['c_tokens']))) for d,g in subsets(audit_rows).items()})
    if args.audit_only:return
    done={(i,a):json.loads(s) for i,a,s in db.execute('SELECT * FROM qa')}
    if len(done)<1428:
        import cv2,torch
        from transformers import AutoModelForCausalLM,AutoProcessor
        if not torch.cuda.is_available():raise SystemExit('CPU audit complete; GPU required for paired QA.')
        proc=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
        proc.image_processor.max_pixels=150000;proc.image_processor.min_pixels=3136
        model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0}).eval()
        codec=importlib.import_module('.codec_video_processing_mage_vl',type(proc).__module__.rsplit('.',1)[0])
        with tqdm(total=1428,initial=len(done),desc='Paired B/C QA') as bar:
            for vid,group in groups.items():
                if all((r['index'],a) in done for r in group for a in 'BC'):continue
                v=read(A/'video_records'/f'{vid}.json')['video']
                directory=local(v['cache_dir'])
                for f,digest in v['cache_files_sha256'].items():assert sha(local(f))==digest,f
                native=codec._load_codec_result(directory)
                video=local(v['video_path']);cap=cv2.VideoCapture(str(video))
                try:
                    for row in group:
                        idx=row['index'];r=audits[idx]
                        if all((idx,a) in done for a in 'BC'):continue
                        fused=render(native,cap,dict(zip(r['slots'],r['frames'])))
                        assert signature(tids(fused['src_positions']))==r['ids_hash']
                        bt=v['evidence'][old.PRIMARY]['continuous']['text']
                        assert signature(bt)==r['b_hash']
                        for arm,sub in [('B',bt),('C',r['c_text'])]:
                            if (idx,arm) in done:continue
                            if arm=='C' and bt==sub:
                                ans=dict(done[idx,'B'],arm='C',origin='identical_B_input')
                            else:
                                prompt=subtitle_prompt(row,sub)
                                chat=proc.apply_chat_template([dict(role='user',content=[dict(type='video'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True)
                                with patch.object(codec,'process_codec_video',return_value=fused) as loader:
                                    x=proc(text=[chat],videos=[str(video)],video_backend='codec',max_pixels=150000,codec_config=dict(old.CODEC,cache_root=directory.parent),return_tensors='pt',padding=True)
                                loader.assert_called_once()
                                assert signature(tids(x['patch_positions'].numpy()))==r['ids_hash']
                                gpu={k:t.to(model.device,dtype=model.dtype) if k=='pixel_values' else t.to(model.device) for k,t in x.items()}
                                with torch.inference_mode():y=model.generate(**gpu,max_new_tokens=32,do_sample=False)
                                raw=proc.tokenizer.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                                ans=dict(id=r['id'],arm=arm,pred=parse_answer(raw),raw=raw,origin='generated',input_tokens=x['input_ids'].shape[1],prompt_hash=signature(prompt))
                                del x,gpu,y
                            with db:db.execute('INSERT INTO qa VALUES(?,?,?)',(idx,arm,json.dumps(ans)))
                            done[idx,arm]=ans;bar.update()
                finally:cap.release()
    for r in audit_rows:
        for a in 'BC':r[a]=done[r['index'],a]['pred']
    metrics={}
    for d,g in subsets(audit_rows).items():
        metrics[d]={}
        for arm in 'BC':
            ok=lambda x,k:x[k]==x['gt']
            metrics[d][arm]=dict(n=len(g),correct=sum(ok(x,arm) for x in g),accuracy=sum(ok(x,arm) for x in g)/len(g),
                gains_vs_A=sum(ok(x,arm) and not ok(x,'A') for x in g),losses_vs_A=sum(ok(x,'A') and not ok(x,arm) for x in g),
                kept_from_A_only=sum(ok(x,'A') and not ok(x,'K20') and ok(x,arm) for x in g),
                recovered_from_visual_only=sum(ok(x,'K20') and not ok(x,'A') and ok(x,arm) for x in g))
        metrics[d]['paired']=dict(C_only=sum(ok(x,'C') and not ok(x,'B') for x in g),B_only=sum(ok(x,'B') and not ok(x,'C') for x in g))
    atom(out/'summary.json',metrics)
    atom(out/'predictions.json',[{k:x[k] for k in ('id','video','duration','gt','A','K20','B','C','same','b_tokens','c_tokens')} for x in audit_rows])
    print(json.dumps(metrics['overall']),flush=True)
if __name__=='__main__':main()
