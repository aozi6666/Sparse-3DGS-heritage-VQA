"""Paired downstream QA. Run separately, after the frozen B/C experiment finishes."""
import collections
import fcntl
import gzip
import importlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import prepare as prep
from tqdm import tqdm

HERE = prep.HERE
sys.path.insert(0, str(prep.V/'aks_mage_subtitle_bc'))
import bc


def main():
    import cv2
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    src = sqlite3.connect(prep.BC.as_uri()+'?mode=ro', uri=True)
    try:
        audits = {i:json.loads(s) for i,s in src.execute('select * from audit')}
        ref = {i:json.loads(s) for i,s in src.execute("select id,data from qa where arm='B'")}
    finally:
        src.close()
    if len(ref)!=714:
        raise SystemExit('Wait for the paired B/C experiment to finish before starting GPU QA.')
    audits = {i:a for i,a in audits.items() if a['duration']=='long'}
    ref = {i:ref[i] for i in audits}
    assert len(ref)==237 and sum(ref[i]['pred']==a['gt'] for i,a in audits.items())==166
    folders = list((HERE/'results').glob('*/summary.json'))
    if len(folders)!=1:
        raise SystemExit('Expected one completed CPU audit version.')
    data = folders[0].parent
    manifest = {a['id']:bc.sha(data/(a['id']+'.json.gz')) for a in audits.values()}
    version = bc.signature([manifest, ref, data.name, bc.sha(Path(__file__)), bc.sha(prep.V/'aks_mage_subtitle_bc/bc.py')])[:16]
    out = data/'qa'/version
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    db = sqlite3.connect(out/'checkpoint.sqlite')
    db.execute('create table if not exists qa(idx text,arm text,data text,primary key(idx,arm))')
    done = {(i,a):json.loads(s) for i,a,s in db.execute('select * from qa')}
    groups = collections.defaultdict(list)
    for r in bc.read(prep.BASE/'plan.json')['rows']:
        if r['index'] in audits:
            groups[r['video_id']].append(r)
    engine = None
    with tqdm(total=474,initial=len(done),desc='Subtitle paired QA') as bar:
        for vid, rows in groups.items():
            cap = None
            try:
                for row in rows:
                    idx = row['index']
                    a = audits[idx]
                    with gzip.open(data/(a['id']+'.json.gz'),'rt') as f:
                        inputs = json.load(f)
                    assert inputs['visual_hash']==a['ids_hash'] and inputs['b_input_ref']==idx
                    fused = None
                    for arm in ('time','hybrid'):
                        if (idx,arm) in done:
                            continue
                        sub = inputs[arm]['text']
                        prompt = bc.subtitle_prompt(row,sub)
                        digest = bc.signature(prompt)
                        if bc.signature(sub) == a['b_hash']:
                            assert digest == ref[idx]['prompt_hash']
                            ans = dict(pred=ref[idx]['pred'],raw=ref[idx]['raw'],origin='same_B')
                        elif arm=='hybrid' and sub==inputs['time']['text']:
                            ans = dict(done[idx,'time'],origin='same_time')
                        else:
                            if engine is None:
                                proc = AutoProcessor.from_pretrained(str(bc.MODEL),trust_remote_code=True,local_files_only=True)
                                proc.image_processor.max_pixels=150000
                                proc.image_processor.min_pixels=3136
                                model = AutoModelForCausalLM.from_pretrained(str(bc.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0}).eval()
                                codec = importlib.import_module('.codec_video_processing_mage_vl',type(proc).__module__.rsplit('.',1)[0])
                                engine = model
                            if cap is None:
                                v = bc.read(prep.BASE/'video_records'/f'{vid}.json')['video']
                                for path,checksum in v['cache_files_sha256'].items():
                                    assert bc.sha(bc.local(path))==checksum
                                directory = bc.local(v['cache_dir'])
                                native = codec._load_codec_result(directory)
                                video = bc.local(v['video_path'])
                                cap = cv2.VideoCapture(str(video))
                            if fused is None:
                                fused = bc.render(native,cap,dict(zip(a['slots'],a['frames'])))
                            chat = proc.apply_chat_template([dict(role='user',content=[dict(type='video'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True)
                            with patch.object(codec,'process_codec_video',return_value=fused) as loader:
                                x = proc(text=[chat],videos=[str(video)],video_backend='codec',max_pixels=150000,codec_config=dict(bc.old.CODEC,cache_root=directory.parent),return_tensors='pt',padding=True)
                            loader.assert_called_once()
                            assert bc.signature(bc.tids(x['patch_positions'].numpy()))==a['ids_hash']
                            gpu = {k:t.to(model.device,dtype=model.dtype) if k=='pixel_values' else t.to(model.device) for k,t in x.items()}
                            with torch.inference_mode():
                                y = model.generate(**gpu,max_new_tokens=32,do_sample=False)
                            raw = proc.tokenizer.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                            ans = dict(raw=raw,pred=bc.parse_answer(raw),origin='generated')
                            del x,gpu,y
                        ans.update(prompt_hash=digest,tokens=inputs[arm]['tokens'])
                        with db:
                            db.execute('insert into qa values(?,?,?)',(idx,arm,json.dumps(ans)))
                        done[idx,arm]=ans
                        bar.update()
            finally:
                if cap is not None:
                    cap.release()
    report = {}
    for duration in ('long',):
        keys = [i for i,a in audits.items() if duration=='overall' or a['duration']==duration]
        def ok(i,k):
            pred = ref[i]['pred'] if k=='B' else done[i,k]['pred']
            return pred==audits[i]['gt']
        report[duration] = dict(n=len(keys),correct={k:sum(ok(i,k) for i in keys) for k in ('B','time','hybrid')})
        old_only = [i for i in keys if audits[i]['A']==audits[i]['gt'] and audits[i]['K20']!=audits[i]['gt']]
        visual_only = [i for i in keys if audits[i]['A']!=audits[i]['gt'] and audits[i]['K20']==audits[i]['gt']]
        for label, subset in [('old_subtitle_only',old_only),('visual_only',visual_only)]:
            report[duration][label] = dict(n=len(subset),correct={k:sum(ok(i,k) for i in subset) for k in ('B','time','hybrid')})
        for test,base in [('time','B'),('hybrid','B'),('hybrid','time')]:
            report[duration][test+'_vs_'+base] = dict(gains=sum(ok(i,test) and not ok(i,base) for i in keys),losses=sum(ok(i,base) and not ok(i,test) for i in keys))
    bc.atom(out/'summary.json',report)
    print(out, report)


if __name__ == '__main__':
    main()
