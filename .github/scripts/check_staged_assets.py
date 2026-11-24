#!/usr/bin/env python3
"""Reject heavyweight/runtime assets and obvious credentials from the Git index."""
import subprocess,re,sys
entries=[]
for item in subprocess.check_output(['git','ls-files','--stage','-z']).decode().split('\0'):
    if not item:continue
    meta,path=item.split('\t',1);mode,oid,stage=meta.split()
    if mode!='160000':entries.append((path,oid))
proc=subprocess.run(['git','cat-file','--batch-check=%(objectname) %(objectsize)'],
    input='\n'.join(h for _,h in entries)+'\n',text=True,capture_output=True,check=True)
sizes={a.split()[0]:int(a.split()[1]) for a in proc.stdout.splitlines()}
errors=[]
for path,oid in entries:
    if sizes[oid]>5*1024**2:errors.append((path,'blob exceeds 5 MiB; use a documented external artifact'))
    if re.search(r'\.(pt|pth|ckpt|safetensors|npy|npz|ply|glb|mp4|pkl|pickle|zip)$',path,re.I):errors.append((path,'weight/data/video/archive payload'))
    if path.endswith(('/.env','deepseek.env')) or '/.venv' in path:errors.append((path,'environment or credential path'))
changed=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z']).decode().split('\0')
mapping=dict(entries)
for path in filter(None,changed):
    if path not in mapping or sizes[mapping[path]]>2_000_000:continue
    if not path.endswith(('.py','.md','.sh','.json','.yaml','.txt','.env')):continue
    text=subprocess.check_output(['git','cat-file','blob',mapping[path]]).decode('utf8',errors='replace')
    if re.search(r'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----)',text):
        errors.append((path,'potential credential; value intentionally not printed'))
if errors:
    for path,reason in errors:print(path+': '+reason)
    sys.exit(1)
print(f'Index asset check passed: {len(entries)} blobs, no oversized weights/data/video or detected private keys/tokens.')
