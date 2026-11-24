#!/usr/bin/env python3
"""只开放旧开发池smoke3/dev30；用户启动run，CPU检查不加载模型。"""
import argparse,fcntl,importlib.util,json,os
from air_protocol import ROOT,OLD,disk_guard,context,write_once
GPU_LOCK=OLD/'cache/gpu.lock'

def preflight(args,ctx):
    print('SPACE',disk_guard(args.min_free,args.budget),flush=True)
    missing=[n for n in ['torch','transformers','numpy','scipy','sklearn','joblib','threadpoolctl','narwhals','cloudpickle','decord','PIL','torchvision','tqdm'] if importlib.util.find_spec(n) is None]
    if missing:raise ImportError('缺少'+','.join(missing)+'；先运行依赖准备命令，不自动安装/下载。')
    print(f'{args.stage}: {len(ctx["expected"]&set(ctx["units"]))}/{len(ctx["expected"])}逻辑答案；开发覆盖评测，非新独立测试。',flush=True)

def run(args):
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户设置PRACTICAL_STAGE_CONFIRMED=1后启动')
    ROOT.mkdir(parents=True,exist_ok=True)
    GPU_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctx=context(args.stage);preflight(args,ctx)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],indices=[r['index'] for r in ctx['rows']],store=str(ctx['store'])))
        if ctx['expected'].issubset(ctx['units']):print('全部答案已完成，跳过模型加载；请summarize。');return
        from air_runtime import run_gpu
        run_gpu(args,ctx)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','preflight','run','summarize'])
    p.add_argument('--stage',choices=['smoke3','dev30'],default='smoke3');p.add_argument('--min-free',type=float,default=10);p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0:p.error('磁盘预算无效')
    print('SPACE',disk_guard(args.min_free,args.budget))
    if args.action=='run':run(args);return
    ctx=context(args.stage,initialize=args.action=='init')
    if args.action=='preflight':preflight(args,ctx)
    elif args.action=='summarize':
        from air_analyze import summarize
        summarize(ctx)
    else:print('CPU清单锁定，未解码/加载模型/推理。签名:',ctx['signature'])

if __name__=='__main__':main()
