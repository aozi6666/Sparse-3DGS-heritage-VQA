"""Lightweight environment validation. Never import torch/transformers/cv2."""
import ast
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
from local_assets import PROJECT, branch_guard, require, validate, save_report

def main():
    branch_guard()
    require(sys.version_info[:2]==(3,11),'请使用Python3.11')
    require(Path(sys.prefix).resolve()==(PROJECT/'.venv').resolve(),'请使用Project/.venv/bin/python')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    print('[1/5] 检查已安装包metadata，不导入模型运行库',flush=True)
    names=['torch','torchvision','transformers','accelerate','safetensors','huggingface_hub','tokenizers','numpy','pillow','opencv-python-headless','fastapi','uvicorn','python-multipart','pydantic','pydantic-settings','tqdm','pytest','httpx']
    packages={name:md.version(name) for name in names}
    subprocess.run([str(PROJECT/'.bootstrap/bin/uv'),'pip','check','--python',sys.executable],check=True)
    print('[2/5] 检查本地模型结构和软链接，不加载权重',flush=True)
    result=validate(PROJECT/'models/mage-vl')
    model=PROJECT/'models/mage-vl'
    classes={n.name for n in ast.walk(ast.parse((model/'modeling_mage_vl.py').read_text())) if isinstance(n,ast.ClassDef)}
    require('MageVLForConditionalGeneration' in classes,'模型类源码不符')
    print('[3/5] 检查ffmpeg/ffprobe',flush=True)
    tools={}
    for name in ('ffmpeg','ffprobe'):
        exe=PROJECT/'tools/bin'/name
        require(exe.is_file(),'工具缺失：'+str(exe))
        tools[name]=subprocess.check_output([str(exe),'-version'],text=True,timeout=10).splitlines()[0]
    print('[4/5] 检查Node/前端包版本',flush=True)
    tools['node']=subprocess.check_output([str(PROJECT/'.node/bin/node'),'--version'],text=True,timeout=10).strip()
    major,minor=map(int,tools['node'].lstrip('v').split('.')[:2]);require(major>22 or (major==22 and minor>=12),'Node版本过低')
    tools['npm']=json.loads((PROJECT/'.node/lib/node_modules/npm/package.json').read_text())['version']
    for name in ('react','react-dom','vite','typescript'):
        tools[name]=json.loads((PROJECT/'frontend/node_modules'/name/'package.json').read_text())['version']
    print('[5/5] 保存轻量检查报告',flush=True)
    require(not any(n in sys.modules for n in ('torch','torchvision','transformers','cv2')),'不应导入重型运行库')
    mem={}
    for name in ('memory.max','memory.current'):
        path=Path('/sys/fs/cgroup')/name
        if path.exists():mem[name]=path.read_text().strip()
    report=dict(passed=True,validation_level='lightweight',python=sys.version.split()[0],python_executable=sys.executable,
        packages=packages,tools=tools,model_structure=result,memory=mem,runtime_imports_tested=False,
        pretrained_weights_loaded=False,model_downloads=0,gpu_inference_tested=False,
        note='仅元数据/结构/工具检查通过。重模块导入、处理器执行和GPU推理均延后，不等于完整运行验收。')
    path=save_report(report,'environment_light');print('轻量检查通过；未导入模型运行库。报告：',path)
if __name__=='__main__':main()
