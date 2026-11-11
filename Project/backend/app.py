"""Local Mage demo. Import/startup never imports torch or loads weights."""
import asyncio
import re
from contextlib import asynccontextmanager, contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Literal
from datetime import datetime, timezone
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = dict(backend='frames', frames=16, max_pixels=150000, max_new_tokens=160, prompt_version=4, timestamps='source_seconds')

def asset_identity():
    # Hash small configuration/source files only. Never read weight payloads.
    model = ROOT/'models/mage-vl'
    result = {'path': str(model.resolve()), 'files': {}}
    if model.is_dir():
        for path in sorted(model.iterdir()):
            if path.suffix in ('.json', '.py', '.jinja'):
                result['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif path.suffix == '.safetensors':
                stat = path.stat()
                result['files'][path.name] = [stat.st_size, stat.st_mtime_ns]
    return result



def generation_prompt(start, end, question, language):
    if language == 'en':
        return (f'These frames cover {start:.2f} to {end:.2f} seconds of the video, in chronological order. '
                'Answer in concise English. Use only visible evidence; do not invent audio or background facts.\n'+question)
    return (f'这些画面来自视频{start:.2f}至{end:.2f}秒，按时间排列。'
            '请用简洁中文回答，只依据可见画面，不编造声音或背景事实。\n'+question)


def commentary_question(language):
    return ('Describe ONE main visible event in ONE complete English sentence, 12 to 18 words. No introduction, repetition, list, or background speculation.'
            if language == 'en' else '只描述一个最主要的可见事件，输出一句完整中文短句，控制在20至30字；不要开场白、重复、列举或推测。')

def narration_too_long(text,language):
    return len(re.findall(r"\S+",text))>18 if language=='en' else len(re.sub(r"\s","",text))>30


class TimestampedFrames:
    """Keep source timestamps: the native processor labels decoded PIL frames 0,1,... seconds."""
    def __init__(self, processor, seconds):
        self.processor=processor
        self.seconds=seconds
    def __getattr__(self, name):
        return getattr(self.processor,name)
    def __call__(self, *args, **kwargs):
        result=self.processor(*args,**kwargs)
        if len(result['frame_timestamps'])!=1 or len(result['frame_timestamps'][0])!=len(self.seconds):
            raise RuntimeError('视频帧与时间戳数目不一致')
        result['frame_timestamps']=[self.seconds]
        return result

class MageEngine:
    def __init__(self):
        self.model = self.proc = None
        self.blocked = False
        self.lock = threading.Lock()
    def generate(self, path, start, end, question, stop, language="zh"):
        with self.lock:
            if stop.is_set(): return
            if self.blocked: raise RuntimeError("前一生成线程未正常退出，需要重启服务后再试")
            if os.getenv('DEMO_ENABLE_MODEL') != '1':
                raise RuntimeError('本地模型未启用；请在GPU可用后显式设置DEMO_ENABLE_MODEL=1。')
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_MODULES_CACHE=str(ROOT/'storage/hf_modules'))
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('未发现可用GPU；不回退CPU推理，不下载模型。')
            from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
            checkpoint = ROOT/'models/mage-vl'
            if not checkpoint.is_dir(): raise RuntimeError('本地模型目录不可用')
            if self.model is None:
                self.proc = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
                self.proc.image_processor.max_pixels = 150000
                self.proc.image_processor.min_pixels = 3136
                self.proc.video_processor.max_pixels = 150000
                self.proc.video_processor.min_pixels = 3136
                self.model = AutoModelForCausalLM.from_pretrained(str(checkpoint), trust_remote_code=True,
                    local_files_only=True, torch_dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(str(path)); frames=[]
            try:
                fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps<=0 or total<=0: raise RuntimeError('视频无有效帧率')
                first=min(total-1,max(0,int(start*fps))); last=min(total-1,max(first,int(end*fps)-1))
                ids=sorted({round(first+(last-first)*i/15) for i in range(16)})
                for fid in ids:
                    if stop.is_set(): return
                    cap.set(cv2.CAP_PROP_POS_FRAMES,fid); ok,frame=cap.read()
                    if not ok: raise RuntimeError(f'无法解码原帧{fid}')
                    height,width=frame.shape[:2]
                    scale=min(1.,(150000/(height*width))**.5)
                    if scale<1: frame=cv2.resize(frame,(max(1,int(width*scale)),max(1,int(height*scale))))
                    frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            finally: cap.release()
            prompt=generation_prompt(start,end,question,language)
            chat=self.proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
            original=self.proc.video_processor
            self.proc.video_processor=TimestampedFrames(original,[fid/fps for fid in ids])
            try:
                values=self.proc(text=[chat],videos=[frames],video_backend='frames',return_tensors='pt',padding=True)
            finally: self.proc.video_processor=original
            values={k:v.to(self.model.device,dtype=self.model.dtype) if k=='pixel_values' else v.to(self.model.device) for k,v in values.items()}
            streamer=TextIteratorStreamer(self.proc.tokenizer,skip_prompt=True,skip_special_tokens=True,timeout=120)
            errors=[]
            class Cancel(StoppingCriteria):
                def __call__(self,*args,**kwargs): return stop.is_set()
            def work():
                try:
                    with torch.inference_mode():
                        self.model.generate(**values,max_new_tokens=160,do_sample=False,streamer=streamer,stopping_criteria=StoppingCriteriaList([Cancel()]))
                except Exception as exc: errors.append(exc); streamer.end()
            thread=threading.Thread(target=work,daemon=True);thread.start()
            finished=False
            try:
                for chunk in streamer:
                    if stop.is_set(): break
                    yield chunk
                finished=True
            finally:
                if not finished: stop.set()
                thread.join(timeout=125)
                if thread.is_alive(): self.blocked=True
            if self.blocked:
                raise RuntimeError('生成线程未结束，请重启本服务后检查')
            if errors: raise RuntimeError(str(errors[0]))

class Question(BaseModel):
    video_id:str
    timestamp:float=Field(ge=0)
    question:str=Field(min_length=1,max_length=2000)
    language:Literal["zh","en"]="zh"
class JobSpec(BaseModel):
    video_id:str
    segment_seconds:float=Field(default=8,ge=4,le=30)
    language:Literal["zh","en"]="zh"
class Playback(BaseModel):
    timestamp:float=Field(ge=0)
    playing:bool

class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'demo.sqlite3'
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS objects (kind TEXT,id TEXT,data TEXT,PRIMARY KEY(kind,id))')
            for row in c.execute("SELECT id,data FROM objects WHERE kind='job'").fetchall():
                j=json.loads(row[1])
                if j['status'] in ('running','queued'):
                    j.update(status='paused',error='服务已重启，可继续未完成片段')
                    c.execute("UPDATE objects SET data=? WHERE kind='job' AND id=?",(json.dumps(j,ensure_ascii=False),row[0]))
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=15)
        try:
            c.execute('PRAGMA journal_mode=WAL')
            with c: yield c
        finally: c.close()
    def put(self,kind,obj):
        with self.connect() as c:c.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?)',(kind,obj['id'],json.dumps(obj,ensure_ascii=False)))
    def get(self,kind,key):
        with self.connect() as c:r=c.execute('SELECT data FROM objects WHERE kind=? AND id=?',(kind,key)).fetchone()
        if not r:raise HTTPException(404,'记录不存在')
        return json.loads(r[0])

def event(name,obj):return f'event: {name}\ndata: {json.dumps(obj,ensure_ascii=False)}\n\n'

def create_app(storage=None,engine=None):
    controls={};workers={};question_runs={};mutex=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        yield
        for ctrl in list(controls.values())+list(question_runs.values()): ctrl['stop'].set()
    app=FastAPI(title='Mage Video Studio',version='0.1.0',lifespan=lifespan)
    store=Store(storage or ROOT/'storage');engine=engine or MageEngine()
    assets=asset_identity()
    def signature(video, seconds, language="zh"):
        data=dict(video=video['sha256'], segment=seconds, language=language, assets=assets, **PIPELINE)
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

    def active_video(key):
        video=store.get('video',key)
        if video.get('deleted_at'): raise HTTPException(404,'视频已移入回收目录，可撤销删除后恢复')
        return video
    def public_video(video):
        return {k:v for k,v in video.items() if k not in ('path','original_path')}

    enabled=lambda: not isinstance(engine,MageEngine) or os.getenv('DEMO_ENABLE_MODEL')=='1'
    def require_model():
        if not enabled():raise HTTPException(409,'模型未启用。当前可测试上传/播放；GPU可用后再启用真实推理。')
    def run_job(j):
        key=j['id'];stop=controls[key]['stop'];j.update(status='running',error=None);store.put('job',j)
        try:
            v=active_video(j['video_id'])
            for seg in j['segments']:
                if seg['status']=='done':continue
                while not stop.is_set():
                    ctrl=controls[key]
                    if ctrl['playing'] and seg['start']<=ctrl['timestamp']+j['segment_seconds']:break
                    time.sleep(.2)
                if stop.is_set():break
                seg.update(status='running',text='');store.put('job',j)
                updated=0
                for text in engine.generate(v['path'],seg['start'],seg['end'],commentary_question(j.get('language','zh')),stop,language=j.get('language','zh')):
                    seg['text']+=text
                    if time.monotonic()-updated>.15:store.put('job',j);updated=time.monotonic()
                    if stop.is_set():break
                if not stop.is_set() and narration_too_long(seg['text'],j.get('language','zh')):
                    original=seg['text'];seg['original_text']=original;seg['phase']='shortening';store.put('job',j)
                    instruction=(f'Rewrite this description into ONE complete English sentence of 12 to 18 words. Preserve only its main visible event; no new facts. Output only the sentence: {original}'
                        if j.get('language','zh')=='en' else f'将以下解说压缩为一句20至30字的完整中文短句，仅保留主要事件，不新增事实，只输出短句：{original}')
                    shortened=''.join(engine.generate(v['path'],seg['start'],seg['end'],instruction,stop,language=j.get('language','zh')))
                    if not stop.is_set() and shortened.strip() and len(shortened)<len(original):seg['text']=shortened
                    seg['length_target_met']=not narration_too_long(seg['text'],j.get('language','zh'));seg['phase']='ready'
                seg['status']='pending' if stop.is_set() else 'done';store.put('job',j)
            j['status']='paused' if stop.is_set() else 'done'
        except Exception as exc:j.update(status='error',error=str(exc))
        finally:store.put('job',j)
    def launch(j):
        with mutex:
            active_video(j['video_id'])
            for key,w in workers.items():
                if w.is_alive():raise HTTPException(409,'已有讲解任务运行，请先停止它')
            store.put('job',j)
            controls[j['id']]=dict(stop=threading.Event(),timestamp=0.,playing=True,video_id=j['video_id'])
            w=threading.Thread(target=run_job,args=(j,),daemon=True);workers[j['id']]=w;w.start()
    @app.get('/api/health')
    def health():return dict(ok=True,model_enabled=enabled(),model_loaded=getattr(engine,'model',None) is not None,model_local=bool((ROOT/'models/mage-vl').is_dir()),tts='browser',backend='frames',gpu_tested=False)
    @app.post('/api/videos')
    async def upload(video:UploadFile=File(...)):
        if Path(video.filename or '').suffix.lower() not in {'.mp4','.mov','.mkv','.webm','.avi','.m4v'}:raise HTTPException(400,'请选择视频文件')
        key=uuid.uuid4().hex;directory=store.root/'uploads';directory.mkdir(exist_ok=True)
        path=directory/(key+Path(video.filename).suffix.lower());size=0;h=hashlib.sha256()
        with path.open('xb') as out:
            while chunk:=await video.read(1024*1024):
                size+=len(chunk)
                if size>2*1024**3 or shutil.disk_usage(directory).free<5*1024**3:raise HTTPException(413,'文件过大或剩余空间不足；已接收部分保留，不自动清理')
                out.write(chunk);h.update(chunk)
        try:
            tool=ROOT/'tools/bin/ffprobe'
            r=await asyncio.to_thread(subprocess.run,[str(tool),'-v','error','-show_entries','format=duration','-of','json',str(path)],capture_output=True,text=True,timeout=30,check=True)
            duration=float(json.loads(r.stdout)['format']['duration'])
            if not 0<duration<24*3600:raise ValueError('时长无效')
        except Exception:raise HTTPException(400,'视频解析失败，原上传文件已保留')
        obj=dict(id=key,name=video.filename,duration=duration,bytes=size,sha256=h.hexdigest(),path=str(path),url=f'/api/videos/{key}/media')
        store.put('video',obj);return public_video(obj)
    @app.get('/api/videos/{key}')
    def video_metadata(key:str):
        return public_video(active_video(key))
    @app.get('/api/videos/{key}/media')
    def media(key:str):return FileResponse(active_video(key)['path'])
    @app.delete('/api/videos/{key}')
    def delete_video(key:str):
        # Reversible deletion: stop this video's work, rename on the same disk,
        # keep historical jobs and their text. Other videos/tasks are untouched.
        with mutex:
            video=store.get('video',key)
            if video.get('deleted_at'): return dict(ok=True,id=key,recoverable=True)
            related=[(workers[k],ctrl) for k,ctrl in controls.items() if ctrl['video_id']==key]
            questions=[ctrl for ctrl in list(question_runs.values()) if ctrl['video_id']==key]
            for _,ctrl in related: ctrl['stop'].set()
            for ctrl in questions: ctrl['stop'].set()
            deadline=time.monotonic()+2
            for worker,_ in related: worker.join(timeout=max(0.,deadline-time.monotonic()))
            for ctrl in questions: ctrl['done'].wait(timeout=max(0.,deadline-time.monotonic()))
            if any(worker.is_alive() for worker,_ in related) or any(not ctrl['done'].is_set() for ctrl in questions):
                raise HTTPException(409,'正在停止此视频的处理，请稍后再次点击删除；原文件尚未移动')
            source=Path(video['path'])
            destination=store.root/'trash'/key/source.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists(): raise HTTPException(409,'回收路径已存在，保留现场，请检查')
            if not source.is_file(): raise HTTPException(409,'原视频文件缺失，请检查存储目录')
            source.rename(destination)
            video.update(path=str(destination),original_path=str(source),deleted_at=datetime.now(timezone.utc).isoformat())
            try: store.put('video',video)
            except Exception:
                destination.rename(source)
                raise
            return dict(ok=True,id=key,recoverable=True)
    @app.post('/api/videos/{key}/restore')
    def restore_video(key:str):
        with mutex:
            video=store.get('video',key)
            if not video.get('deleted_at'): return public_video(video)
            source=Path(video['path']);destination=Path(video['original_path'])
            if not source.is_file() or destination.exists(): raise HTTPException(409,'恢复路径异常，文件保留，请检查')
            source.rename(destination)
            video.update(path=str(destination));video.pop('deleted_at');video.pop('original_path')
            try: store.put('video',video)
            except Exception:
                destination.rename(source)
                raise
            return public_video(video)
    @app.post('/api/jobs')
    def start(spec:JobSpec):
        require_model();v=active_video(spec.video_id)
        sig=signature(v,spec.segment_seconds,spec.language)
        j=dict(id=uuid.uuid4().hex,video_id=spec.video_id,segment_seconds=spec.segment_seconds,language=spec.language,status='queued',signature=sig,error=None,segments=[])
        t=0.
        while t<v['duration']:
            j['segments'].append(dict(index=len(j['segments']),start=t,end=min(t+spec.segment_seconds,v['duration']),status='pending',text=''));t+=spec.segment_seconds
        launch(j);return dict(id=j['id'])
    @app.get('/api/jobs/{key}')
    def get_job(key:str):return store.get('job',key)
    @app.post('/api/jobs/{key}/stop')
    def stop_job(key:str):
        store.get('job',key)
        if key in controls:controls[key]['stop'].set()
        return dict(ok=True)
    @app.post('/api/jobs/{key}/resume')
    def resume(key:str):
        require_model();j=store.get('job',key)
        if j['signature']!=signature(active_video(j['video_id']),j['segment_seconds'],j.get('language','zh')):
            raise HTTPException(409,'视频或模型配置已改变，请新建任务；旧结果仍保留')
        if j['status']=='done':return dict(id=key)
        launch(j);return dict(id=key)
    @app.post('/api/jobs/{key}/playback')
    def playback(key:str,spec:Playback):
        store.get('job',key)
        if key in controls:controls[key].update(timestamp=spec.timestamp,playing=spec.playing)
        return dict(ok=True)
    @app.get('/api/jobs/{key}/events')
    async def job_events(key:str,request:Request):
        store.get('job',key)
        async def stream():
            previous=''
            while not await request.is_disconnected():
                j=await asyncio.to_thread(store.get,'job',key);serialized=json.dumps(j,ensure_ascii=False)
                if serialized!=previous:yield event('state',j);previous=serialized
                if j['status'] in ('done','error','paused'):break
                await asyncio.sleep(.4)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
    @app.post('/api/questions')
    async def ask(spec:Question,request:Request):
        require_model();v=active_video(spec.video_id);stop=threading.Event();q=queue.Queue(maxsize=32)
        run_id=uuid.uuid4().hex;done=threading.Event()
        # Pause scheduling; a current segment completes/cancels through the same engine lock.
        for ctrl in list(controls.values()):ctrl['playing']=False
        def produce():
            try:
                end=min(spec.timestamp,v['duration'])
                for text in engine.generate(v['path'],max(0,end-12),end,spec.question,stop,language=spec.language):
                    while not stop.is_set():
                        try:q.put(('token',{'text':text}),timeout=.2);break
                        except queue.Full:pass
            except Exception as exc:
                if not stop.is_set():q.put(('error',{'message':str(exc)}),timeout=1)
            finally:
                if not stop.is_set():
                    try:q.put(('done',{}),timeout=1)
                    except queue.Full:pass
                done.set()
                question_runs.pop(run_id,None)
        with mutex:
            active_video(spec.video_id)
            question_runs[run_id]=dict(video_id=spec.video_id,stop=stop,done=done)
            threading.Thread(target=produce,daemon=True).start()
        async def stream():
            try:
                while not await request.is_disconnected():
                    if stop.is_set() and done.is_set(): break
                    try:kind,data=await asyncio.to_thread(q.get,True,1)
                    except queue.Empty:yield ': keepalive\n\n';continue
                    yield event(kind,data)
                    if kind in ('done','error'):break
            finally:stop.set()
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
    dist=ROOT/'frontend/dist'
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount('/',StaticFiles(directory=str(dist),html=True),name='frontend')
    return app

app=create_app(storage=os.getenv('DEMO_STORAGE_DIR'))
