import sys,time,os,tempfile,threading
_test_storage=tempfile.TemporaryDirectory(prefix="mage-api-tests-")
os.environ["DEMO_STORAGE_DIR"]=_test_storage.name
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as module

class FakeEngine:
    def __init__(self):
        self.calls=0;self.languages=[];self.questions=[]
    def generate(self,path,start,end,question,stop,language="zh"):
        self.calls+=1;self.languages.append(language);self.questions.append(question)
        yield '测试';yield '片段。'

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.delenv('DEMO_ENABLE_MODEL',raising=False)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='{"format":{"duration":"8"}}'))
    engine=FakeEngine()
    with TestClient(module.create_app(tmp_path,engine)) as c:
        c.app.state.test_storage=tmp_path
        yield c,engine

def upload(c):
    r=c.post('/api/videos',files={'video':('../../clip.mp4',b'fixture','video/mp4')});assert r.status_code==200
    return r.json()

def test_health_does_not_import_model(tmp_path,monkeypatch):
    monkeypatch.delenv('DEMO_ENABLE_MODEL',raising=False)
    with TestClient(module.create_app(tmp_path)) as c:
        assert c.get('/api/health').json()['model_enabled'] is False
        assert c.post('/api/jobs',json={'video_id':'x'}).status_code==409
    assert not any(x in sys.modules for x in ('torch','transformers','cv2'))

def test_upload_and_media(client):
    c,_=client;v=upload(c)
    assert v['duration']==8 and 'path' not in v
    assert c.get('/api/videos/'+v['id']).json()['id']==v['id']
    assert c.get(v['url']).content==b'fixture'
    assert c.post('/api/videos',files={'video':('x.txt',b'x')}).status_code==400

def test_completed_job_is_persistent_and_not_repeated(client):
    c,engine=client;v=upload(c);key=c.post('/api/jobs',json={'video_id':v['id']}).json()['id']
    for _ in range(100):
        job=c.get('/api/jobs/'+key).json()
        if job['status']=='done':break
        time.sleep(.02)
    assert job['status']=='done' and job['segments'][0]['text']=='测试片段。'
    assert c.post(f'/api/jobs/{key}/resume').status_code==200 and engine.calls==1
    assert 'event: state' in c.get(f'/api/jobs/{key}/events').text

def test_question_stream(client):
    c,_=client;v=upload(c)
    r=c.post('/api/questions',json={'video_id':v['id'],'timestamp':0,'question':'测试提问'})
    assert r.status_code==200 and 'event: token' in r.text and 'event: done' in r.text

def test_invalid_and_missing(client):
    c,_=client
    assert c.get('/api/jobs/missing').status_code==404
    assert c.post('/api/questions',json={'video_id':'x','timestamp':-1,'question':'x'}).status_code==422


def test_restart_pauses_unfinished_job_and_resume_skips_done(client):
    c,engine=client;v=upload(c)
    storage=module.Store(c.app.state.test_storage)
    sig=c.post('/api/jobs',json={'video_id':v['id']}).json()['id']
    for _ in range(100):
        old=storage.get('job',sig)
        if old['status']=='done':break
        time.sleep(.02)
    old['status']='running'
    old['segments'].append(dict(index=1,start=4,end=8,status='running',text='半段输出'))
    storage.put('job',old)
    with TestClient(module.create_app(storage.root,engine)) as resumed:
        assert resumed.get('/api/jobs/'+sig).json()['status']=='paused'
        before=engine.calls
        assert resumed.post(f'/api/jobs/{sig}/resume').status_code==200
        for _ in range(100):
            saved=resumed.get('/api/jobs/'+sig).json()
            if saved['status']=='done':break
            time.sleep(.02)
        assert saved['status']=='done' and engine.calls==before+1
        assert saved['segments'][0]['text']=='测试片段。'
        assert saved['segments'][1]['text']=='测试片段。'

def test_changed_signature_does_not_mix_results(client):
    c,_=client;v=upload(c);storage=module.Store(c.app.state.test_storage)
    key=c.post('/api/jobs',json={'video_id':v['id']}).json()['id']
    for _ in range(100):
        j=storage.get('job',key)
        if j['status']=='done':break
        time.sleep(.02)
    j['signature']='obsolete';storage.put('job',j)
    assert c.post(f'/api/jobs/{key}/resume').status_code==409


def test_predecoded_frames_keep_source_timestamps():
    fake=lambda **kw: {'frame_timestamps':[[0.,1.,2.]],'pixel_values':'untouched'}
    result=module.TimestampedFrames(fake,[8.,8.5,9.])()
    assert result['frame_timestamps']==[[8.,8.5,9.]]
    assert result['pixel_values']=='untouched'
    with pytest.raises(RuntimeError):module.TimestampedFrames(fake,[1.])()


def wait_done(c,key):
    for _ in range(100):
        job=c.get('/api/jobs/'+key).json()
        if job['status']=='done':return job
        time.sleep(.02)
    raise AssertionError(job)

def test_delete_is_recoverable_and_keeps_results(client):
    c,_=client;v=upload(c)
    key=c.post('/api/jobs',json={'video_id':v['id']}).json()['id']
    old=wait_done(c,key)
    assert c.delete('/api/videos/'+v['id']).json()['recoverable'] is True
    assert c.get(v['url']).status_code==404
    assert c.get('/api/videos/'+v['id']).status_code==404
    store=module.Store(c.app.state.test_storage)
    saved=store.get('video',v['id'])
    assert 'trash' in Path(saved['path']).parts
    assert Path(saved['path']).read_bytes()==b'fixture'
    assert c.get('/api/jobs/'+key).json()['segments']==old['segments']
    assert c.post('/api/jobs',json={'video_id':v['id']}).status_code==404
    assert c.delete('/api/videos/'+v['id']).status_code==200
    assert c.post('/api/videos/'+v['id']+'/restore').json()['id']==v['id']
    assert c.get(v['url']).content==b'fixture'
    assert upload(c)['id']!=v['id']

class InterruptibleEngine:
    def __init__(self):self.started=threading.Event()
    def generate(self,path,start,end,question,stop,language='zh'):
        self.started.set();yield 'partial'
        stop.wait(5)

def test_delete_stops_its_running_job(tmp_path,monkeypatch):
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='{"format":{"duration":"8"}}'))
    engine=InterruptibleEngine()
    with TestClient(module.create_app(tmp_path,engine)) as c:
        v=upload(c);key=c.post('/api/jobs',json={'video_id':v['id']}).json()['id']
        assert engine.started.wait(1)
        assert c.delete('/api/videos/'+v['id']).status_code==200
        assert c.get('/api/jobs/'+key).json()['status']=='paused'
        assert c.get(v['url']).status_code==404

def test_delete_stops_its_running_question(tmp_path,monkeypatch):
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='{"format":{"duration":"8"}}'))
    engine=InterruptibleEngine();responses=[]
    with TestClient(module.create_app(tmp_path,engine)) as c:
        v=upload(c)
        worker=threading.Thread(target=lambda:responses.append(c.post('/api/questions',json={'video_id':v['id'],'timestamp':0,'question':'x'})),daemon=True)
        worker.start();assert engine.started.wait(1)
        assert c.delete('/api/videos/'+v['id']).status_code==200
        worker.join(3);assert not worker.is_alive()
        assert responses[0].status_code==200

def test_language_is_recorded_sent_to_engine_and_part_of_signature(client):
    c,engine=client;v=upload(c)
    zh=c.post('/api/jobs',json={'video_id':v['id'],'language':'zh'}).json()['id'];a=wait_done(c,zh)
    en=c.post('/api/jobs',json={'video_id':v['id'],'language':'en'}).json()['id'];b=wait_done(c,en)
    assert a['language']=='zh' and b['language']=='en' and a['signature']!=b['signature']
    assert engine.languages==['zh','en']
    assert 'English' in engine.questions[-1]
    assert c.post('/api/questions',json={'video_id':v['id'],'timestamp':0,'question':'What is visible?','language':'en'}).status_code==200
    assert engine.languages[-1]=='en'
    assert c.post('/api/jobs',json={'video_id':v['id'],'language':'xx'}).status_code==422
    assert 'concise English' in module.generation_prompt(0,8,'test','en')
    assert '简洁中文' in module.generation_prompt(0,8,'test','zh')

def test_overlong_narration_is_shortened_once_and_original_kept(tmp_path,monkeypatch):
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='{"format":{"duration":"8"}}'))
    class LongEngine:
        calls=0
        def generate(self,path,start,end,question,stop,language='zh'):
            self.calls+=1
            yield '画面中球员正在球场上奔跑。'*5 if self.calls==1 else '球员在球场奔跑。'
    engine=LongEngine()
    with TestClient(module.create_app(tmp_path,engine)) as c:
        v=upload(c);key=c.post('/api/jobs',json={'video_id':v['id']}).json()['id'];job=wait_done(c,key)
        seg=job['segments'][0]
        assert engine.calls==2 and seg['text']=='球员在球场奔跑。'
        assert len(seg['original_text'])>30 and seg['length_target_met'] is True
