import {useEffect,useRef,useState} from 'react';
import {SpeechQueue,type Language} from './speech';

type Video={id:string;name:string;duration:number;url:string};
type Segment={index:number;start:number;end:number;status:string;text:string};
type Job={id:string;status:string;error:string|null;segments:Segment[];language?:Language};
type Health={model_enabled:boolean;model_loaded:boolean;model_local:boolean};
const clock=(s:number)=>`${Math.floor(s/60).toString().padStart(2,'0')}:${Math.floor(s%60).toString().padStart(2,'0')}`;
async function api<T>(url:string,init?:RequestInit):Promise<T>{
 const r=await fetch(url,init);
 if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText}));throw Error(e.detail||r.statusText);}
 return r.json();
}
const post=(data:unknown)=>({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});

// The visible control contains a real file input. Do not rely on .click() of a
// display:none element: embedded browsers can suppress its picker activation.
function UploadPicker({label,disabled,onFile,primary=false}:{label:string;disabled:boolean;onFile:(f:File)=>void;primary?:boolean}){
 return <label className={`upload-picker ${primary?'primary':''} ${disabled?'disabled':''}`}>
  <span>{label}</span>
  <input type="file" aria-label={label} disabled={disabled} accept="video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v"
   onChange={e=>{const f=e.currentTarget.files?.[0];e.currentTarget.value='';if(f)onFile(f);}}/>
 </label>;
}

export default function App(){
 const [health,setHealth]=useState<Health|null>(null);
 const [video,setVideo]=useState<Video|null>(null),[job,setJob]=useState<Job|null>(null);
 const [error,setError]=useState(''),[uploading,setUploading]=useState(false),[deleting,setDeleting]=useState(false),[generating,setGenerating]=useState(false);
 const [removed,setRemoved]=useState<Video|null>(null),[language,setLanguage]=useState<Language>('zh');
 const [question,setQuestion]=useState(''),[answer,setAnswer]=useState('');
 const [tts,setTts]=useState(false),[voices,setVoices]=useState<SpeechSynthesisVoice[]>([]),[voice,setVoice]=useState(''),[speechState,setSpeechState]=useState('待播报');
 const [time,setTime]=useState(0),[tab,setTab]=useState<'commentary'|'qa'>('commentary');
 const player=useRef<HTMLVideoElement>(null),source=useRef<EventSource|null>(null),abort=useRef<AbortController|null>(null);
 const speech=useRef(new SpeechQueue()),spoken=useRef(new Set<number>()),latestJob=useRef<Job|null>(null);
 const lastSync=useRef(0),version=useRef(0),waiting=useRef(false),fileOperation=useRef(false);
 const fileBusy=uploading||deleting;
 const activeSpeechSegment=useRef<number|null>(null),hold=useRef<number|null>(null),internalSeek=useRef(false);
 const [speechWaiting,setSpeechWaiting]=useState(false);
 useEffect(()=>{speech.current.onIdle=releaseHold;const timer=setInterval(()=>sync(),80);return()=>clearInterval(timer);},[tts]);
 function releaseHold(){const el=player.current,end=hold.current;if(!el||end===null)return;hold.current=null;setSpeechWaiting(false);internalSeek.current=true;el.currentTime=end;if(end<el.duration)void el.play().catch(()=>setError("请点击播放器继续播放。"));}

 useEffect(()=>{
  const refresh=()=>api<Health>('/api/health').then(setHealth).catch(e=>setError(String(e)));
  void refresh();const timer=setInterval(()=>void refresh(),10000);
  return()=>clearInterval(timer);
 },[]);
 useEffect(()=>{
  let active=true;const stamp=version.current;
  const valid=()=>active&&stamp===version.current;
  try{
   const saved=JSON.parse(localStorage.getItem('mage-demo-session')||'null');
   if(saved?.video_id){
    api<Video>(`/api/videos/${saved.video_id}`).then(async v=>{
     if(!valid())return;setVideo(v);setLanguage(saved.language==='en'?'en':'zh');
     if(saved.job_id){const j=await api<Job>(`/api/jobs/${saved.job_id}`);if(!valid())return;
      setJob(j);latestJob.current=j;setLanguage(j.language||'zh');
      if(['running','queued'].includes(j.status))connect(j.id);
     }
    }).catch(()=>{});
   }
  }catch{/* Optional browser session storage. */}
  return()=>{active=false;};
 },[]);
 useEffect(()=>{
  try{
   if(video)localStorage.setItem('mage-demo-session',JSON.stringify({video_id:video.id,job_id:job?.id,language}));
   else localStorage.removeItem('mage-demo-session');
  }catch{/* Private browsing can disable storage. */}
 },[video,job?.id,language]);
 useEffect(()=>{
  speech.current.onState=setSpeechState;
  return()=>{speech.current.clear();source.current?.close();abort.current?.abort();};
 },[]);
 useEffect(()=>{
  speech.current.language=language;speech.current.voiceURI='';setVoice('');
  const synth=window.speechSynthesis;if(!synth)return;
  const load=()=>setVoices(synth.getVoices().filter(v=>v.lang.toLowerCase().startsWith(language)));
  load();synth.addEventListener('voiceschanged',load);
  return()=>synth.removeEventListener('voiceschanged',load);
 },[language]);

 function cancel(){hold.current=null;setSpeechWaiting(false);activeSpeechSegment.current=null;version.current++;abort.current?.abort();speech.current.clear();setGenerating(false);}
 async function stopWork(){
  waiting.current=false;cancel();source.current?.close();player.current?.pause();
  const current=latestJob.current;
  if(current)await api(`/api/jobs/${current.id}/stop`,post({})).catch(()=>{});
 }
 function clearCurrent(){
  setVideo(null);setJob(null);latestJob.current=null;setAnswer('');setQuestion('');setTime(0);spoken.current.clear();
 }
 function connect(id:string){
  source.current?.close();const stamp=version.current;
  const es=new EventSource(`/api/jobs/${id}/events`);source.current=es;
  es.addEventListener('state',ev=>{
   if(stamp!==version.current)return;
   const j=JSON.parse((ev as MessageEvent).data) as Job;setJob(j);latestJob.current=j;
   const el=player.current;
   if(waiting.current&&el&&j.segments.some(s=>s.status==='done'&&s.start<=el.currentTime&&s.end>el.currentTime)){
    waiting.current=false;void el.play().catch(()=>setError('片段已就绪，请点击播放器继续播放。'));
   }
   if(['error','paused'].includes(j.status))waiting.current=false;
   if(j.error)setError(j.error);
   if(['done','error','paused'].includes(j.status))es.close();
  });
  es.onerror=()=>{es.close();if(stamp===version.current)setError('讲解连接中断；任务记录已保留，可点击继续恢复。');};
 }
 async function upload(f:File){
  if(fileOperation.current)return;
  fileOperation.current=true;setUploading(true);setError('');setRemoved(null);
  try{
   await stopWork();const form=new FormData();form.append('video',f);
   const v=await api<Video>('/api/videos',{method:'POST',body:form});
   clearCurrent();setVideo(v);
  }catch(e){setError(String(e));}
  finally{fileOperation.current=false;setUploading(false);}
 }
 async function removeVideo(){
  if(!video||fileOperation.current)return;
  fileOperation.current=true;setDeleting(true);setError('');
  try{
   await stopWork();await api(`/api/videos/${video.id}`,{method:'DELETE'});
   setRemoved(video);clearCurrent();
  }catch(e){
   setError(String(e));
   if(latestJob.current)connect(latestJob.current.id);
  }finally{fileOperation.current=false;setDeleting(false);}
 }
 async function undoDelete(){
  if(!removed||fileOperation.current)return;
  fileOperation.current=true;setDeleting(true);setError('');
  try{
   const restored=await api<Video>(`/api/videos/${removed.id}/restore`,post({}));
   clearCurrent();setVideo(restored);setRemoved(null);
  }catch(e){setError(String(e));}
  finally{fileOperation.current=false;setDeleting(false);}
 }
 async function changeLanguage(next:Language){
  if(next===language)return;
  const stopping=stopWork();setLanguage(next);setJob(null);latestJob.current=null;setAnswer('');spoken.current.clear();
  await stopping;
 }
 async function start(){
  if(!video)return;if(!health?.model_enabled){setError('本地 Mage 模型尚未启用。上传和播放已可用；真实解说需要在 GPU 可用后以 DEMO_ENABLE_MODEL=1 启动后端。');return;}setError('');setGenerating(true);
  cancel();const stamp=version.current;setGenerating(true);spoken.current.clear();
  if(player.current){player.current.pause();player.current.currentTime=0;}waiting.current=true;
  try{
   const r=await api<{id:string}>('/api/jobs',post({video_id:video.id,segment_seconds:8,language}));
   if(stamp===version.current)connect(r.id);
   else void api(`/api/jobs/${r.id}/stop`,post({}));
  }catch(e){if(stamp===version.current){waiting.current=false;setError(String(e));}}
  finally{if(stamp===version.current)setGenerating(false);}
 }
 async function resume(){
  if(!job)return;setError('');
  try{waiting.current=true;await api(`/api/jobs/${job.id}/resume`,post({}));connect(job.id);sync(true);}
  catch(e){waiting.current=false;setError(String(e));}
 }
 function sync(force=false){
  const el=player.current,j=latestJob.current;if(!el)return;setTime(el.currentTime);
  const speaking=j?.segments.find(s=>s.index===activeSpeechSegment.current);
  if(tts&&speaking&&speech.current.busy()&&el.currentTime>=speaking.end-.08){
   if(hold.current===null){hold.current=speaking.end;setSpeechWaiting(true);el.pause();internalSeek.current=true;el.currentTime=Math.max(speaking.start,speaking.end-.08);speech.current.resume();}
   return;
  }
  if(hold.current!==null)return;
  if(j&&j.status==='running'&&!el.paused&&j.segments.some(s=>s.start<=el.currentTime&&s.end>el.currentTime&&s.status!=='done')){waiting.current=true;el.pause();}
  if(j&&(force||Date.now()-lastSync.current>1000)){
   lastSync.current=Date.now();void api(`/api/jobs/${j.id}/playback`,post({timestamp:el.currentTime,playing:!el.paused||waiting.current})).catch(()=>{});
  }
  const current=j?.segments.find(s=>s.start<=el.currentTime&&el.currentTime<s.end);
  if(activeSpeechSegment.current!==null&&activeSpeechSegment.current!==current?.index&&!speech.current.busy())activeSpeechSegment.current=null;
  if(tts&&!el.paused&&!speech.current.busy()&&current?.status==='done'&&!spoken.current.has(current.index)){
   spoken.current.add(current.index);activeSpeechSegment.current=current.index;
   speech.current.resume();speech.current.speakSegment(current.text,current.end-el.currentTime);
  }
 }
 function seek(){
  if(internalSeek.current){internalSeek.current=false;return;}
  hold.current=null;setSpeechWaiting(false);
  activeSpeechSegment.current=null;abort.current?.abort();speech.current.clear();const t=player.current?.currentTime||0;
  spoken.current=new Set((job?.segments||[]).filter(s=>s.end<t).map(s=>s.index));sync(true);
 }
 async function stop(){
  const id=latestJob.current?.id;await stopWork();if(id)connect(id);
 }
 async function ask(){
  if(!video||!question.trim())return;if(!health?.model_enabled){setError('本地 Mage 模型尚未启用，暂时不能生成回答。GPU 可用后需要以 DEMO_ENABLE_MODEL=1 启动后端。');return;}
  waiting.current=false;cancel();player.current?.pause();setTab('qa');setError('');setAnswer('');setGenerating(true);
  const controller=new AbortController();abort.current=controller;const stamp=version.current;let full='';
  try{
   const r=await fetch('/api/questions',{...post({video_id:video.id,timestamp:player.current?.currentTime||0,question,language}),signal:controller.signal});
   if(!r.ok){const x=await r.json();throw Error(x.detail);}
   const reader=r.body!.getReader(),decoder=new TextDecoder();let buffer='';
   while(true){
    const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});let end;
    while((end=buffer.indexOf('\n\n'))>=0){
     const block=buffer.slice(0,end);buffer=buffer.slice(end+2);
     const kind=block.split('\n').find(s=>s.startsWith('event:'))?.slice(6).trim();
     const data=block.split('\n').filter(s=>s.startsWith('data:')).map(s=>s.slice(5).trim()).join('\n');
     if(!data)continue;const payload=JSON.parse(data);
     if(kind==='error')throw Error(payload.message);
     if(kind==='token'&&stamp===version.current){full+=payload.text;setAnswer(full);}
    }
    if(done)break;
   }
   if(tts&&stamp===version.current){speech.current.resume();speech.current.add(full);}
  }catch(e){if(!controller.signal.aborted&&stamp===version.current)setError(String(e));}
  finally{if(stamp===version.current)setGenerating(false);}
 }
 function replaySpeech(){
  const el=player.current,j=latestJob.current;
  const segment=j?.segments.find(s=>s.start<=(el?.currentTime||0)&&(el?.currentTime||0)<s.end);
  const text=tab==='qa'?answer:segment?.status==='done'?segment.text:'';
  if(!text.trim())return;
  speech.current.clear();speech.current.resume();setTts(true);
  if(tab==='qa'){hold.current=null;setSpeechWaiting(false);activeSpeechSegment.current=null;el?.pause();speech.current.add(text);}
  else if(segment){spoken.current.add(segment.index);activeSpeechSegment.current=segment.index;speech.current.speakSegment(text,Math.max(1,segment.end-(el?.currentTime||0)));}
 }
 const replayAvailable=tab==='qa'?!!answer.trim():!!job?.segments.some(s=>s.status==='done'&&s.start<=time&&time<s.end);
 const completed=job?.segments.filter(s=>s.status==='done').length||0;
 const pickerLabel=uploading?'正在上传…':video?'更换视频':'上传视频';
 return <div className="shell">
  <header><div className="brand"><span className="mark">M</span><div>MAGE<span className="studio">VIDEO STUDIO</span></div></div><div className="status"><i className={health?.model_enabled?'green':''}/>{health?.model_enabled?'本地模型已启用':'GPU推理未启用 · 可上传预览'}</div></header>
  <main>
   <div className="heading"><div><span className="eyebrow">YOUR VIDEO, UNDERSTOOD.</span><h1>让画面，有话可说。</h1><p>上传一段视频，逐段理解、连续讲解，也可以随时提问。</p></div>
    <div className="video-actions"><UploadPicker label={pickerLabel} disabled={fileBusy} onFile={f=>void upload(f)}/>
     {video&&<button className="danger-button" disabled={fileBusy} onClick={()=>void removeVideo()}>{deleting?'正在删除…':'删除视频'}</button>}
    </div>
   </div>
   {error&&<div className="error" role="alert">{error}<button onClick={()=>setError('')}>×</button></div>}
   {removed&&<div className="notice" role="status"><span>已删除「{removed.name}」，可重新上传。原文件已移入回收目录，历史解说保留。</span><button className="quiet" disabled={fileBusy} onClick={()=>void undoDelete()}>撤销删除</button></div>}
   <div className="workspace"><section className="viewer">
    <div className="video-box">{video?<video ref={player} src={video.url} controls onEnded={()=>sync(true)} onTimeUpdate={()=>sync()} onSeeked={seek} onPause={()=>{if(hold.current===null)speech.current.pause();sync(true);}} onPlay={()=>{speech.current.resume();sync(true);}}/>:
     <div className="empty" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f)void upload(f);}}>
      <div className="play-icon">▷</div><h2>{uploading?'正在接收视频…':'把视频拖到这里'}</h2><p>MP4 / MOV / WEBM / MKV · 原文件保留在本机服务器</p>
      <UploadPicker label={uploading?'正在上传…':'选择视频'} disabled={fileBusy} primary onFile={f=>void upload(f)}/>
     </div>}
    </div>
    {job?.status==='running'&&job.segments.some(s=>s.start<=time&&time<s.end&&s.status!=='done')&&<div className="buffer-notice" role="status">正在生成 {clock(time)} 所在片段的解说，视频暂时等待…</div>}
    {speechWaiting&&<div className="buffer-notice" role="status">画面已到段尾，等待当前解说完整播完…</div>}
    <div className="video-meta"><div><b>{video?.name||'尚未选择视频'}</b><span>{video?`${clock(time)} / ${clock(video.duration)}`:'本地模型 · 不调用外部LLM'}</span></div><span className="pill">MAGE-VL</span></div>
    <div className="language-control"><label htmlFor="commentary-language">解说语言</label><select id="commentary-language" value={language} disabled={fileBusy} onChange={e=>void changeLanguage(e.target.value as Language)}><option value="zh">中文</option><option value="en">English（英文）</option></select><small>切换后重新生成 · 历史结果保留</small></div>
    <div className="controls"><button disabled={!video||fileBusy||generating||job?.status==='running'||job?.status==='queued'} onClick={()=>void start()}>▷ 开始自动讲解</button>
     {job&&!['done','running','queued'].includes(job.status)&&<button className="outline" disabled={fileBusy||generating} onClick={()=>void resume()}>继续任务</button>}
     <button className="quiet" onClick={()=>void stop()} disabled={!job&&!generating}>停止</button><span>{job?`${completed} / ${job.segments.length} 段已完成`:'每8秒一段 · 文字就绪后播放'}</span>
    </div>
    {job&&<progress aria-label="片段完成进度" value={completed} max={job.segments.length}/>}
    <div className="speech"><div><b>语音播报</b><span>完整短句播报 · 段尾等待 · 下一段提前准备</span></div><label><input type="checkbox" checked={tts} disabled={!speech.current.supported()} onChange={e=>{setTts(e.target.checked);if(!e.target.checked){speech.current.clear();activeSpeechSegment.current=null;releaseHold();}else{const current=latestJob.current?.segments.find(s=>s.start<=time&&time<s.end);if(current)spoken.current.delete(current.index);}}}/> 开启</label>
     <select aria-label="播报音色" value={voice} onChange={e=>{setVoice(e.target.value);speech.current.voiceURI=e.target.value;}}><option value="">系统{language==='en'?'英文':'中文'}音色</option>{voices.map(v=><option key={v.voiceURI} value={v.voiceURI}>{v.name}{v.localService?' · 本机':''}</option>)}</select>
     <button className="quiet" disabled={!speech.current.supported()} onClick={()=>{speech.current.resume();speech.current.add(language==='en'?'Hello. This is the browser speech test.':'你好，这是浏览器语音播报测试。');}}>试听</button><button className="outline" disabled={!speech.current.supported()||!replayAvailable} onClick={replaySpeech}>重新播报</button><small>{speech.current.supported()?speechState:'此浏览器不支持语音合成'}</small>
    </div>
   </section><aside>
    <div className="tabs"><button className={tab==='commentary'?'selected':''} onClick={()=>setTab('commentary')}>片段讲解</button><button className={tab==='qa'?'selected':''} onClick={()=>setTab('qa')}>视频问答</button></div>
    <div className="transcript">{tab==='commentary'?job?.segments.filter(s=>s.start<=time).map(s=><article key={s.index} className={time>=s.start&&time<s.end?'current':''}>
     <button className="timestamp" onClick={()=>{if(player.current){player.current.currentTime=s.start;seek();}}}>{clock(s.start)} — {clock(s.end)}</button><p>{s.text||(s.status==='running'?'正在理解这一段画面…':'等待处理')}</p><small>{s.status==='done'?'已完成':s.status==='running'?'生成中':'待处理'}</small>
    </article>)||<div className="placeholder"><span>✧</span><h3>从第一段画面开始</h3><p>上传后选择中文或英文，开始分段讲解。<br/>点击开始后，首段文字会逐步显示在这里。</p></div>:
     <div className="qa-answer"><span className="eyebrow">MAGE ANSWER</span><p>{answer||'播放到你关心的位置，在下方输入问题。回答范围为当前画面及此前12秒。'}</p></div>}
    </div>
    <div className="question"><textarea value={question} onChange={e=>setQuestion(e.target.value)} placeholder="问问这段视频，比如：刚才发生了什么？" maxLength={2000}/><div><small>提问会暂停播放 · 使用所选解说语言回答</small><button disabled={!video||fileBusy||generating||!question.trim()} onClick={()=>void ask()}>{generating?'处理中…':'发送 ↗'}</button></div></div>
   </aside></div>
   <footer><span>LOCAL INTELLIGENCE. YOUR OWN STORY.</span><span>模型推理尚待GPU验收 · 已上传视频分段处理</span></footer>
  </main>
 </div>;
}
