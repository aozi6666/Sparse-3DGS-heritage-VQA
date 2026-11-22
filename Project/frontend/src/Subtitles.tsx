import {useState} from 'react';
export type Cue={start:number;end:number;text:string};
const seconds=(value:string)=>value.replace(',','.').split(':').reduce((n,x)=>n*60+Number(x),0);
export function Subtitles({cues,onChange,onSeek}:{cues:Cue[];onChange:(list:Cue[])=>void;onSeek:(t:number)=>void}){
 const [filter,setFilter]=useState(''),[message,setMessage]=useState('');
 async function read(file:File){
  if(file.size>2*1024*1024){setMessage('字幕文件上限为 2 MB');return;}
  try{
   const blocks=(await file.text()).replace(/\r/g,'').split(/\n\s*\n/);
   const list:Cue[]=[];
   for(const block of blocks){
    const lines=block.split('\n'),i=lines.findIndex(x=>x.includes('-->'));
    if(i<0)continue;
    const times=lines[i].match(/(?:\d+:)?\d{2}:\d{2}[.,]\d{3}/g);
    if(times?.length!==2)continue;
    const [start,end]=times.map(seconds),text=lines.slice(i+1).join(' ').replace(/<[^>]+>/g,'').trim();
    if(end>start && text)list.push({start,end,text});
   }
   onChange(list.sort((a,b)=>a.start-b.start));setMessage(list.length?'已读取 '+list.length+' 段字幕':'未找到有效字幕，请检查时间码');
  }catch{setMessage('读取字幕失败');}
 }
 return <details className="subtitle-tools"><summary>字幕与片段定位</summary><label>导入 SRT / VTT <input type="file" accept=".srt,.vtt" onChange={e=>{const f=e.target.files?.[0];if(f)void read(f);}}/></label><small role="status">{message}</small>
 {cues.length>0 && <><input aria-label="搜索整段视频字幕" placeholder="搜索整段视频，例如「内部结构」" value={filter} onChange={e=>setFilter(e.target.value)}/><div className="cue-list">{cues.filter(c=>c.text.toLowerCase().includes(filter.trim().toLowerCase())).slice(0,100).map((c,i)=><button key={i} onClick={()=>onSeek(c.start)}><time>{Math.floor(c.start/60)}:{String(Math.floor(c.start%60)).padStart(2,'0')}</time>{c.text}</button>)}</div></>}
 </details>;
}
