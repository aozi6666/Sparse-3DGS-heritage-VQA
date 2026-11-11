export type Language = 'zh' | 'en';
export class SpeechQueue {
  private queue:string[]=[];
  private epoch=0;
  private active=false;
  onState: (value:string)=>void=()=>{};
  voiceURI='';
  language:Language='zh';
  rate=1.05;
  onIdle:()=>void=()=>{};
  busy(){return this.active||this.queue.length>0;}
  speakSegment(text:string,remainingSeconds:number){
    this.rate=1.1;
    this.add(text);
  }
  supported(){return typeof window!=='undefined'&&'speechSynthesis' in window;}
  clear(){this.epoch++;this.queue=[];this.active=false;if(this.supported())window.speechSynthesis.cancel();this.onState('待播报');}
  pause(){if(this.supported())window.speechSynthesis.pause();this.onState('播报暂停');}
  resume(){if(this.supported())window.speechSynthesis.resume();this.pump();}
  add(text:string){
    const sentences=text.split(/(?<=[。！？!?])|(?<=\.)\s+|\n+/).map(x=>x.trim()).filter(Boolean);
    this.queue.push(...sentences);
    if(this.queue.length>12){this.queue=this.queue.slice(-12);this.onState('队列过长，已跳过旧解说');}
    this.pump();
  }
  private pump(){
    if(this.active||!this.queue.length||!this.supported())return;
    const stamp=this.epoch,u=new SpeechSynthesisUtterance(this.queue.shift()!);
    u.lang=this.language==='en'?'en-US':'zh-CN';u.rate=this.rate;
    const voices=window.speechSynthesis.getVoices().filter(x=>x.lang.toLowerCase().startsWith(this.language));
    u.voice=voices.find(x=>x.voiceURI===this.voiceURI)||voices.find(x=>x.localService)||voices[0]||null;
    this.active=true;this.onState('正在播报');
    const done=()=>{if(stamp!==this.epoch)return;this.active=false;this.onState('待播报');this.pump();if(!this.busy())this.onIdle();};
    u.onend=done;u.onerror=done;window.speechSynthesis.speak(u);
  }
}
