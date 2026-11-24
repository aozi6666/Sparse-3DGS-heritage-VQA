const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const ts=require('typescript');
const code=ts.transpileModule(fs.readFileSync('src/speech.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
function fixture(){
 const calls=[],synth={speak:u=>calls.push(u),cancel(){},pause(){},resume(){},getVoices:()=>[{voiceURI:'local',lang:'zh-CN',localService:true},{voiceURI:'chosen',lang:'zh-CN',localService:false},{voiceURI:'english',lang:'en-US',localService:true}]};
 const context={exports:{},window:{speechSynthesis:synth},SpeechSynthesisUtterance:class{constructor(text){this.text=text}}};
 vm.runInNewContext(code,context);return {queue:new context.exports.SpeechQueue(),calls};
}
test('sentences are spoken sequentially with a local Chinese voice',()=>{
 const {queue,calls}=fixture();queue.add('第一句。第二句！');assert.equal(calls.length,1);assert.equal(calls[0].text,'第一句。');assert.equal(calls[0].voice.voiceURI,'local');calls[0].onend();assert.equal(calls.length,2);assert.equal(calls[1].text,'第二句！');
});
test('cancelled callbacks cannot consume the new session queue',()=>{
 const {queue,calls}=fixture();queue.add('旧句。另一旧句。');const old=calls[0];queue.clear();queue.add('新句。第二新句。');old.onend();assert.equal(calls.length,2);calls[1].onend();assert.equal(calls.length,3);assert.equal(calls[2].text,'第二新句。');
});
test('explicit voice selection and bounded waiting queue',()=>{
 const {queue,calls}=fixture();queue.voiceURI='chosen';queue.add(Array.from({length:20},(_,i)=>`句${i}。`).join(''));assert.equal(calls[0].voice.voiceURI,'chosen');let i=0;while(i<calls.length){calls[i++].onend();}assert.equal(calls.length,12);
});

test('English uses an English voice and sentence queue',()=>{
 const {queue,calls}=fixture();queue.language='en';queue.voiceURI='local';queue.add('First sentence. Second sentence!');
 assert.equal(calls[0].lang,'en-US');assert.equal(calls[0].voice.voiceURI,'english');assert.equal(calls[0].text,'First sentence.');
 calls[0].onend();assert.equal(calls[1].text,'Second sentence!');
});

test('segment speech finishes fully and notifies only after its last sentence',()=>{
 const {queue,calls}=fixture();let ended=0;queue.onIdle=()=>ended++;
 queue.speakSegment('第一句。第二句。',8);assert.equal(queue.busy(),true);
 calls[0].onend();assert.equal(ended,0);assert.equal(calls[1].text,'第二句。');
 calls[1].onend();assert.equal(ended,1);assert.equal(queue.busy(),false);
 assert.equal(calls[0].rate,1.1);
});
test('explicit cancellation does not trigger automatic segment advance',()=>{
 const {queue,calls}=fixture();let ended=0;queue.onIdle=()=>ended++;
 queue.speakSegment('旧句。',8);queue.clear();calls[0].onend();assert.equal(ended,0);
});
