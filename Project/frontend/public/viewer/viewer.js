import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from './spark.module.min.js';

const canvas = document.querySelector('canvas');
const status = document.querySelector('#status');
const picker = document.querySelector('#picker');
const sample = document.querySelector('#sample');
const reset = document.querySelector('#reset');
try {
  const renderer = new THREE.WebGLRenderer({canvas, antialias:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x17201e);
  scene.add(new SparkRenderer({renderer}));
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  let current;
  new ResizeObserver(() => {
    const {width,height} = canvas.parentElement.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width,height,false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }).observe(canvas.parentElement);
  const record = document.querySelector('#record');
  let orbit;
  renderer.setAnimationLoop(() => {
    if (orbit) {
      const t = Math.min(1,(performance.now()-orbit.start)/8000);
      camera.position.copy(orbit.offset).applyAxisAngle(camera.up,t*Math.PI*2).add(controls.target);
      if (t===1 && orbit.recorder.state==='recording') orbit.recorder.stop();
    }
    controls.update(); renderer.render(scene,camera);
  });
  record.onclick = () => {
    if (!current || orbit) return;
    let stream;
    try {
      stream = canvas.captureStream(30);
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = e => { if(e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach(track=>track.stop());
        const link = document.createElement('a');
        link.href = URL.createObjectURL(new Blob(chunks,{type:recorder.mimeType}));
        link.download = '实物展示.'+(recorder.mimeType.includes('mp4')?'mp4':'webm');
        link.click(); setTimeout(()=>URL.revokeObjectURL(link.href),60000);
        camera.position.copy(orbit.offset).add(controls.target); orbit=null;
        controls.enabled=true; record.disabled=sample.disabled=picker.disabled=reset.disabled=build.disabled=false;
        status.textContent='展示视频已导出';
      };
      recorder.start();
      orbit={start:performance.now(),offset:camera.position.clone().sub(controls.target),recorder};
      controls.enabled=false; record.disabled=sample.disabled=picker.disabled=reset.disabled=build.disabled=true;
      status.textContent='正在录制环绕视频，请保持此页可见…';
    } catch(error){stream?.getTracks().forEach(track=>track.stop());status.textContent='当前浏览器无法录制：'+error.message;}
  };
  const build = document.querySelector('#build');
  build.onclick = async () => {
    const files = [...document.querySelector('#sources').files];
    const progress = document.querySelector('#progress');
    if (!files.length) { progress.textContent='请先选择拍摄素材';return; }
    const form = new FormData();
    files.forEach(f=>form.append('files',f));
    build.disabled=record.disabled=true; progress.textContent='正在匹配拍摄视角并计算三维点，请保持页面打开…';
    try {
      const response=await fetch('/api/reconstruct',{method:'POST',body:form});
      if(!response.ok) throw Error((await response.json()).detail);
      const file=new File([await response.blob()],'实物预览.splat');
      const link=document.querySelector('#download');
      if(link.href) URL.revokeObjectURL(link.href);
      link.href=URL.createObjectURL(file);link.hidden=false;
      await load(file);progress.textContent='已生成稀疏模型，可下载保存';
    } catch(error){progress.textContent=String(error.message);}
    finally {build.disabled=false;record.disabled=!current || !window.MediaRecorder || !canvas.captureStream;}
  };
  async function load(file) {
    sample.disabled = picker.disabled = true;
    status.textContent = '正在读取模型…';
    let next;
    try {
      next = new SplatMesh(file ? {fileBytes:await file.arrayBuffer(),fileName:file.name} : {url:'/api/exhibit/trex.ply'});
      await next.initialized;
      const sphere = next.getBoundingBox(true).getBoundingSphere(new THREE.Sphere());
      if (!Number.isFinite(sphere.radius) || sphere.radius <= 0) throw Error('模型缺少有效空间范围');
      if (current) {scene.remove(current); current.dispose();}
      current = next;
      scene.add(current);
      controls.target.copy(sphere.center);
      const distance = sphere.radius * 1.15 / Math.sin(Math.atan(Math.tan(camera.fov*Math.PI/360)*Math.min(camera.aspect,1)));
      camera.near = sphere.radius / 10000;
      camera.far = distance * 100;
      camera.updateProjectionMatrix();
      camera.position.copy(sphere.center).add(new THREE.Vector3(0,0,distance));
      controls.update();
      controls.saveState();
      reset.disabled = false; record.disabled = !window.MediaRecorder || !canvas.captureStream;
      status.textContent = (file?.name || '恐龙骨架 · 30,000 次迭代') + '　左键旋转 · 右键平移 · 滚轮或双指缩放';
    } catch(error) {
      if(next && next !== current) next.dispose();
      status.textContent = '加载失败：' + error.message;
    } finally {sample.disabled = picker.disabled = false; picker.value = '';}
  }
  sample.onclick = () => load();
  picker.onchange = () => {if(picker.files[0]) load(picker.files[0]);};
  reset.onclick = () => controls.reset();
  load();
} catch(error) {status.textContent = '无法初始化 3D 视窗，请使用支持 WebGL2 的浏览器。' + error.message;}
