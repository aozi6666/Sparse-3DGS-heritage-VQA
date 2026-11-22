import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from './spark.module.min.js';

const canvas = document.querySelector('canvas');
const status = document.querySelector('output');
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
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene,camera); });
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
      reset.disabled = false;
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
