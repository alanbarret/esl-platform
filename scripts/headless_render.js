/**
 * Headless 3D Avatar Renderer
 * Uses Playwright + Three.js to render the Arab Man GLB signing HELLO
 * Outputs PNG frames to /tmp/frames/ then stitches with OpenCV
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const FRAMES_DIR = '/tmp/avatar_frames';
const AVATAR_URL = 'http://localhost:3001/avatar/arab-man.glb';
const ANIM_API   = 'http://localhost:8001/api/v1/translate';
const FPS = 25;
const WIDTH = 540;
const HEIGHT = 720;

// Get animation data from API
async function getAnimation() {
  const res = await fetch(ANIM_API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'Hello', output_format: 'gltf' })
  });
  return res.json();
}

const HTML = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background: #0a0a14; overflow: hidden; }
  canvas { display:block; }
</style>
</head>
<body>
<script type="importmap">
  {"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const W = ${WIDTH}, H = ${HEIGHT};
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a14);
scene.fog = new THREE.Fog(0x0a0a14, 8, 20);

const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 100);
camera.position.set(0, 1.2, 3.2);
camera.lookAt(0, 1.0, 0);

// Lighting
const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);

const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(2, 4, 2);
key.castShadow = true;
scene.add(key);

const fill = new THREE.DirectionalLight(0x8060ff, 0.4);
fill.position.set(-2, 2, -1);
scene.add(fill);

const rim = new THREE.DirectionalLight(0x00ffaa, 0.3);
rim.position.set(0, 3, -3);
scene.add(rim);

// Floor
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 8),
  new THREE.MeshStandardMaterial({ color: 0x111120, roughness: 0.9 })
);
floor.rotation.x = -Math.PI/2;
floor.receiveShadow = true;
scene.add(floor);

// State
let mixer = null;
let clock = new THREE.Clock();
let animData = null;
let avatarScene = null;
let boneMap = new Map();
let restMap = new Map();
let currentTime = 0;
let ready = false;

// Load avatar
const loader = new GLTFLoader();
loader.load('${AVATAR_URL}', (gltf) => {
  avatarScene = gltf.scene;
  avatarScene.traverse(obj => {
    obj.castShadow = true;
    obj.receiveShadow = true;
    if (obj.name) {
      boneMap.set(obj.name, obj);
      restMap.set(obj.name, obj.quaternion.clone());
    }
  });
  avatarScene.position.set(0, 0, 0);
  scene.add(avatarScene);
  
  // Signal ready
  window.__avatarLoaded = true;
  console.log('Avatar loaded, bones:', boneMap.size);
});

// Apply animation frame
function applyFrame(t) {
  if (!animData || !avatarScene) return;
  const channels = animData.channels;
  const samplers = animData.samplers;

  for (const ch of channels) {
    if (ch.target.path !== 'rotation') continue;
    const sampler = samplers[ch.sampler];
    const boneName = ch.target.node;
    const bone = boneMap.get(boneName);
    const rest = restMap.get(boneName);
    if (!bone || !rest) continue;

    // Sample rotation at time t
    const times = sampler.input;
    const values = sampler.output;
    let rot = [0,0,0,1];
    
    if (t <= times[0]) {
      rot = values.slice(0,4);
    } else if (t >= times[times.length-1]) {
      rot = values.slice(-4);
    } else {
      for (let i=0; i<times.length-1; i++) {
        if (t >= times[i] && t < times[i+1]) {
          const a = (t-times[i])/(times[i+1]-times[i]);
          const q0 = new THREE.Quaternion(...values.slice(i*4,i*4+4));
          const q1 = new THREE.Quaternion(...values.slice((i+1)*4,(i+1)*4+4));
          q0.slerp(q1, a);
          rot = [q0.x, q0.y, q0.z, q0.w];
          break;
        }
      }
    }
    
    const offset = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]);
    bone.quaternion.copy(rest).multiply(offset);
  }
}

// Expose control
window.__setAnim = (data) => { animData = data; ready = true; };
window.__setTime = (t) => { currentTime = t; };
window.__render = () => {
  applyFrame(currentTime);
  renderer.render(scene, camera);
};
window.__getCanvas = () => renderer.domElement;
</script>
</body>
</html>`;

async function main() {
  fs.mkdirSync(FRAMES_DIR, { recursive: true });
  
  // Clean old frames
  fs.readdirSync(FRAMES_DIR).forEach(f => fs.unlinkSync(path.join(FRAMES_DIR, f)));
  
  console.log('Getting animation data...');
  const animResult = await getAnimation();
  const animation = animResult.gltf_animation;
  const duration = animation.duration;
  const totalFrames = Math.ceil(duration * FPS);
  
  console.log(`Animation: ${animation.name}, ${duration.toFixed(1)}s, ${totalFrames} frames`);
  
  console.log('Launching browser...');
  const browser = await chromium.launch({
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage',
           '--disable-gpu','--use-gl=swiftshader']
  });
  
  const page = await browser.newPage();
  await page.setViewportSize({ width: WIDTH, height: HEIGHT });
  
  // Serve the HTML
  await page.setContent(HTML, { waitUntil: 'networkidle' });
  
  // Wait for avatar to load
  console.log('Waiting for avatar...');
  await page.waitForFunction(() => window.__avatarLoaded, { timeout: 30000 });
  console.log('Avatar loaded!');
  
  // Pass animation data
  await page.evaluate((anim) => window.__setAnim(anim), animation);
  
  // Render each frame
  console.log(`Rendering ${totalFrames} frames...`);
  for (let f = 0; f < totalFrames; f++) {
    const t = (f / FPS) % duration;
    await page.evaluate((time) => {
      window.__setTime(time);
      window.__render();
    }, t);
    
    const framePath = path.join(FRAMES_DIR, `frame_${String(f).padStart(4,'0')}.png`);
    await page.screenshot({ path: framePath });
    
    if (f % 10 === 0) process.stdout.write(`\r  Frame ${f}/${totalFrames}`);
  }
  console.log('\nDone rendering!');
  
  await browser.close();
  console.log(`Frames saved to ${FRAMES_DIR}`);
  console.log(`Total: ${totalFrames} frames`);
}

main().catch(console.error);
