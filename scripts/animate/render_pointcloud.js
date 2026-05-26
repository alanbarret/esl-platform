#!/usr/bin/env node
/**
 * Render MediaPipe 3D landmarks as a point cloud video using Three.js.
 * Reads the holistic JSON (pose_world + hand_world_landmarks in meters) and
 * displays them as colored spheres connected by lines.
 *
 * Usage:
 *   node render_pointcloud.js <holistic.json> <output.mp4> [--fps 25] [--w 600] [--h 700]
 *                              [--trim-start N] [--trim-end N]
 */

import puppeteer from 'puppeteer';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, basename } from 'node:path';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { fps: 25, w: 600, h: 700, trimStart: 0, trimEnd: null };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--fps') args.fps = parseInt(argv[++i], 10);
    else if (a === '--w') args.w = parseInt(argv[++i], 10);
    else if (a === '--h') args.h = parseInt(argv[++i], 10);
    else if (a === '--trim-start') args.trimStart = parseInt(argv[++i], 10);
    else if (a === '--trim-end') args.trimEnd = parseInt(argv[++i], 10);
    else positional.push(a);
  }
  if (positional.length < 2) {
    console.error('Usage: render_pointcloud.js <holistic.json> <output.mp4> [opts]');
    process.exit(1);
  }
  args.input = resolve(positional[0]);
  args.output = resolve(positional[1]);
  return args;
}

const VIEWER_HTML = `
<!doctype html>
<html><head><meta charset="utf-8" />
<title>MP Point Cloud</title>
<style>html,body{margin:0;background:#1a1a2e;overflow:hidden;} canvas{display:block;}</style>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}
</script>
</head>
<body>
<script type="module">
import * as THREE from 'three';

const W = WIDTH, H = HEIGHT;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// Camera looking at the subject from the front (like the source video).
const camera = new THREE.PerspectiveCamera(35, W / H, 0.01, 100);
camera.position.set(0, 0, 2.0);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.8));

// Subtle ground grid
const grid = new THREE.GridHelper(2, 10, 0x444466, 0x33334a);
grid.position.y = -1.0;
scene.add(grid);

// Pose connections (BlazePose 33-point skeleton, body subset)
const POSE_EDGES = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
];
const HAND_EDGES = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const SPHERE_GEO = new THREE.SphereGeometry(0.012, 8, 8);
const POSE_MAT = new THREE.MeshBasicMaterial({ color: 0xff8080 });
const LH_MAT = new THREE.MeshBasicMaterial({ color: 0x80ff80 });
const RH_MAT = new THREE.MeshBasicMaterial({ color: 0xffb060 });
const LINE_POSE_MAT = new THREE.LineBasicMaterial({ color: 0xff8080 });
const LINE_LH_MAT = new THREE.LineBasicMaterial({ color: 0x80ff80 });
const LINE_RH_MAT = new THREE.LineBasicMaterial({ color: 0xffb060 });

const dynamicGroup = new THREE.Group();
scene.add(dynamicGroup);

function mpToWorld(p) {
  // MediaPipe: +X subject-left, +Y down, +Z away. Three.js: +X right, +Y up, +Z toward camera.
  // Want avatar to face camera (chest in +Z direction).
  return new THREE.Vector3(p[0], -p[1], -p[2]);
}

function buildSkeleton(frameData) {
  // Clear previous frame
  while (dynamicGroup.children.length) {
    const obj = dynamicGroup.children.pop();
    obj.geometry?.dispose?.();
  }
  if (!frameData) return null;

  let poseCenterY = 0;
  let hipMidpoint = null;
  let chest = null;

  // Body (pose_world_landmarks)
  if (frameData.pose) {
    const verts = frameData.pose.map(p => mpToWorld(p));
    // Add subject's pelvis height offset so the figure sits above the grid
    const lhip = verts[23], rhip = verts[24];
    hipMidpoint = new THREE.Vector3().addVectors(lhip, rhip).multiplyScalar(0.5);
    chest = new THREE.Vector3().addVectors(verts[11], verts[12]).multiplyScalar(0.5);
    // Translate everything so hips sit at y=0
    const offset = new THREE.Vector3(-hipMidpoint.x, -hipMidpoint.y, -hipMidpoint.z);
    for (let i = 0; i < verts.length; i++) verts[i].add(offset);
    hipMidpoint.add(offset);
    chest.add(offset);

    // Spheres
    for (let i = 0; i < verts.length; i++) {
      // Skip very fine-grained face points for clarity
      if ([1,2,3,4,5,6,7,8,9,10].includes(i)) continue;
      const m = new THREE.Mesh(SPHERE_GEO, POSE_MAT);
      m.position.copy(verts[i]);
      dynamicGroup.add(m);
    }
    // Edges
    for (const [a, b] of POSE_EDGES) {
      const geo = new THREE.BufferGeometry().setFromPoints([verts[a], verts[b]]);
      dynamicGroup.add(new THREE.Line(geo, LINE_POSE_MAT));
    }

    // Anchor hands at the pose wrist positions using metric scale
    if (frameData.lh) addHand(frameData.lh, verts[15], LH_MAT, LINE_LH_MAT);
    if (frameData.rh) addHand(frameData.rh, verts[16], RH_MAT, LINE_RH_MAT);
  }
}

function addHand(handData, wristWorld, sphereMat, lineMat) {
  // hand_world_landmarks are in meters with origin at hand center.
  // Apply same axis flip and place wrist (index 0) at the pose's wrist position.
  const handVerts = handData.map(p => mpToWorld(p));
  const handWristLocal = handVerts[0].clone();
  // Offset so handVerts[0] lands at wristWorld
  for (let i = 0; i < handVerts.length; i++) {
    handVerts[i].sub(handWristLocal);
    handVerts[i].add(wristWorld);
  }
  const finerGeo = new THREE.SphereGeometry(0.006, 6, 6);
  for (const v of handVerts) {
    const m = new THREE.Mesh(finerGeo, sphereMat);
    m.position.copy(v);
    dynamicGroup.add(m);
  }
  for (const [a, b] of HAND_EDGES) {
    const geo = new THREE.BufferGeometry().setFromPoints([handVerts[a], handVerts[b]]);
    dynamicGroup.add(new THREE.Line(geo, lineMat));
  }
}

window.__renderFrame = function (frameData) {
  buildSkeleton(frameData);

  // Frame: look at the upper body
  if (frameData && frameData.pose) {
    const lhip = mpToWorld(frameData.pose[23]);
    const rhip = mpToWorld(frameData.pose[24]);
    const lsh = mpToWorld(frameData.pose[11]);
    const rsh = mpToWorld(frameData.pose[12]);
    const hipMid = new THREE.Vector3().addVectors(lhip, rhip).multiplyScalar(0.5);
    const shMid = new THREE.Vector3().addVectors(lsh, rsh).multiplyScalar(0.5);
    // Camera target: shoulder midpoint (relative to hip-aligned origin = 0)
    const targetY = shMid.y - hipMid.y;
    // Frame upper body
    const bodyHeight = Math.abs(shMid.y - hipMid.y) * 3.0;
    const distance = bodyHeight / (2 * Math.tan(THREE.MathUtils.degToRad(35) / 2)) / 0.7;
    camera.position.set(0, targetY, distance);
    camera.lookAt(0, targetY * 0.5, 0);
  }

  renderer.render(scene, camera);
  return renderer.domElement.toDataURL('image/png');
};

window.__ready = true;
</script>
</body></html>
`;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  console.log('Args:', args);

  const data = JSON.parse(await readFile(args.input, 'utf-8'));
  let frames = data.frames;
  const trimStart = args.trimStart || 0;
  const trimEnd = args.trimEnd ?? frames.length;
  frames = frames.slice(trimStart, trimEnd);
  console.log(`Frames to render: ${frames.length} (trim=${trimStart}..${trimEnd})`);

  const tmpDir = resolve(__dirname, '.pc-frames-' + Date.now());
  await mkdir(tmpDir, { recursive: true });

  const html = VIEWER_HTML.replace('WIDTH', args.w).replace('HEIGHT', args.h);
  const server = createServer((req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(html);
  });
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  const port = server.address().port;

  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: '/usr/bin/chromium-browser',
    args: [
      '--no-sandbox', '--disable-setuid-sandbox',
      '--enable-unsafe-swiftshader',
      '--use-angle=swiftshader',
      '--use-gl=angle',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--disable-features=Vulkan',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: args.w, height: args.h, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[page-error]', e.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.waitForFunction('window.__ready === true', { timeout: 15000 });

  for (let i = 0; i < frames.length; i++) {
    const dataUrl = await page.evaluate((d) => window.__renderFrame(d), frames[i]);
    const b64 = dataUrl.split(',', 2)[1];
    const buf = Buffer.from(b64, 'base64');
    const out = resolve(tmpDir, `frame_${String(i).padStart(5, '0')}.png`);
    await writeFile(out, buf);
    if (i % 10 === 0 || i === frames.length - 1) {
      process.stdout.write(`\r  frame ${i + 1}/${frames.length}`);
    }
  }
  process.stdout.write('\n');

  await browser.close();
  server.close();

  // Encode
  const framePattern = resolve(tmpDir, 'frame_%05d.png');
  await new Promise((done, reject) => {
    const ff = spawn('ffmpeg', [
      '-y', '-framerate', String(args.fps),
      '-i', framePattern,
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
      '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
      '-crf', '20',
      args.output,
    ], { stdio: ['ignore', 'pipe', 'pipe'] });
    ff.stderr.on('data', (d) => process.stderr.write(d));
    ff.on('exit', (code) => code === 0 ? done() : reject(new Error('ffmpeg exit ' + code)));
  });
  console.log('✅ Wrote', args.output);
  await rm(tmpDir, { recursive: true, force: true });
}

main().catch((e) => { console.error(e); process.exit(1); });
