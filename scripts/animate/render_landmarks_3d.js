#!/usr/bin/env node
/**
 * Render MediaPipe 3D landmarks (pose_world + hands) as a 3D skeleton overlay
 * in Three.js. We project them through the SAME camera used to render the avatar,
 * so the resulting frames can be composited as a 3D overlay on both the source
 * video and the avatar render.
 *
 * Output: a sequence of transparent PNG frames containing only the 3D skeleton.
 *
 * Usage:
 *   node render_landmarks_3d.js <holistic.json> <out_dir> [--fps 25] [--w 600] [--h 700]
 *                                [--trim-start N] [--mode anchored-to-pose]
 */

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { fps: 25, w: 600, h: 700, trimStart: 0 };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--fps') args.fps = parseInt(argv[++i], 10);
    else if (a === '--w') args.w = parseInt(argv[++i], 10);
    else if (a === '--h') args.h = parseInt(argv[++i], 10);
    else if (a === '--trim-start') args.trimStart = parseInt(argv[++i], 10);
    else positional.push(a);
  }
  if (positional.length < 2) {
    console.error('Usage: render_landmarks_3d.js <holistic.json> <out_dir> [opts]');
    process.exit(1);
  }
  args.holistic = resolve(positional[0]);
  args.outDir = resolve(positional[1]);
  return args;
}

const VIEWER_HTML = `
<!doctype html>
<html><head><meta charset="utf-8"/>
<style>html,body{margin:0;background:rgba(0,0,0,0);} canvas{display:block;}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}</script>
</head>
<body>
<script type="module">
import * as THREE from 'three';

const W = WIDTH, H = HEIGHT;
const scene = new THREE.Scene();
// Transparent background
scene.background = null;

// Camera: match the avatar viewer's camera (35° fov, look at character chest)
const camera = new THREE.PerspectiveCamera(35, W / H, 0.01, 100);
// Position will be set per-frame to follow shoulder anchor

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, premultipliedAlpha: false, preserveDrawingBuffer: true });
renderer.setClearColor(0x000000, 0);
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

// MediaPipe Pose: BlazePose 33 landmarks. We define edges.
const POSE_EDGES = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
];
// Hand 21 landmarks. Edges.
const HAND_EDGES = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// Containers we will populate per frame
const poseGroup = new THREE.Group();
const lhGroup = new THREE.Group();
const rhGroup = new THREE.Group();
scene.add(poseGroup, lhGroup, rhGroup);

const SPHERE_GEO = new THREE.SphereGeometry(0.012, 8, 8);
const POSE_MAT = new THREE.MeshBasicMaterial({ color: 0xff6464 });
const LH_MAT = new THREE.MeshBasicMaterial({ color: 0x64ff64 });
const RH_MAT = new THREE.MeshBasicMaterial({ color: 0xffaa64 });

const LINE_POSE_MAT = new THREE.LineBasicMaterial({ color: 0xff8080, linewidth: 2 });
const LINE_LH_MAT = new THREE.LineBasicMaterial({ color: 0x80ff80, linewidth: 2 });
const LINE_RH_MAT = new THREE.LineBasicMaterial({ color: 0xffc080, linewidth: 2 });

// Important: MediaPipe pose_world: +X subject-left, +Y down, +Z away from camera.
// glTF / Three.js: +Y up, +Z toward viewer.
// Transform: (x, y, z) -> (x, -y, -z)
function mpToGltf(p) { return new THREE.Vector3(p[0], -p[1], -p[2]); }

// Hand landmarks (image-relative): origin at wrist, +X = image right, +Y = image down, z roughly normalized to x.
// They are NOT in the same metric world as pose_world. We anchor each hand at the pose's
// wrist location (in world), and use the relative landmarks scaled to physical hand size.
function buildPose(pose) {
  poseGroup.clear();
  if (!pose) return null;
  const verts = pose.map(p => mpToGltf(p));
  // Add spheres
  for (const v of verts) {
    const m = new THREE.Mesh(SPHERE_GEO, POSE_MAT);
    m.position.copy(v); poseGroup.add(m);
  }
  // Add edges
  for (const [a, b] of POSE_EDGES) {
    const geo = new THREE.BufferGeometry().setFromPoints([verts[a], verts[b]]);
    poseGroup.add(new THREE.Line(geo, LINE_POSE_MAT));
  }
  return verts;
}

function buildHand(hand, side, poseWristWorld, poseElbowWorld) {
  const group = side === 'lh' ? lhGroup : rhGroup;
  const sphereMat = side === 'lh' ? LH_MAT : RH_MAT;
  const lineMat = side === 'lh' ? LINE_LH_MAT : LINE_RH_MAT;
  group.clear();
  if (!hand || !poseWristWorld) return;

  // Hand landmark coords are image-normalized. Use them only relative to the wrist.
  // We need to estimate a hand-size scale: pose_world forearm length is roughly 0.27m,
  // and hand length is roughly forearm * 0.7.
  const forearmLen = poseElbowWorld ? poseWristWorld.distanceTo(poseElbowWorld) : 0.25;
  const handScale = forearmLen * 0.7;

  // Build a hand-local frame using MCP positions in image coords.
  const wrist = hand[0];
  const middle = hand[9];
  const index = hand[5];
  const pinky = hand[17];

  // In image coords: image +Y is DOWN, so flip Y so +Y_local = up.
  const toLocal = (p) => new THREE.Vector3(p[0] - wrist[0], -(p[1] - wrist[1]), -(p[2] - wrist[2]));
  const yLocal = toLocal(middle).normalize(); // wrist->middle = "up the hand"
  const idxV = toLocal(index);
  const pkyV = toLocal(pinky);
  const xLocalRaw = pkyV.clone().sub(idxV); // pinky -> index... wait, we want pinky->index
  // Actually pinky_to_index = index - pinky:
  xLocalRaw.copy(idxV).sub(pkyV);
  // Orthogonalize against yLocal
  const xLocal = xLocalRaw.clone().sub(yLocal.clone().multiplyScalar(xLocalRaw.dot(yLocal))).normalize();
  const zLocal = new THREE.Vector3().crossVectors(xLocal, yLocal).normalize();

  // We need to place the hand in WORLD so its wrist aligns with poseWristWorld.
  // For orientation, we use the pose's forearm direction as the hand's "up" in world.
  // Then we apply the local hand frame as a rotation about that.
  // For simplicity here, since pose already gives us a forearm direction, we'll:
  //   - place wrist at poseWristWorld
  //   - orient yLocal toward poseElbowWorld -> poseWristWorld direction (forearm extended)
  //   - allow xLocal/zLocal twist freely (carries the in-plane hand rotation)
  //
  // To do this we build a quaternion from the local frame (xLocal, yLocal, zLocal)
  // to the world frame where Y_world ≈ forearm direction, X_world and Z_world are perpendicular.
  let forearmDir = new THREE.Vector3(0, 1, 0);
  if (poseElbowWorld) {
    forearmDir = poseWristWorld.clone().sub(poseElbowWorld).normalize();
  }

  // Pick an X_world perpendicular to forearmDir using world UP as helper.
  let worldUp = new THREE.Vector3(0, 1, 0);
  if (Math.abs(forearmDir.dot(worldUp)) > 0.95) worldUp.set(1, 0, 0);
  const xWorld = new THREE.Vector3().crossVectors(worldUp, forearmDir).normalize();
  const zWorld = new THREE.Vector3().crossVectors(xWorld, forearmDir).normalize();
  // Build basis matrix world (columns x,y,z): rotation that maps local to world
  const M = new THREE.Matrix4();
  M.makeBasis(xWorld, forearmDir, zWorld);
  // Now M maps (1,0,0) local -> xWorld, (0,1,0) local -> forearmDir, (0,0,1) local -> zWorld.
  // We want to convert each hand landmark's local coords (in xLocal, yLocal, zLocal basis)
  // into world. First, project local hand landmark to (a, b, c) in (xLocal, yLocal, zLocal).
  // Then world position = wrist + (a * xWorld + b * forearmDir + c * zWorld) * handScale.

  // But hand landmarks are noisy in z; the user's diagnosis said use only x,y (image plane).
  // So we'll just place each landmark at wrist + a * xWorld + b * forearmDir, scaled.

  const positions = hand.map((p) => {
    const local = toLocal(p);
    // Project to (xLocal, yLocal) basis to get hand-local 2D
    const a = local.dot(xLocal);
    const b = local.dot(yLocal);
    return poseWristWorld.clone()
      .add(xWorld.clone().multiplyScalar(a * handScale * 5))
      .add(forearmDir.clone().multiplyScalar(b * handScale * 5));
  });

  for (const v of positions) {
    const s = new THREE.Mesh(SPHERE_GEO, sphereMat);
    s.position.copy(v); group.add(s);
  }
  for (const [a, b] of HAND_EDGES) {
    const geo = new THREE.BufferGeometry().setFromPoints([positions[a], positions[b]]);
    group.add(new THREE.Line(geo, lineMat));
  }
}

window.__renderFrame = function(frameData, anchorY) {
  poseGroup.clear(); lhGroup.clear(); rhGroup.clear();
  if (!frameData || !frameData.pose) {
    renderer.render(scene, camera);
    return renderer.domElement.toDataURL('image/png');
  }
  const poseVerts = buildPose(frameData.pose);

  // Set camera position to match avatar viewer: center on mid_shoulder, distance = 1.6 * subject height
  const hipMid = new THREE.Vector3().addVectors(poseVerts[23], poseVerts[24]).multiplyScalar(0.5);
  const shMid = new THREE.Vector3().addVectors(poseVerts[11], poseVerts[12]).multiplyScalar(0.5);
  const heightApprox = Math.abs(poseVerts[0].y - hipMid.y) * 2.5; // nose to hips x2.5 ≈ body height
  // Position camera in front of subject
  camera.position.set(0, shMid.y + 0.05, heightApprox * 1.4);
  camera.lookAt(0, shMid.y - 0.1, 0);

  // Hands
  const lWrist = poseVerts[15];
  const lElbow = poseVerts[13];
  const rWrist = poseVerts[16];
  const rElbow = poseVerts[14];
  if (frameData.lh) buildHand(frameData.lh, 'lh', lWrist, lElbow);
  if (frameData.rh) buildHand(frameData.rh, 'rh', rWrist, rElbow);

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

  const holistic = JSON.parse(await readFile(args.holistic, 'utf-8'));
  const frames = holistic.frames.slice(args.trimStart);
  console.log(`Frames to render: ${frames.length}`);

  await mkdir(args.outDir, { recursive: true });

  const html = VIEWER_HTML.replace('WIDTH', args.w).replace('HEIGHT', args.h);
  const server = createServer((req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(html);
  });
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  const port = server.address().port;
  console.log('Static server on', port);

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
    const data = frames[i];
    const dataUrl = await page.evaluate((d) => window.__renderFrame(d), data);
    const b64 = dataUrl.split(',', 2)[1];
    const buf = Buffer.from(b64, 'base64');
    const out = resolve(args.outDir, `frame_${String(i).padStart(5, '0')}.png`);
    await writeFile(out, buf);
    if (i % 10 === 0 || i === frames.length - 1) {
      process.stdout.write(`\r  frame ${i+1}/${frames.length}`);
    }
  }
  process.stdout.write('\n');

  await browser.close();
  server.close();
  console.log('Done. Frames in', args.outDir);
}

main().catch((e) => { console.error(e); process.exit(1); });
