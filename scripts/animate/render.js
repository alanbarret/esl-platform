#!/usr/bin/env node
/**
 * Render a GLB animation to MP4 using Puppeteer + Chromium + ffmpeg.
 *
 * Usage:
 *   node render.js <input.glb> <output.mp4> [--fps 24] [--duration 4] [--w 720] [--h 720]
 */

import puppeteer from 'puppeteer';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, basename } from 'node:path';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { fps: 24, w: 720, h: 720, duration: null };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--fps') args.fps = parseInt(argv[++i], 10);
    else if (a === '--w') args.w = parseInt(argv[++i], 10);
    else if (a === '--h') args.h = parseInt(argv[++i], 10);
    else if (a === '--duration') args.duration = parseFloat(argv[++i]);
    else positional.push(a);
  }
  if (positional.length < 2) {
    console.error('Usage: render.js <input.glb> <output.mp4> [--fps 24] [--duration 4] [--w 720] [--h 720]');
    process.exit(1);
  }
  args.input = resolve(positional[0]);
  args.output = resolve(positional[1]);
  return args;
}

async function startStaticServer(modelPath, viewerDir) {
  const modelBytes = await readFile(modelPath);
  const viewerHtml = await readFile(resolve(viewerDir, 'viewer.html'));
  const server = createServer((req, res) => {
    if (req.url === '/' || req.url.startsWith('/viewer')) {
      res.setHeader('content-type', 'text/html');
      res.end(viewerHtml);
    } else if (req.url.startsWith('/model.glb')) {
      res.setHeader('content-type', 'model/gltf-binary');
      res.end(modelBytes);
    } else {
      res.statusCode = 404; res.end();
    }
  });
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  const port = server.address().port;
  return { server, port };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  console.log('Args:', args);

  if (!existsSync(args.input)) {
    console.error('Input not found:', args.input); process.exit(1);
  }

  const tmpDir = resolve(__dirname, '.frames-' + Date.now());
  await mkdir(tmpDir, { recursive: true });
  console.log('Temp frames dir:', tmpDir);

  const { server, port } = await startStaticServer(args.input, __dirname);
  console.log('Static server on', port);

  const url = `http://127.0.0.1:${port}/viewer.html?model=/model.glb&w=${args.w}&h=${args.h}&fps=${args.fps}`;
  console.log('URL:', url);

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
  page.on('console', (msg) => console.log('  [page]', msg.text()));
  page.on('pageerror', (err) => console.error('  [page-error]', err.message));

  await page.goto(url, { waitUntil: 'load' });

  // Wait for the model + animations to be ready
  await page.waitForFunction('window.__renderReady === true || window.__renderError', { timeout: 30000 });
  const err = await page.evaluate('window.__renderError');
  if (err) { console.error('Renderer error:', err); process.exit(2); }

  const duration = args.duration ?? await page.evaluate('window.__animDuration');
  console.log('Animation duration:', duration, 's');

  // Switch to manual mode so the rAF loop doesn't advance time
  await page.evaluate('window.__manualMode = true');

  const totalFrames = Math.max(1, Math.round(duration * args.fps));
  console.log(`Rendering ${totalFrames} frames at ${args.fps}fps -> ${tmpDir}`);

  const bonesByFrame = [];
  let cameraMatrices = null;
  for (let i = 0; i < totalFrames; i++) {
    const t = (i / args.fps);
    const result = await page.evaluate((time) => window.__renderFrameAtTime(time), t);
    const dataUrl = result.img || result; // backward compat
    const bones = result.bones || null;
    bonesByFrame.push(bones);
    if (result.camera && !cameraMatrices) cameraMatrices = result.camera;
    const b64 = dataUrl.split(',', 2)[1];
    const buf = Buffer.from(b64, 'base64');
    const out = resolve(tmpDir, `frame_${String(i).padStart(5, '0')}.png`);
    await writeFile(out, buf);
    if (i % 10 === 0 || i === totalFrames - 1) {
      process.stdout.write(`\r  frame ${i + 1}/${totalFrames}`);
    }
  }
  // Also write bones JSON next to the output mp4
  const bonesPath = args.output.replace(/\.mp4$/, '.bones.json');
  await writeFile(bonesPath, JSON.stringify({ fps: args.fps, width: args.w, height: args.h, camera: cameraMatrices, frames: bonesByFrame }, null, 2));
  console.log('Wrote bones JSON:', bonesPath);
  process.stdout.write('\n');

  await browser.close();
  server.close();

  // Encode with ffmpeg
  console.log('Encoding MP4...');
  const framePattern = resolve(tmpDir, 'frame_%05d.png');
  await new Promise((done, reject) => {
    const ff = spawn('ffmpeg', [
      '-y',
      '-framerate', String(args.fps),
      '-i', framePattern,
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
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
