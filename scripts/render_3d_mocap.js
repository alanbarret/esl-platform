/**
 * Headless render: load mocap-render.html in Chromium,
 * capture each frame as PNG, then stitch to MP4 with ffmpeg.
 */
const puppeteer = require('/usr/lib/node_modules/puppeteer');
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const FRAMES_DIR = '/tmp/mocap_frames';
const OUTPUT_MP4 = '/root/.openclaw/workspace/avatar_3d_motion.mp4';
const PAGE_URL = 'http://localhost:3001/mocap-render.html';

async function main() {
  fs.mkdirSync(FRAMES_DIR, { recursive: true });
  // Clean old frames
  fs.readdirSync(FRAMES_DIR).forEach(f => fs.unlinkSync(path.join(FRAMES_DIR,f)));

  console.log('Launching headless Chromium...');
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium-browser',
    args: [
      '--no-sandbox','--disable-setuid-sandbox',
      '--enable-webgl','--use-gl=swiftshader',
      '--enable-accelerated-2d-canvas',
      '--disable-gpu-sandbox',
      '--window-size=640,720',
    ],
    headless: true,
  });

  const page = await browser.newPage();
  await page.setViewport({width:640, height:720});

  // Load page
  console.log('Loading page...');
  await page.goto(PAGE_URL, {waitUntil:'networkidle0', timeout:30000});

  // Wait for motion data to load
  await page.waitForFunction(() => {
    const s = document.getElementById('status');
    return s && s.textContent.includes('Ready');
  }, {timeout:20000});

  const motionData = await page.evaluate(() => window._motionLoaded || true);
  console.log('Motion data loaded. Starting frame capture...');

  // Expose frame capture function
  const frameCount = await page.evaluate(() => {
    return fetch('/motion_data.json').then(r=>r.json()).then(d => d.frames.length);
  });
  console.log(`Total frames: ${frameCount}`);

  // Inject capture loop into page
  await page.evaluate(() => {
    window._captureReady = false;
    window._captureFrameIdx = 0;
    window._captureDone = false;
  });

  // Capture frame by frame via DevTools
  for (let fi = 0; fi < frameCount; fi++) {
    await page.evaluate((fi) => {
      if(window.applyFrame) window.applyFrame(fi);
      if(window.renderer && window.scene && window.camera) {
        window.renderer.render(window.scene, window.camera);
      }
    }, fi);

    // Small wait for WebGL to flush
    await new Promise(r => setTimeout(r, 16));

    const shot = await page.screenshot({
      clip: {x:0, y:0, width:640, height:720},
      type: 'png',
    });
    const framePath = path.join(FRAMES_DIR, `frame_${String(fi).padStart(5,'0')}.png`);
    fs.writeFileSync(framePath, shot);

    if ((fi+1) % 25 === 0) {
      process.stdout.write(`\r  Captured ${fi+1}/${frameCount} frames`);
    }
  }
  console.log(`\nCapture done. Assembling MP4...`);
  await browser.close();

  // Stitch frames to MP4
  execSync(`ffmpeg -y -framerate 25 -i ${FRAMES_DIR}/frame_%05d.png \
    -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
    "${OUTPUT_MP4}" 2>/dev/null`);

  const sz = Math.round(fs.statSync(OUTPUT_MP4).size / 1024);
  console.log(`Done! ${OUTPUT_MP4} (${sz}KB)`);
}

main().catch(e => { console.error(e); process.exit(1); });
