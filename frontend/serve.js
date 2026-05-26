#!/usr/bin/env node
/**
 * Static server for the built React platform UI.
 * Serves frontend/dist/ on port 3001 and proxies /api/* to the lean backend.
 *
 * Usage: node frontend/serve.js
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.PORT || '3001', 10);
const API_TARGET = process.env.API_TARGET || 'http://localhost:8001';
const DIST = path.join(__dirname, 'dist');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json',
  '.mp4': 'video/mp4',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ico': 'image/x-icon',
};

function proxyApi(clientReq, clientRes) {
  const url = new URL(API_TARGET);
  const opts = {
    hostname: url.hostname,
    port: url.port || 80,
    path: clientReq.url,
    method: clientReq.method,
    headers: { ...clientReq.headers, host: url.host },
  };
  const proxyReq = http.request(opts, (proxyRes) => {
    clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(clientRes, { end: true });
  });
  proxyReq.on('error', (err) => {
    console.error('[proxy]', err.message);
    clientRes.writeHead(502, { 'content-type': 'application/json' });
    clientRes.end(JSON.stringify({ error: 'proxy failed', detail: err.message }));
  });
  clientReq.pipe(proxyReq, { end: true });
}

const server = http.createServer((req, res) => {
  // CORS for everything
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.statusCode = 204; res.end(); return; }

  // Proxy API calls
  if (req.url.startsWith('/api/')) {
    return proxyApi(req, res);
  }

  // Static
  let url = decodeURI(req.url.split('?')[0]);
  if (url === '/' || url.endsWith('/')) url = '/index.html';
  const filePath = path.join(DIST, url);
  if (!filePath.startsWith(DIST)) { res.statusCode = 403; res.end('forbidden'); return; }
  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      // SPA fallback: serve index.html for unknown routes (no file extension)
      if (!path.extname(url)) {
        return fs.readFile(path.join(DIST, 'index.html'), (e2, data) => {
          if (e2) { res.statusCode = 404; res.end('not found'); return; }
          res.setHeader('Content-Type', 'text/html; charset=utf-8');
          res.end(data);
        });
      }
      res.statusCode = 404; res.end('not found'); return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.setHeader('Content-Type', MIME[ext] || 'application/octet-stream');
    res.setHeader('Cache-Control', ext === '.html' ? 'no-cache' : 'public, max-age=300');
    fs.createReadStream(filePath).pipe(res);
  });
});

server.listen(PORT, () => {
  console.log(`[esl-frontend] listening on :${PORT} | dist=${DIST} | api -> ${API_TARGET}`);
});
