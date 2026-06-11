#!/usr/bin/env node
/**
 * Dev server for dist/ with no-cache headers on HTML.
 * Eliminates the need for hard-refresh after every code change.
 *
 * Usage: node scripts/serve.js [port]
 */
const http = require('http');
const fs   = require('fs');
const path = require('path');

const PORT = parseInt(process.argv[2] ?? '8082', 10);
const ROOT = path.join(__dirname, '..', 'dist');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.bin':  'application/octet-stream',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

http.createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  if (urlPath === '/') urlPath = '/index.html';
  const filePath = path.join(ROOT, urlPath);

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }

    const ext  = path.extname(filePath).toLowerCase();
    const mime = MIME[ext] ?? 'application/octet-stream';
    const isHtml = ext === '.html';

    res.writeHead(200, {
      'Content-Type':  mime,
      // HTML: never cache — always fetch fresh so changes appear on normal reload
      // Static assets: short cache (they use ?t= busting anyway)
      'Cache-Control': isHtml ? 'no-cache, no-store, must-revalidate' : 'public, max-age=60',
      'Pragma':        isHtml ? 'no-cache' : '',
    });
    res.end(data);
  });
}).listen(PORT, () => {
  console.log(`Ghost dev server → http://localhost:${PORT}`);
  console.log('HTML served with no-cache — normal reload (Cmd+R) is enough.');
});
