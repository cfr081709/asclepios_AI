const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = Number(process.env.PORT || 8000);
const MODEL_PORT = Number(process.env.ASCLEPIOS_MODEL_PORT || 8001);
const APP_ROOT = __dirname;
const PUBLIC_DIR = APP_ROOT;

const pythonProcess = spawn(
  process.env.PYTHON || 'python3',
  ['src/app/server.py'],
  {
    cwd: path.resolve(APP_ROOT, '..', '..'),
    env: { ...process.env, PORT: String(MODEL_PORT) },
    stdio: 'inherit'
  }
);

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  res.end(JSON.stringify(payload));
}

function sanitizeMessage(raw) {
  if (typeof raw !== 'string') return '';
  return raw.trim().replace(/\s+/g, ' ');
}

function serveFile(res, filePath) {
  fs.readFile(filePath, (error, content) => {
    if (error) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }

    const extension = path.extname(filePath).toLowerCase();
    const contentTypes = {
      '.html': 'text/html; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
      '.js': 'application/javascript; charset=utf-8',
      '.json': 'application/json; charset=utf-8',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.svg': 'image/svg+xml'
    };

    res.writeHead(200, { 'Content-Type': contentTypes[extension] || 'text/plain; charset=utf-8' });
    res.end(content);
  });
}

function handleApiChat(req, res) {
  let body = '';

  req.on('data', (chunk) => {
    body += chunk;
    if (body.length > 1e6) {
      req.destroy();
    }
  });

  req.on('end', () => {
    try {
      const proxyRequest = http.request(
        {
          hostname: '127.0.0.1',
          port: MODEL_PORT,
          path: '/api/chat',
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(body)
          }
        },
        (proxyResponse) => {
          let responseBody = '';
          proxyResponse.setEncoding('utf8');
          proxyResponse.on('data', (chunk) => { responseBody += chunk; });
          proxyResponse.on('end', () => {
            res.writeHead(proxyResponse.statusCode || 502, {
              'Content-Type': 'application/json; charset=utf-8',
              'Access-Control-Allow-Origin': '*'
            });
            res.end(responseBody);
          });
        }
      );

      proxyRequest.on('error', () => {
        sendJson(res, 503, {
          error: 'The Python model service is unavailable.'
        });
      });
      proxyRequest.end(body);
    } catch (error) {
      sendJson(res, 400, { error: 'Invalid JSON body.' });
    }
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    });
    res.end();
    return;
  }

  if (url.pathname === '/api/health') {
    sendJson(res, 200, {
      status: 'ok',
      service: 'asclepios-ai',
      message: 'Backend is running.'
    });
    return;
  }

  if (url.pathname === '/api/chat') {
    if (req.method !== 'POST') {
      sendJson(res, 405, { error: 'Method not allowed. Use POST.' });
      return;
    }

    handleApiChat(req, res);
    return;
  }

  if (url.pathname === '/' || url.pathname === '/index.html') {
    serveFile(res, path.join(PUBLIC_DIR, 'index.html'));
    return;
  }

  if (url.pathname === '/stylesheet.css') {
    serveFile(res, path.join(PUBLIC_DIR, 'stylesheet.css'));
    return;
  }

  if (url.pathname === '/favicon.ico') {
    res.writeHead(204);
    res.end();
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end('Route not found');
});

server.listen(PORT, () => {
  console.log(`Asclepios AI backend is running at http://localhost:${PORT}`);
});

function shutdown() {
  pythonProcess.kill();
  process.exit();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);