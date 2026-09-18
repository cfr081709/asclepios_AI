const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const APP_ROOT = __dirname;
const PUBLIC_DIR = APP_ROOT;

const CRISIS_KEYWORDS = [
  'suicide', 'kill myself', 'want to die', 'end my life', 'hurt myself',
  'self harm', 'self-harm', 'cant go on', 'can\'t go on'
];

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

function containsCrisisLanguage(message) {
  const lower = message.toLowerCase();
  return CRISIS_KEYWORDS.some((item) => lower.includes(item));
}

function safeGeneralReply(message) {
  const lower = message.toLowerCase();

  if (lower.includes('pain') || lower.includes('headache') || lower.includes('chest pain')) {
    return 'I can provide general health information, but not diagnosis. If your pain is severe, sudden, or you feel unwell, please get urgent medical care or call emergency services.';
  }

  if (lower.includes('fever') || lower.includes('cough') || lower.includes('flu') || lower.includes('cold')) {
    return 'For general symptom questions, rest, hydration, and monitoring are helpful. Seek medical advice if symptoms are severe, worsening, or last more than a few days.';
  }

  if (lower.includes('medication') || lower.includes('prescription') || lower.includes('medicine')) {
    return 'Medication guidance should be personalized to your medical history. Please check with a clinician or pharmacist before starting or changing a treatment.';
  }

  return 'I can share general wellness information, but I cannot diagnose medical conditions or replace a doctor. If your symptoms feel serious or worsening, please contact a healthcare professional.';
}

function generateMedicalReply(message) {
  const clean = sanitizeMessage(message);

  if (!clean) {
    return 'Please share the health question or symptom you want help with.';
  }

  if (containsCrisisLanguage(clean)) {
    return 'I am really concerned about what you shared. Please contact a crisis line or emergency services right away, or go to the nearest emergency room. In the US, call or text 988 for immediate support.';
  }

  return safeGeneralReply(clean);
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
      const data = body ? JSON.parse(body) : {};
      const message = sanitizeMessage(data.message || data.prompt || '');

      if (!message) {
        sendJson(res, 400, { error: 'Please provide a message.' });
        return;
      }

      const reply = generateMedicalReply(message);
      sendJson(res, 200, {
        reply,
        model: 'Asclepios AI fallback assistant',
        safe: true
      });
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

module.exports = { generateMedicalReply, containsCrisisLanguage };