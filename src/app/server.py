import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

MODEL_DIR = Path(os.getenv('ASCLEPIOS_MODEL_DIR', ROOT_DIR / 'models'))
MODEL_PATH = Path(os.getenv('ASCLEPIOS_MODEL_PATH', MODEL_DIR / 'asclepios_model'))
LORA_PATH = Path(os.getenv('ASCLEPIOS_LORA_PATH', MODEL_DIR / 'asclepios_lora'))
DATA_DIR = Path(os.getenv('ASCLEPIOS_DATA_DIR', ROOT_DIR / 'data'))


def fallback_reply(message: str) -> str:
    lowered = message.lower()

    if re.search(r'st\w*omach', lowered):
        return (
            "For a mild stomach ache, sip water, eat small bland meals, and avoid "
            "alcohol, greasy food, and medicines such as ibuprofen until you know "
            "what is causing it. If it is not improving within 24 to 48 hours, or "
            "keeps returning, contact a clinician. Seek urgent care now for severe "
            "or worsening pain, a rigid or swollen abdomen, repeated vomiting, "
            "blood in vomit or stool, black stool, fainting, fever with severe pain, "
            "or pain concentrated in the lower right abdomen. How long has it been "
            "going on, and where is the pain located?"
        )

    return (
        "I can give general information, but I need more detail to make the answer "
        "useful. Please include the main symptom or question, when it started, how "
        "severe it is, and anything that makes it better or worse. Seek urgent care "
        "now for trouble breathing, chest pressure, sudden weakness, confusion, "
        "severe bleeding, or a rapidly worsening condition."
    )


def load_model_reply(message: str) -> str:
    """Try to use the trained model if available; otherwise fall back to a safe assistant reply."""
    try:
        import sys

        model_dir = ROOT_DIR / "models"
        project_root = str(ROOT_DIR)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.model.asclepios_model import get_model_response, load_model_with_fallback

        candidate_paths = [
            MODEL_PATH,
            LORA_PATH,
            model_dir / "asclepios_model",
            model_dir / "asclepios_lora",
            ROOT_DIR / "src" / "model",
        ]

        for path in candidate_paths:
            if path.exists():
                try:
                    model, tokenizer = load_model_with_fallback(str(path), str(ROOT_DIR / "src" / "model"))
                    return get_model_response(model, tokenizer, message)
                except Exception:
                    continue

        return fallback_reply(message)
    except Exception:
        return fallback_reply(message)


def handle_chat_message(message: str) -> dict:
    cleaned = re.sub(r'\s+', ' ', (message or '').strip())
    if not cleaned:
        return {"error": "Please provide a message."}, 400

    lower = cleaned.lower()
    crisis_phrases = [
        "suicide", "kill myself", "want to die", "end my life", "hurt myself",
        "self harm", "self-harm", "can't go on", "cant go on"
    ]
    if any(phrase in lower for phrase in crisis_phrases):
        return {
            "reply": "I am really concerned about what you shared. Please contact a crisis line or emergency services right away, or go to the nearest emergency room. In the US call or text 988 for immediate support.",
            "model": "Asclepios AI safety assistant",
            "safe": True,
        }, 200

    reply = load_model_reply(cleaned)
    return {
        "reply": reply,
        "model": "Asclepios AI",
        "safe": True,
    }, 200


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/health':
            self._send_json({"status": "ok", "service": "asclepios-ai"})
            return

        if parsed.path in ('/', '/index.html'):
            self._serve_file(BASE_DIR / 'index.html', 'text/html; charset=utf-8')
            return

        if parsed.path == '/stylesheet.css':
            self._serve_file(BASE_DIR / 'stylesheet.css', 'text/css; charset=utf-8')
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not found')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/api/chat':
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(content_length) if content_length > 0 else b''
            payload = json.loads(raw.decode('utf-8')) if raw else {}
            message = str(payload.get('message', payload.get('prompt', '')) or '')
            response, status_code = handle_chat_message(message)
            self._send_json(response, status_code=status_code)
        except Exception:
            self._send_json({"error": "Invalid JSON body."}, status_code=400)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, payload, status_code=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path, content_type: str):
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')

    def log_message(self, *args, **kwargs):
        return


def main():
    port = int(os.environ.get('PORT', '8000'))
    server = ThreadingHTTPServer(('0.0.0.0', port), AppHandler)
    print(f'Asclepios AI backend is running at http://localhost:{port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
