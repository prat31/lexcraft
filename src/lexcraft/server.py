"""Minimal OpenAI-compatible server for local models (stdlib only).

Lets Promptfoo (or any OpenAI client) talk to the local LoRA model exactly like a
frontier API, so every system in an eval goes through the same code path.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lexcraft.backends.base import Backend
from lexcraft.verify import clean_output


def make_handler(backend: Backend, model_name: str, default_max_tokens: int = 1024):
    class Handler(BaseHTTPRequestHandler):
        server_version = "lexcraft"

        def log_message(self, fmt: str, *args: object) -> None:  # quiet
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            elif self.path.rstrip("/") in ("/health", ""):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length) or b"{}")
                messages = req["messages"]
                max_tokens = int(req.get("max_tokens") or default_max_tokens)
                temperature = float(req.get("temperature") or 0.0)
            except (ValueError, KeyError) as e:
                self._send(400, {"error": {"message": f"bad request: {e}"}})
                return
            t0 = time.perf_counter()
            try:
                text = clean_output(backend.generate(messages, max_tokens, temperature))
            except Exception as e:  # surface backend errors as 500 instead of crashing
                self._send(500, {"error": {"message": str(e)}})
                return
            self._send(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"latency_ms": round((time.perf_counter() - t0) * 1000)},
                },
            )

    return Handler


def serve(backend: Backend, model_name: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(backend, model_name))
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
