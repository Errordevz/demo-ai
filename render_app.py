from __future__ import annotations

import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from api.common import client_key, generate_api_key, handle_chat, json_response, rate_limiter

ROOT = Path(__file__).resolve().parent

STATIC = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        path = urlparse(self.path).path
        if path in {"/api/chat", "/api/key", "/api/v1/chat"}:
            handle_chat(self, path == "/api/v1/chat")
        else:
            json_response(self, {"ok": True})

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            json_response(self, {"ok": True})
            return

        if path == "/api/status":
            json_response(
                self,
                {
                    "ok": True,
                    "service": "Demo AI",
                    "model": "demo-ai-100m-moe",
                    "parameters": 100000000,
                    "context": 1024,
                    "architecture": "sparse_moe",
                    "experts": 4,
                    "top_k": 2,
                    "active_expert_fraction": 0.5,
                    "cloud_chat": True,
                    "api_key_required_for_web_chat": False,
                    "api_key_required_for_developer_api": True,
                    "checkpoint_status": "development",
                },
            )
            return

        if path == "/api/chat":
            json_response(
                self,
                {
                    "ok": True,
                    "model": "demo-ai-100m-moe",
                    "architecture": "sparse_moe",
                    "experts": 4,
                    "top_k": 2,
                    "auth": "not_required_for_web_chat",
                },
            )
            return

        if path == "/api/v1/chat":
            json_response(
                self,
                {
                    "ok": True,
                    "model": "demo-ai-100m-moe",
                    "architecture": "sparse_moe",
                    "experts": 4,
                    "top_k": 2,
                    "auth": "DEMO_AI_KEY_required",
                },
            )
            return

        filename = STATIC.get(path)
        if filename:
            self._serve_static(filename)
            return

        json_response(self, {"ok": False, "error": "not_found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/chat":
            handle_chat(self, False)
            return

        if path == "/api/v1/chat":
            handle_chat(self, True)
            return

        if path == "/api/key":
            try:
                if not rate_limiter.allow("key:" + client_key(self), 5):
                    json_response(
                        self,
                        {"error": "rate_limited", "retry_after_seconds": 60},
                        429,
                        {"Retry-After": "60"},
                    )
                    return
                key = generate_api_key()
                json_response(
                    self,
                    {
                        "key": key,
                        "type": "DEMO_AI_KEY",
                        "note": "Store this key securely. This hosted demo keeps key validation in deployment memory.",
                    },
                    201,
                )
            except Exception as exc:
                json_response(
                    self,
                    {"error": "key_service_unavailable", "message": str(exc)},
                    503,
                )
            return

        json_response(self, {"ok": False, "error": "not_found"}, 404)

    def _serve_static(self, filename):
        target = (ROOT / filename).resolve()
        if ROOT not in target.parents and target != ROOT:
            json_response(self, {"error": "forbidden"}, 403)
            return
        if not target.is_file():
            json_response(self, {"error": "not_found"}, 404)
            return

        body = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        if target.suffix == ".js":
            content_type = "application/javascript"
        content_type = content_type or "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
