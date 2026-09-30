from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os

from api.common import handle_chat, json_response


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        if self.path.startswith("/api/v1/chat"):
            handle_chat(self, True)
        else:
            json_response(self, {"ok": True})

    def do_GET(self):
        if self.path == "/api/status":
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
                },
            )
        elif self.path == "/health":
            json_response(self, {"ok": True})
        else:
            json_response(self, {"ok": True, "service": "Demo AI"})

    def do_POST(self):
        if self.path == "/api/chat":
            handle_chat(self, False)
        elif self.path == "/api/v1/chat":
            handle_chat(self, True)
        elif self.path == "/api/key":
            from api.key import handler as KeyHandler
            KeyHandler.do_POST(self)
        else:
            json_response(self, {"error": "not_found"}, 404)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
