from http.server import BaseHTTPRequestHandler
from api.common import handle_chat, json_response


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        handle_chat(self, False)

    def do_POST(self):
        handle_chat(self, False)

    def do_GET(self):
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
