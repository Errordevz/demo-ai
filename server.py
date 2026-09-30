from __future__ import annotations
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from api.common import client_key, generate_api_key, handle_chat, json_response, rate_limiter

ROOT=Path(__file__).resolve().parent
STATIC_ROOT=ROOT
STATIC={
    "/":("index.html","text/html; charset=utf-8"),
    "/index.html":("index.html","text/html; charset=utf-8"),
    "/styles.css":("styles.css","text/css; charset=utf-8"),
    "/app.js":("app.js","application/javascript; charset=utf-8"),
}

class Handler(BaseHTTPRequestHandler):
    server_version="DemoAI/1.0"

    def do_OPTIONS(self):
        path=urlparse(self.path).path
        if path in {"/api/chat","/api/key","/api/v1/chat"}:
            json_response(self,{"ok":True}); return
        self.send_response(204); self.end_headers()

    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/status":
            json_response(self,{
                "ok":True,
                "service":"Demo AI",
                "model":"demo-ai-100m",
                "parameters":100000000,
                "context":1024,
                "cloud_chat":True,
                "api_key_required_for_web_chat":False,
                "api_key_required_for_developer_api":True,
                "checkpoint_status":"development"
            }); return
        if path=="/api/v1/chat":
            json_response(self,{"ok":True,"model":"demo-ai-100m","auth":"DEMO_AI_KEY_required"}); return
        if path in STATIC:
            self._serve(STATIC[path][0],STATIC[path][1]); return
        self.send_error(404,"Not Found")

    def do_POST(self):
        path=urlparse(self.path).path
        if path=="/api/chat":
            handle_chat(self,False); return
        if path=="/api/v1/chat":
            handle_chat(self,True); return
        if path=="/api/key":
            try:
                if not rate_limiter.allow("key:"+client_key(self),5):
                    json_response(self,{"error":"rate_limited","retry_after_seconds":60},429,{"Retry-After":"60"}); return
                key=generate_api_key()
                json_response(self,{
                    "key":key,
                    "type":"DEMO_AI_KEY",
                    "note":"Store this key securely. This hosted demo keeps key validation in deployment memory."
                },201)
            except Exception as exc:
                json_response(self,{"error":"key_service_unavailable","message":str(exc)},503)
            return
        self.send_error(404,"Not Found")

    def _serve(self,name,content_type):
        target=(STATIC_ROOT/name).resolve()
        if STATIC_ROOT not in target.parents and target!=STATIC_ROOT:
            self.send_error(403,"Forbidden"); return
        if not target.is_file():
            self.send_error(404,"Not Found"); return
        body=target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

def main():
    port=int(os.getenv("PORT","7860"))
    server=ThreadingHTTPServer(("0.0.0.0",port),Handler)
    print(f"Demo AI listening on :{port}",flush=True)
    server.serve_forever()

if __name__=="__main__":
    main()
