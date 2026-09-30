from http.server import BaseHTTPRequestHandler
from api.common import generate_api_key,json_response,rate_limiter,client_key

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        json_response(self,{"ok":True})
    def do_POST(self):
        try:
            if not rate_limiter.allow("key:"+client_key(self),5):
                json_response(self,{"error":"rate_limited"},429,{"Retry-After":"60"}); return
            key=generate_api_key()
            json_response(self,{"key":key,"type":"DEMO_AI_KEY","note":"Store this key securely. This hosted demo keeps key validation in deployment memory."},201)
        except Exception as e:
            json_response(self,{"error":"key_service_unavailable","message":str(e)},503)
