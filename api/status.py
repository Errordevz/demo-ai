from http.server import BaseHTTPRequestHandler
from api.common import json_response
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        json_response(self,{"ok":True,"service":"Demo AI","model":"demo-ai-100m","parameters":100000000,"context":1024,"cloud_chat":True,"api_key_required_for_web_chat":False,"api_key_required_for_developer_api":True,"checkpoint_status":"development"})
