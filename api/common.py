from __future__ import annotations
import base64,hashlib,hmac,json,os
from api.runtime import get_model,build_prompt,rate_limiter

def json_response(h,payload,status=200,extra=None):
    body=json.dumps(payload,ensure_ascii=False).encode()
    h.send_response(status); h.send_header("Content-Type","application/json; charset=utf-8"); h.send_header("Cache-Control","no-store")
    h.send_header("Access-Control-Allow-Origin","*"); h.send_header("Access-Control-Allow-Headers","Content-Type,Authorization,X-Demo-AI-Key,X-Demo-Session"); h.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
    if extra:
        for k,v in extra.items(): h.send_header(k,v)
    h.end_headers(); h.wfile.write(body)

def parse_json(h):
    n=int(h.headers.get("Content-Length","0"))
    if n<=0 or n>256000: raise ValueError("Request body is missing or too large.")
    return json.loads(h.rfile.read(n).decode())

def client_key(h): return h.headers.get("X-Demo-Session","anonymous")[:128]

def authenticate_api_key(h):
    raw=h.headers.get("X-Demo-AI-Key") or h.headers.get("Authorization","")
    if raw.lower().startswith("bearer "): raw=raw[7:].strip()
    secret=os.getenv("DEMO_AI_API_SIGNING_SECRET","")
    if not secret or not raw.startswith("demo_sk_") or "." not in raw: return False
    try:
        part=raw.split("demo_sk_",1)[1]; payload,sig=part.split(".",1)
        nonce=base64.urlsafe_b64decode(payload+"="*(-len(payload)%4)); provided=base64.urlsafe_b64decode(sig+"="*(-len(sig)%4))
        return hmac.compare_digest(provided,hmac.new(secret.encode(),nonce,hashlib.sha256).digest())
    except Exception: return False

def generate_api_key():
    secret=os.getenv("DEMO_AI_API_SIGNING_SECRET","")
    if not secret: raise RuntimeError("API signing secret is not configured.")
    nonce=os.urandom(32); payload=base64.urlsafe_b64encode(nonce).decode().rstrip("="); sig=base64.urlsafe_b64encode(hmac.new(secret.encode(),nonce,hashlib.sha256).digest()).decode().rstrip("=")
    return "demo_sk_"+payload+"."+sig

def handle_chat(h,require_key=False):
    if h.command=="OPTIONS": json_response(h,{"ok":True}); return
    try:
        if require_key and not authenticate_api_key(h): json_response(h,{"error":"invalid_api_key"},401); return
        if not require_key and not rate_limiter.allow(client_key(h),12): json_response(h,{"error":"rate_limited","retry_after_seconds":60},429,{"Retry-After":"60"}); return
        d=parse_json(h); msgs=d.get("messages")
        if not isinstance(msgs,list) or not msgs: raise ValueError("messages must be a non-empty array.")
        clean=[]
        for m in msgs[-12:]:
            if isinstance(m,dict) and m.get("role") in {"user","assistant"} and str(m.get("content","")).strip(): clean.append({"role":m["role"],"content":str(m["content"]).strip()[:8000]})
        if not clean: raise ValueError("No usable messages were provided.")
        answer=get_model().generate(build_prompt(clean),max_new=d.get("max_new",96),temperature=d.get("temperature",.7),top_k=d.get("top_k",40),seed=d.get("seed"))
        answer=answer.split("### Assistant\n",1)[-1].split("### User\n",1)[0].strip()
        json_response(h,{"id":"demo-chat","object":"chat.completion","model":"demo-ai-100m","choices":[{"index":0,"message":{"role":"assistant","content":answer},"finish_reason":"stop"}]})
    except ValueError as e: json_response(h,{"error":"bad_request","message":str(e)},400)
    except Exception as e: json_response(h,{"error":"inference_error","message":str(e)},500)
