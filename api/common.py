from __future__ import annotations
import hashlib,json,secrets,threading
from api.runtime import get_model,build_prompt,rate_limiter

_api_key_hashes=set()
_api_key_lock=threading.Lock()
_API_KEY_LIMIT=10000
_CLIENT_SALT=secrets.token_bytes(32)

def json_response(h,payload,status=200,extra=None):
    body=json.dumps(payload,ensure_ascii=False).encode()
    h.send_response(status)
    h.send_header("Content-Type","application/json; charset=utf-8")
    h.send_header("Cache-Control","no-store")
    h.send_header("Access-Control-Allow-Origin","*")
    h.send_header("Access-Control-Allow-Headers","Content-Type,Authorization,X-Demo-AI-Key,X-Demo-Session")
    h.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
    if extra:
        for k,v in extra.items(): h.send_header(k,v)
    h.end_headers(); h.wfile.write(body)

def parse_json(h):
    try:
        n=int(h.headers.get("Content-Length","0"))
    except ValueError:
        raise ValueError("Invalid Content-Length.")
    if n<=0 or n>256000: raise ValueError("Request body is missing or too large.")
    return json.loads(h.rfile.read(n).decode())

def _hash_client(value):
    return hashlib.sha256(_CLIENT_SALT + value.encode("utf-8")).hexdigest()

def client_key(h):
    # Use the address nearest the trusted reverse proxy when available.
    # Only a process-scoped digest is retained; the raw address is discarded.
    forwarded=h.headers.get("X-Forwarded-For","")
    if forwarded:
        candidate=forwarded.split(",")[-1].strip()
        if candidate: return "ip:"+_hash_client(candidate)
    real_ip=h.headers.get("X-Real-IP","").strip()
    if real_ip: return "ip:"+_hash_client(real_ip)
    session=h.headers.get("X-Demo-Session","").strip()[:128]
    if session: return "session:"+_hash_client(session)
    return "anon"

def _digest(value): return hashlib.sha256(value.encode("utf-8")).digest()

def authenticate_api_key(h):
    raw=h.headers.get("X-Demo-AI-Key") or h.headers.get("Authorization","")
    if raw.lower().startswith("bearer "): raw=raw[7:].strip()
    if not raw.startswith("demo_sk_"): return False
    digest=_digest(raw)
    with _api_key_lock:
        return any(secrets.compare_digest(digest, item) for item in _api_key_hashes)

def generate_api_key():
    key="demo_sk_"+secrets.token_urlsafe(32)
    digest=_digest(key)
    with _api_key_lock:
        if len(_api_key_hashes)>=_API_KEY_LIMIT:
            _api_key_hashes.clear()
        _api_key_hashes.add(digest)
    return key

def handle_chat(h,require_key=False):
    if h.command=="OPTIONS":
        json_response(h,{"ok":True}); return
    try:
        if require_key:
            if not authenticate_api_key(h):
                json_response(h,{"error":"invalid_api_key"},401); return
        elif not rate_limiter.allow(client_key(h),12):
            json_response(h,{"error":"rate_limited","retry_after_seconds":60},429,{"Retry-After":"60"}); return
        d=parse_json(h); msgs=d.get("messages")
        if not isinstance(msgs,list) or not msgs: raise ValueError("messages must be a non-empty array.")
        clean=[]
        for m in msgs[-12:]:
            if isinstance(m,dict) and m.get("role") in {"user","assistant"} and str(m.get("content","")).strip():
                clean.append({"role":m["role"],"content":str(m["content"]).strip()[:8000]})
        if not clean: raise ValueError("No usable messages were provided.")
        answer=get_model().generate(build_prompt(clean),max_new=d.get("max_new",96),temperature=d.get("temperature",.7),top_k=d.get("top_k",40),seed=d.get("seed"))
        json_response(h,{"id":"demo-chat","object":"chat.completion","model":"demo-ai-100m","choices":[{"index":0,"message":{"role":"assistant","content":answer},"finish_reason":"stop"}]})
    except ValueError as e:
        json_response(h,{"error":"bad_request","message":str(e)},400)
    except Exception as e:
        json_response(h,{"error":"inference_error","message":str(e)},500)
