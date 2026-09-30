from __future__ import annotations

import hashlib
import json
import secrets
import threading

from api.runtime import build_prompt, get_model, rate_limiter, _inference_lock


_api_key_hashes = set()
_api_key_lock = threading.Lock()
_API_KEY_LIMIT = 10000
_CLIENT_SALT = secrets.token_bytes(32)


def json_response(h, payload, status=200, extra=None):
    body = json.dumps(payload, ensure_ascii=False).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Cache-Control", "no-store")
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header(
        "Access-Control-Allow-Headers",
        "Content-Type,Authorization,X-Demo-AI-Key,X-Demo-Session",
    )
    h.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    if extra:
        for key, value in extra.items():
            h.send_header(key, value)
    h.end_headers()
    h.wfile.write(body)


def parse_json(h):
    try:
        size = int(h.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Invalid Content-Length.") from exc
    if size <= 0 or size > 256000:
        raise ValueError("Request body is missing or too large.")
    return json.loads(h.rfile.read(size).decode())


def _hash_client(value):
    return hashlib.sha256(_CLIENT_SALT + value.encode("utf-8")).hexdigest()


def client_key(h):
    forwarded = h.headers.get("X-Forwarded-For", "")
    if forwarded:
        candidate = forwarded.split(",")[-1].strip()
        if candidate:
            return "ip:" + _hash_client(candidate)
    real_ip = h.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return "ip:" + _hash_client(real_ip)
    session = h.headers.get("X-Demo-Session", "").strip()[:128]
    if session:
        return "session:" + _hash_client(session)
    return "anon"


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).digest()


def authenticate_api_key(h):
    raw = h.headers.get("X-Demo-AI-Key") or h.headers.get("Authorization", "")
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    if not raw.startswith("demo_sk_"):
        return False
    digest = _digest(raw)
    with _api_key_lock:
        return any(secrets.compare_digest(digest, item) for item in _api_key_hashes)


def generate_api_key():
    key = "demo_sk_" + secrets.token_urlsafe(32)
    digest = _digest(key)
    with _api_key_lock:
        if len(_api_key_hashes) >= _API_KEY_LIMIT:
            _api_key_hashes.clear()
        _api_key_hashes.add(digest)
    return key


def handle_chat(h, require_key=False):
    if h.command == "OPTIONS":
        json_response(h, {"ok": True})
        return
    try:
        if require_key:
            if not authenticate_api_key(h):
                json_response(h, {"error": "invalid_api_key"}, 401)
                return
        elif not rate_limiter.allow(client_key(h), 12):
            json_response(
                h,
                {"error": "rate_limited", "retry_after_seconds": 60},
                429,
                {"Retry-After": "60"},
            )
            return

        data = parse_json(h)
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array.")

        clean = []
        for message in messages[-12:]:
            if (
                isinstance(message, dict)
                and message.get("role") in {"user", "assistant"}
                and str(message.get("content", "")).strip()
            ):
                clean.append(
                    {
                        "role": message["role"],
                        "content": str(message.get("content", "")).strip()[:8000],
                    }
                )
        if not clean:
            raise ValueError("No usable messages were provided.")

        model = get_model()
        requested_max = int(data.get("max_new", 12))
        max_new = min(max(requested_max, 1), 24 if require_key else 12)
        requested_top_k = int(data.get("top_k", 16))
        top_k = min(max(requested_top_k, 1), 32)

        # Demo AI is CPU-hosted on a small instance. Serialize generation so
        # simultaneous requests cannot multiply CPU/RAM pressure.
        with _inference_lock:
            answer = model.generate(
                build_prompt(clean),
                max_new=max_new,
                temperature=data.get("temperature", 0.7),
                top_k=top_k,
                seed=data.get("seed"),
                max_prompt_tokens=256,
            )
        model = get_model()
        json_response(
            h,
            {
                "id": "demo-chat",
                "object": "chat.completion",
                "model": model.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "architecture": "sparse_moe",
                    "experts": model.n_experts,
                    "top_k": model.top_k,
                },
            },
        )
    except ValueError as exc:
        json_response(h, {"error": "bad_request", "message": str(exc)}, 400)
    except Exception as exc:
        json_response(h, {"error": "inference_error", "message": str(exc)}, 500)
