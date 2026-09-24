"""
Veyra AI — Simple Prompt API
POST prompt, get response. No auth needed.

Run:
    pip install flask flask-cors requests
    python veyra_ai.py

Endpoints:
    GET  /                      -> API info
    GET  /v1/models             -> model list
    POST /v1/chat/<model>       -> {"prompt": "..."}   (per-model src)
    POST /v1/chat               -> {"model": "...", "prompt": "..."}

Example:
    curl -X POST http://localhost:5000/v1/chat/deepseek-v4-flash \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Hello!"}'

Response:
    {"choices": [{"message": {"content": "..."}}], "usage": {...}}
"""

import time
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

APP_NAME = "Veyra AI"
TOKENHARBOR_URL = "https://tokenharbor.ai/v1/chat/completions"
API_KEY = "thk_live_h_cLT18lAnZT2oQHqKVrxQziGV7ZXacSgJCG1u2GodzK5v5uJV7jfWRuQ3WZLR0x"

MODELS = {
    "deepseek-v4-flash":   "deepseek-v4-flash:free",
    "deepseek-v4-1-flash": "deepseek-v4.1-flash:free",
    "qwen3-8-flash":       "qwen3.8-flash:free",
    "mimo-v2-6-flash":     "mimo-v2.6-flash:free",
    "mimo-v2-5":           "mimo-v2.5:free",
}
DEFAULT_MODEL = "deepseek-v4-flash"
TIMEOUT = 60  # seconds
RETRIES = 2   # 1 attempt + 1 retry on transient failures

app = Flask(APP_NAME.lower().replace(" ", "-"))
CORS(app)


def call_upstream(model_id, messages, temperature=None, max_tokens=None, stream=False):
    """POST to TokenHarbor, retry once on transient 429/5xx."""
    payload = {"model": model_id, "messages": messages, "stream": stream}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    last_err = None
    for attempt in range(RETRIES):
        if attempt > 0:
            time.sleep(0.8)
        try:
            r = requests.post(
                TOKENHARBOR_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT,
                stream=stream,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                last_err = RuntimeError(f"upstream {r.status_code}")
                continue
            return r
        except requests.RequestException as e:
            last_err = e

    raise last_err if last_err else RuntimeError("upstream failed")


def error(message, code, status):
    return jsonify({"error": {"message": message, "code": code}}), status


@app.get("/")
def index():
    return jsonify({
        "name": APP_NAME,
        "version": "1.0",
        "docs": {
            "models": "GET /v1/models",
            "chat": "POST /v1/chat/<model>  body: {\"prompt\": \"...\"}",
            "chat_universal": "POST /v1/chat  body: {\"model\": \"...\", \"prompt\": \"...\"}",
        },
        "models": list(MODELS.keys()),
    })


@app.get("/v1/models")
def models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": slug, "upstream": upstream, "endpoint": f"/v1/chat/{slug}"}
            for slug, upstream in MODELS.items()
        ],
    })


@app.post("/v1/chat")
def chat_universal():
    body = request.get_json(silent=True) or {}
    slug = (body.get("model") or DEFAULT_MODEL).strip()
    return _chat(slug, body)


@app.post("/v1/chat/<slug>")
def chat_by_model(slug):
    return _chat(slug, request.get_json(silent=True) or {})


def _chat(slug, body):
    model_id = MODELS.get(slug)
    if not model_id:
        return error(
            f"Unknown model '{slug}'. Valid: {', '.join(MODELS)}",
            "unknown_model", 404,
        )

    # prompt > input > messages
    prompt = body.get("prompt") or body.get("input")
    messages = body.get("messages") or []
    if not messages:
        if not prompt or not str(prompt).strip():
            return error('Provide "prompt" (string) or "messages" list.', "missing_prompt", 400)
        messages = [{"role": "user", "content": prompt}]

    try:
        r = call_upstream(
            model_id,
            messages,
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            stream=bool(body.get("stream")),
        )
    except Exception as e:
        return error(f"Upstream failed: {e}", "upstream_error", 502)

    # Streaming passthrough (SSE)
    if body.get("stream") and r.ok:
        return app.response_class(
            r.iter_content(chunk_size=None),
            content_type="text/event-stream",
        )

    try:
        data = r.json()
    except ValueError:
        return error(f"Upstream returned non-JSON (HTTP {r.status_code}).", "upstream_error", 502)

    if not r.ok:
        return jsonify(data), r.status_code

    # Veyra-style response (OpenAI-compatible, thoda clean)
    return jsonify({
        "model": slug,
        "upstream": data.get("model", model_id),
        "content": data["choices"][0]["message"]["content"],
        "choices": data.get("choices"),
        "usage": data.get("usage"),
    })


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 5000))
    print(f"\n  {APP_NAME} running → http://localhost:{port}")
    print(f"  Try: curl -X POST http://localhost:{port}/v1/chat/deepseek-v4-flash "
          f"-H \"Content-Type: application/json\" -d '{{\"prompt\":\"hi\"}}'\n")
    app.run(host="0.0.0.0", port=port, threaded=True)
