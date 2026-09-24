"""
Veyra AI — Cloudflare Python Worker
Flask API
"""

import json
from pyodide.ffi import run_sync
import aiohttp

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from workers import wsgi


APP_NAME = "Veyra AI"
TOKENHARBOR_URL = "https://tokenharbor.ai/v1/chat/completions"

# Set this in Cloudflare as a Secret named API_KEY.
API_KEY = None

MODELS = {
    "deepseek-v4-flash": "deepseek-v4-flash:free",
    "deepseek-v4-1-flash": "deepseek-v4.1-flash:free",
    "qwen3-8-flash": "qwen3.8-flash:free",
    "mimo-v2-6-flash": "mimo-v2.6-flash:free",
    "mimo-v2-5": "mimo-v2.5:free",
}

DEFAULT_MODEL = "deepseek-v4-flash"
TIMEOUT = 60
RETRIES = 2


app = Flask(APP_NAME.lower().replace(" ", "-"))
CORS(app)


def get_api_key():
    """
    Get API_KEY from Cloudflare Worker environment.
    Flask WSGI exposes Worker bindings through request.environ.
    """
    try:
        env = request.environ.get("workers.env")

        if env is not None:
            key = getattr(env, "API_KEY", None)

            if key:
                return str(key)

    except Exception:
        pass

    return None


async def upstream_request(
    model_id,
    messages,
    api_key,
    temperature=None,
    max_tokens=None,
    stream=False,
):
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": stream,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    last_error = None

    for attempt in range(RETRIES):
        try:
            async with aiohttp.ClientSession(
                timeout=timeout
            ) as session:

                async with session.post(
                    TOKENHARBOR_URL,
                    headers=headers,
                    json=payload,
                ) as response:

                    status = response.status
                    body = await response.read()

                    if (
                        status in (429, 500, 502, 503, 504)
                        and attempt < RETRIES - 1
                    ):
                        last_error = RuntimeError(
                            f"upstream {status}"
                        )
                        continue

                    return status, body

        except Exception as exc:
            last_error = exc

    raise last_error or RuntimeError("upstream failed")


def call_upstream(
    model_id,
    messages,
    temperature=None,
    max_tokens=None,
    stream=False,
):
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "API_KEY secret is not configured in Cloudflare."
        )

    return run_sync(
        upstream_request(
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
    )


def error(message, code, status):
    return jsonify(
        {
            "error": {
                "message": message,
                "code": code,
            }
        }
    ), status


@app.get("/")
def index():
    return jsonify(
        {
            "name": APP_NAME,
            "version": "1.0",
            "status": "online",
            "docs": {
                "models": "GET /v1/models",
                "chat": (
                    "POST /v1/chat/<model> "
                    'body: {"prompt": "..."}'
                ),
                "chat_universal": (
                    "POST /v1/chat "
                    'body: {"model": "...", "prompt": "..."}'
                ),
            },
            "models": list(MODELS.keys()),
        }
    )


@app.get("/v1/models")
def models():
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": slug,
                    "upstream": upstream,
                    "endpoint": f"/v1/chat/{slug}",
                }
                for slug, upstream in MODELS.items()
            ],
        }
    )


@app.post("/v1/chat")
def chat_universal():
    body = request.get_json(silent=True) or {}

    slug = (
        body.get("model")
        or DEFAULT_MODEL
    ).strip()

    return _chat(slug, body)


@app.post("/v1/chat/<slug>")
def chat_by_model(slug):
    body = request.get_json(silent=True) or {}

    return _chat(slug, body)


def _chat(slug, body):
    model_id = MODELS.get(slug)

    if not model_id:
        return error(
            f"Unknown model '{slug}'. "
            f"Valid: {', '.join(MODELS)}",
            "unknown_model",
            404,
        )

    # prompt > input > messages
    prompt = body.get("prompt") or body.get("input")
    messages = body.get("messages") or []

    if not messages:

        if not prompt or not str(prompt).strip():
            return error(
                'Provide "prompt" (string) '
                'or "messages" list.',
                "missing_prompt",
                400,
            )

        messages = [
            {
                "role": "user",
                "content": str(prompt),
            }
        ]

    try:
        status, raw_body = call_upstream(
            model_id=model_id,
            messages=messages,
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            stream=bool(body.get("stream")),
        )

    except Exception as exc:
        return error(
            f"Upstream failed: {exc}",
            "upstream_error",
            502,
        )

    # Streaming/SSE passthrough
    if body.get("stream") and 200 <= status < 300:
        return Response(
            raw_body,
            status=status,
            content_type="text/event-stream",
        )

    try:
        data = json.loads(
            raw_body.decode("utf-8")
        )

    except Exception:
        return error(
            f"Upstream returned non-JSON "
            f"(HTTP {status}).",
            "upstream_error",
            502,
        )

    if not 200 <= status < 300:
        return jsonify(data), status

    choices = data.get("choices") or []

    content = ""

    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content", "")

    return jsonify(
        {
            "model": slug,
            "upstream": data.get(
                "model",
                model_id,
            ),
            "content": content,
            "choices": data.get("choices"),
            "usage": data.get("usage"),
        }
    )


# Cloudflare Python Worker entrypoint.
Default = wsgi.entrypoint(app)
