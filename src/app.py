import json
from flask import Flask, request, Response
from workers import wsgi
from js import fetch
from pyodide.ffi import run_sync

app = Flask(__name__)

UPSTREAM_URL = "https://tokenharbor.ai/v1/chat/completions"

MODELS = {
    "deepseek-v4-flash": "deepseek-v4-flash:free",
    "deepseek-v4-1-flash": "deepseek-v4.1-flash:free",
    "qwen3-8-flash": "qwen3.8-flash:free",
    "mimo-v2-6-flash": "mimo-v2.6-flash:free",
    "mimo-v2-5": "mimo-v2.5:free",
}


# -------------------------
# CORS
# -------------------------

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def home():
    return {
        "status": "online",
        "name": "Ai-Models API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat",
            "model_chat": "/v1/chat/<model>"
        }
    }


# -------------------------
# Models
# -------------------------

@app.route("/v1/models", methods=["GET"])
def models():
    data = []

    for model_id, upstream_model in MODELS.items():
        data.append({
            "id": model_id,
            "object": "model",
            "owned_by": "ai-models",
            "upstream_model": upstream_model
        })

    return {
        "object": "list",
        "data": data
    }


# -------------------------
# Upstream request
# -------------------------

def call_upstream(payload, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    options = {
        "method": "POST",
        "headers": headers,
        "body": json.dumps(payload),
    }

    try:
        js_response = run_sync(fetch(UPSTREAM_URL, options))

        status = int(js_response.status)
        text = run_sync(js_response.text())

        return status, text

    except Exception as e:
        return 502, json.dumps({
            "error": {
                "message": f"Upstream request failed: {str(e)}",
                "type": "upstream_error"
            }
        })


# -------------------------
# Chat API
# -------------------------

@app.route("/v1/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return Response(status=204)

    body = request.get_json(silent=True)

    if not body:
        return {
            "error": {
                "message": "Request body must be valid JSON",
                "type": "invalid_request_error"
            }
        }, 400

    model = body.get("model")

    if not model:
        return {
            "error": {
                "message": "Missing required field: model",
                "type": "invalid_request_error"
            }
        }, 400

    if model not in MODELS:
        return {
            "error": {
                "message": f"Unknown model: {model}",
                "type": "invalid_request_error",
                "available_models": list(MODELS.keys())
            }
        }, 400

    # Convert public model name to upstream model name
    payload = dict(body)
    payload["model"] = MODELS[model]

    # Get Cloudflare Secret
    try:
        env = request.environ["workers.env"]
        api_key = env.UPSTREAM_API_KEY
    except Exception:
        return {
            "error": {
                "message": "UPSTREAM_API_KEY secret is not configured",
                "type": "configuration_error"
            }
        }, 500

    status, upstream_text = call_upstream(payload, api_key)

    # Preserve upstream response
    content_type = "application/json"

    if status >= 400:
        return Response(
            upstream_text,
            status=status,
            content_type=content_type
        )

    return Response(
        upstream_text,
        status=status,
        content_type=content_type
    )


# -------------------------
# Model-specific Chat API
# -------------------------

@app.route("/v1/chat/<model>", methods=["POST", "OPTIONS"])
def model_chat(model):
    if request.method == "OPTIONS":
        return Response(status=204)

    if model not in MODELS:
        return {
            "error": {
                "message": f"Unknown model: {model}",
                "type": "invalid_request_error",
                "available_models": list(MODELS.keys())
            }
        }, 400

    body = request.get_json(silent=True)

    if not body:
        return {
            "error": {
                "message": "Request body must be valid JSON",
                "type": "invalid_request_error"
            }
        }, 400

    payload = dict(body)
    payload["model"] = MODELS[model]

    try:
        env = request.environ["workers.env"]
        api_key = env.UPSTREAM_API_KEY
    except Exception:
        return {
            "error": {
                "message": "UPSTREAM_API_KEY secret is not configured",
                "type": "configuration_error"
            }
        }, 500

    status, upstream_text = call_upstream(payload, api_key)

    return Response(
        upstream_text,
        status=status,
        content_type="application/json"
    )


# -------------------------
# Cloudflare Worker entry
# -------------------------

Default = wsgi.entrypoint(app)
