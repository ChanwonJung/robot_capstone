"""
qwen_vllm_api.py

FastAPI wrapper around a remote vLLM OpenAI-compatible endpoint.
Accepts image + text queries, enforces structured JSON output via
vLLM's guided_json, and returns parsed results.

Fill in:
  MODEL_NAME      — the vLLM model identifier (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
  VLLM_BASE_URL   — http://<remote-host>:<port>/v1
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — fill these in
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME: str = "PLACEHOLDER_MODEL_NAME"   # e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
VLLM_BASE_URL: str = "http://PLACEHOLDER_HOST:8000/v1"
VLLM_API_KEY: str = "EMPTY"                  # vLLM default; set if you added auth
REQUEST_TIMEOUT: float = 60.0               # seconds

# ──────────────────────────────────────────────────────────────────────────────
# Output schema — replace with your actual target structure
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_JSON_SCHEMA: dict = {
    # TODO: define your real schema here.
    # Example placeholder:
    "type": "object",
    "properties": {
        "result": {"type": "string"},
        "confidence": {"type": "number"},
        "metadata": {"type": "object"}
    },
    "required": ["result"]
}

SYSTEM_PROMPT: str = (
    "You are a visual analysis assistant. "
    "Respond ONLY with a valid JSON object that strictly conforms to the schema provided. "
    "Do not include any explanation, markdown fences, or extra text."
)

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen vLLM Vision API",
    description="Proxy for image+text → structured JSON via vLLM guided_json.",
    version="0.1.0",
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    parsed: dict
    raw_text: str          # raw model output before JSON parse (useful for debugging)
    model: str
    usage: dict | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _encode_image(data: bytes, media_type: str) -> str:
    """Return a data-URI string suitable for the vLLM image_url content block."""
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def _build_messages(image_data_uri: str, text_prompt: str) -> list[dict]:
    """Construct the OpenAI-style chat messages list with an image + text turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
                {
                    "type": "text",
                    "text": text_prompt,
                },
            ],
        },
    ]


async def _call_vllm(messages: list[dict]) -> dict:
    """
    POST to vLLM's /v1/chat/completions with guided_json extra_body.
    Returns the full API response dict.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,      # deterministic for structured output
        "max_tokens": 512,
        # vLLM guided decoding — forces output to conform to OUTPUT_JSON_SCHEMA
        "extra_body": {
            "guided_json": OUTPUT_JSON_SCHEMA,
            "guided_backend": "outlines",  # or "lm-format-enforcer"
        },
    }

    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        log.error("vLLM returned %d: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"vLLM upstream error {resp.status_code}: {resp.text[:400]}",
        )

    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(
    image: Annotated[UploadFile, File(description="Image file (JPEG / PNG)")],
    prompt: Annotated[str, Form(description="Text instruction / query")],
):
    """
    Send an image + text prompt to the remote Qwen model.
    Returns a structured JSON object defined by OUTPUT_JSON_SCHEMA.
    """
    # --- read and encode image ---
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    media_type = image.content_type or "image/jpeg"
    if media_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {media_type}")

    data_uri = _encode_image(raw_bytes, media_type)
    log.info("Query received — prompt: %.80s | image: %s (%d bytes)",
             prompt, image.filename, len(raw_bytes))

    # --- call vLLM ---
    vllm_resp = await _call_vllm(_build_messages(data_uri, prompt))

    # --- extract and parse model output ---
    raw_text: str = vllm_resp["choices"][0]["message"]["content"]
    usage: dict | None = vllm_resp.get("usage")

    try:
        # Strip markdown fences if the model ignored the system prompt
        clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        log.error("JSON parse failed: %s\nRaw output: %s", exc, raw_text)
        raise HTTPException(
            status_code=500,
            detail=f"Model output was not valid JSON: {exc}. Raw: {raw_text[:300]}",
        )

    return QueryResponse(
        parsed=parsed,
        raw_text=raw_text,
        model=vllm_resp.get("model", MODEL_NAME),
        usage=usage,
    )


@app.get("/health")
async def health():
    """Lightweight health check — does NOT ping vLLM."""
    return {"status": "ok", "model": MODEL_NAME, "vllm_base": VLLM_BASE_URL}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qwen_vllm_api:app", host="0.0.0.0", port=8080, reload=False)
