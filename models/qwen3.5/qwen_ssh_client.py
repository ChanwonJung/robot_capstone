"""
qwen_ssh_client.py

Client for the SSH-tunneled Qwen vLLM endpoint.

Typical SSH tunnel setup (run before using this):
    ssh -L 8000:localhost:8000 user@remote-cluster -N

Then call:
    python qwen_ssh_client.py --image path/to/image.jpg --text "What objects do you see?"

Or import and use programmatically:
    from qwen_ssh_client import ask_qwen
    response = ask_qwen("Which object should I pick up?", image_path="scene.jpg")
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Port must match your SSH tunnel: ssh -L <LOCAL_PORT>:localhost:<REMOTE_PORT> ...
LOCAL_PORT: int = 8000
BASE_URL: str = f"http://localhost:{LOCAL_PORT}/v1"

# vLLM default key; set to match your server if you added auth
API_KEY: str = "EMPTY"

# Must match the model name your vLLM server was started with
MODEL_NAME: str = "Qwen/Qwen2.5-VL-7B-Instruct"

REQUEST_TIMEOUT: float = 120.0


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def ask_qwen(
    text: str,
    *,
    image_path: str | Path | None = None,
    image_bytes: bytes | None = None,
    image_media_type: str = "image/jpeg",
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Send a text prompt (and optional image) to the SSH-tunneled Qwen endpoint.

    Provide exactly one of image_path or image_bytes (or neither for text-only).

    Returns the model's reply as a plain string.
    """
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=REQUEST_TIMEOUT,
    )

    # Build the user message content
    content: list[dict] = []

    # Attach image if provided
    if image_path is not None:
        image_bytes = Path(image_path).read_bytes()
        suffix = Path(image_path).suffix.lower()
        image_media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{image_media_type};base64,{b64}"
        content.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })

    content.append({"type": "text", "text": text})

    # Build messages list
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the SSH-tunneled Qwen VLM with text and an optional image."
    )
    parser.add_argument("--text", required=True, help="Text prompt to send.")
    parser.add_argument("--image", default=None, help="Path to image file (JPEG/PNG/WebP).")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--port", type=int, default=LOCAL_PORT,
        help=f"Local SSH tunnel port (default: {LOCAL_PORT}).",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model name served by vLLM (default: {MODEL_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Allow CLI overrides without editing the file
    global BASE_URL, MODEL_NAME
    BASE_URL = f"http://localhost:{args.port}/v1"
    MODEL_NAME = args.model

    print(f"Endpoint : {BASE_URL}")
    print(f"Model    : {MODEL_NAME}")
    print(f"Image    : {args.image or '(none)'}")
    print(f"Prompt   : {args.text}")
    print("-" * 60)

    try:
        reply = ask_qwen(
            args.text,
            image_path=args.image,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(reply)


if __name__ == "__main__":
    main()
