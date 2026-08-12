"""Qwen VLM client for the A100 vLLM endpoint.

ROS-free.  Holds the transport, the prompt harness, and reply validation.
No ROS types, no topic names, no node state — so it can be exercised offline
via ``qwen_cli.py`` without a running graph.

Transport is HTTP (vLLM's OpenAI-compatible API), NOT the ZMQ/msgpack channel
that graspgen and swindrnet use.  Both live on the cluster but they are
different services on different ports; do not unify the two clients.

Endpoint reachability is via the SSH tunnel launch_env.bash opens:
localhost:8000 -> tta@123.37.28.208:8000 (the A100, loopback-only), the same
host that serves GraspGen on 5556 and SwinDRNet on 5557.  If the tunnel is down
every call fails with a connection error rather than a timeout — a sub-second
failure means the tunnel, a slow failure means the model.
"""
from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from .qwen_schema import (
    clamp_boxes,
    DEST_REGIONS,
    DEST_RELATIONS,
    DEST_TYPES,
    Detection,
    GroundingResult,
    order_detections,
    scale_boxes,
    target_candidates,
    VLM_SCHEMA,
)

_SYSTEM_PROMPT = (
    "You are a robot perception system. You look at a workspace image and a "
    "manipulation instruction, and you return structured JSON describing what "
    "the robot should pick up and where it should put it. "
    "Respond only with JSON matching the provided schema."
)


# Dual-view role split.  The wrist camera cannot see the destination at all in
# this workspace (its near limit on the table is x ~ 0.58 m in panda_link0,
# while the basket sits at x = 0.48), and the overhead camera resolves objects
# at 1/4 the wrist camera's pixel density.  So each view is asked for exactly
# the thing it can actually see, and qwen_bridge merges the two replies.
#
# These are appended to the shared prompt rather than replacing it, so the
# object list, the box format and the ambiguity protocol stay identical across
# both calls.
VIEW_HINT_WRIST = (
    "\n\nVIEW: close-up wrist camera, looking down at the objects.\n"
    "Emit exactly one TARGET and classify everything else as OBSTACLE.\n"
    "Do NOT emit a DESTINATION from this view and do NOT fill the destination\n"
    "field — a separate overhead view handles the destination, and the\n"
    "destination is usually outside this camera's field of view entirely."
)

VIEW_HINT_OVERHEAD = (
    "\n\nVIEW: overhead camera, looking straight down at the whole workspace.\n"
    "Emit ONLY the DESTINATION — the container or surface the object must end\n"
    "up at — and fill the destination field describing it. Classify every other\n"
    "object as OBSTACLE. Do NOT emit a TARGET from this view; a separate\n"
    "close-up view handles the target.\n"
    "If the instruction does not say where the object should go, emit no\n"
    "DESTINATION and omit the destination field."
)


def _build_user_prompt(instruction: str, width: int, height: int) -> str:
    return (
        f"Workspace image is {width}x{height} pixels.\n"
        f"Instruction: {instruction}\n\n"
        "Do all of the following:\n\n"
        "1. Detect every distinct manipulable object in the image. For each, give\n"
        "   a short noun label, a tight bounding box, and a confidence.\n"
        "   IMPORTANT: if the instruction names a surface or container to place\n"
        "   on or in — a table, tray, shelf, counter — include THAT as an object\n"
        "   too, with its own box, even though it is scenery rather than\n"
        "   something to pick up. Step 3 requires its label to appear in this\n"
        "   list, and without a box there is nothing to segment and the robot\n"
        "   has nowhere to place.\n\n"
        "2. Classify each object as exactly one of:\n"
        "     TARGET      - the single object the robot must pick up\n"
        "     DESTINATION - the single object/surface it must be placed at\n"
        "     OBSTACLE    - everything else\n"
        "   Emit exactly one TARGET, unless the instruction is ambiguous — see\n"
        "   step 4. Emit a DESTINATION only if the instruction says where the\n"
        "   object should go.\n\n"
        "3. Describe the destination spatially:\n"
        f"     type: one of {list(DEST_TYPES)}\n"
        "       'container' - place inside it (box, bowl, bin)\n"
        "       'surface'   - place on top of it (table, tray, shelf)\n"
        "       'relation'  - place positioned relative to it\n"
        f"     region (surface only): one of {list(DEST_REGIONS)}\n"
        f"     relation (relation only): one of {list(DEST_RELATIONS)}\n"
        "     reference_label: the label string of the destination object,\n"
        "       exactly as it appears in your objects list. If you named a\n"
        "       destination here, step 1's list MUST contain it — a destination\n"
        "       with no matching detection is the one failure that silently\n"
        "       disables placing.\n\n"
        "If the instruction does NOT say where to put the object (e.g. 'pick up\n"
        "the book'), OMIT the destination field entirely and classify nothing as\n"
        "DESTINATION. Do not invent a destination.\n\n"
        "4. Ambiguity. If SEVERAL objects match the instruction equally well\n"
        "   (e.g. 'pick up the cup' with three cups on the table):\n"
        "     - mark EVERY plausible one as TARGET, not just your favourite\n"
        "     - set needs_clarification = true\n"
        "     - write clarification_question: one short question that names the\n"
        "       features that tell them apart (colour, position, size), e.g.\n"
        "       \"I see three cups. Did you mean the red one on the left, the\n"
        "       white one in the middle, or the glass one on the right?\"\n"
        "   If ONE object is clearly the best match, set needs_clarification =\n"
        "   false and mark only that object TARGET. Do not ask when you know."
    )


def encode_image(image_bgr: np.ndarray, quality: int = 90) -> str:
    """BGR ndarray -> base64 JPEG data URI payload."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed on the source frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def ground(
    image_bgr: np.ndarray,
    instruction: str,
    *,
    endpoint_url: str,
    model: str,
    bbox_convention: str = "absolute",
    timeout_sec: float = 120.0,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    view_hint: str = "",
) -> tuple[list[Detection], GroundingResult, dict[str, Any]]:
    """Run one grounding pass.

    view_hint is appended verbatim to the user prompt.  Pass VIEW_HINT_WRIST or
    VIEW_HINT_OVERHEAD to restrict a call to one role in the dual-view pipeline;
    leave it empty for the single-view behaviour.

    Returns
    -------
    (detections, grounding, meta)
        detections — ordered TARGET, DESTINATION, then obstacles.  Boxes are in
                     ORIGINAL image pixel space and clipped to bounds.
        grounding  — validated destination spec for bt_pkg.
        meta       — timing/diagnostic fields for logging.

    Raises on transport failure, malformed JSON, or a reply that does not
    validate.  The caller decides whether to retry; failing loudly beats
    publishing a half-parsed scene.

    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("empty source frame")

    height, width = image_bgr.shape[:2]
    client = OpenAI(base_url=endpoint_url, api_key="EMPTY", timeout=timeout_sec)
    b64 = encode_image(image_bgr)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text",
                     "text": _build_user_prompt(instruction, width, height) + view_hint},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "grounding_result", "schema": VLM_SCHEMA},
        },
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise RuntimeError(f"vLLM returned no choices: {response}")

    content = response.choices[0].message.content
    if content is None:
        # Thinking mode swallowed the answer — the real text went to
        # reasoning_content and the schema-constrained reply was never emitted.
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        raise RuntimeError(
            "Model returned null content (thinking mode active?). "
            f"reasoning_content={str(reasoning)[:200]!r}"
        )

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"VLM output is not valid JSON: {exc}\nRaw: {content[:500]!r}") from exc

    objects = [Detection(**o) for o in raw["objects"]]

    # VLM_SCHEMA constrains bbox_xyxy to exactly 4 numbers, but a server that
    # ignores response_format would slip through — and Detection no longer
    # carries a length constraint (pydantic v1 compatibility).
    for det in objects:
        if len(det.bbox_xyxy) != 4:
            raise RuntimeError(
                f"detection {det.label!r} has {len(det.bbox_xyxy)} bbox values, "
                "expected 4 — is the endpoint honouring the JSON schema?")

    objects = scale_boxes(objects, bbox_convention, width, height)
    n_scaled = len(objects)
    objects = clamp_boxes(objects, width, height)
    # A wrong bbox_convention does not error anywhere — it pushes coordinates
    # outside the frame, clamp_boxes collapses them to zero area and drops them,
    # and the caller sees an empty detection list with no explanation. Count the
    # casualties so the failure can name itself.
    n_dropped = n_scaled - len(objects)

    # Ambiguity must be read BEFORE order_detections, which keeps the first
    # TARGET and demotes the rest to OBSTACLE. That normalisation is required
    # for the mask contract but it erases the evidence.
    candidates = target_candidates(objects)
    needs_clarification = bool(raw.get("needs_clarification", False))
    # A model that marked several TARGETs but forgot the flag is still
    # ambiguous — trust the detections over the boolean.
    ambiguous = needs_clarification or len(candidates) > 1

    question = raw.get("clarification_question", "").strip()
    if ambiguous and not question:
        names = ", ".join(f"{d.label!r}" for d in candidates) or "several objects"
        question = (f"Multiple objects match that instruction ({names}). "
                    "Which one did you mean?")

    objects = order_detections(objects)

    # A pick-only instruction legitimately has no destination. Treat an omitted
    # or empty destination as "none" rather than substituting a guess.
    dest_raw = raw.get("destination")
    grounding = GroundingResult(
        target_label=raw["target_label"],
        destination=dest_raw if dest_raw else None,
        confidence=raw.get("confidence", 1.0),
    )

    # No destination warning here. parse() sees ONE view, and under dual-view
    # the destination belongs to the other one: the wrist call legitimately has
    # no "table" among its detections, because to the wrist camera the table is
    # background. Warning from here fired on every overhead-only destination —
    # exactly the case dual-view exists for — and said "place phase will have no
    # pose" while the place was in fact succeeding. A warning that cries wolf on
    # the normal path is worse than none, because it hides the real failure.
    #
    # `has_destination` below reports the fact; the caller, which knows how many
    # views it ran, decides whether that is a problem. qwen_bridge checks the
    # merged pair (`dual and not top_has_dest`), qwen_cli checks its single view.
    meta_warn = ""

    if n_dropped:
        drop_note = (
            f"{n_dropped} of {n_scaled} boxes collapsed to zero area and were "
            f"dropped under bbox_convention={bbox_convention!r} — the convention "
            "is probably wrong. Check with qwen_cli --annotate."
        )
        meta_warn = f"{meta_warn} | {drop_note}" if meta_warn else drop_note

    meta = {
        "source_size": [width, height],
        "n_objects": len(objects),
        "n_raw": n_scaled,
        "n_dropped_degenerate": n_dropped,
        "has_target": any(d.category == "TARGET" for d in objects),
        # Both must hold for a place to be possible: a spec to place BY, and a
        # segmented object to place AT.
        "has_destination": (grounding.destination is not None
                            and any(d.category == "DESTINATION" for d in objects)),
        "pick_only": grounding.destination is None,
        # Ambiguity — the caller decides whether to ask; ground() never guesses.
        "ambiguous": ambiguous,
        "question": question if ambiguous else "",
        "target_candidates": [
            {"label": d.label,
             "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
             "confidence": round(d.confidence, 3)}
            for d in candidates
        ],
        "usage": getattr(response, "usage", None),
        "warning": meta_warn,
    }
    return objects, grounding, meta
