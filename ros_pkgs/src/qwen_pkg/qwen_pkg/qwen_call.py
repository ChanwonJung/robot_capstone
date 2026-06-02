"""
qwen_call.py

Input/output schema and inference harness for Qwen VLM.
No ROS dependencies, no hardcoded deployment details.
"""

import json
from typing import Annotated, Literal, Optional, Union

from openai import OpenAI
from pydantic import BaseModel, Field


# ── Output types ──────────────────────────────────────────────────────────────

class ContainerDest(BaseModel):
    type: Literal["container"]
    reference_id: int


class SurfaceDest(BaseModel):
    type: Literal["surface"]
    reference_id: int
    region: Optional[Literal["left_edge", "right_edge", "center", "far_end", "near_end"]] = None


class RelationDest(BaseModel):
    type: Literal["relation"]
    reference_id: int
    relation: Literal["left_of", "right_of", "in_front_of", "behind", "on_top_of", "near"]


DestinationSpec = Annotated[
    Union[ContainerDest, SurfaceDest, RelationDest],
    Field(discriminator="type"),
]


class GroundingResult(BaseModel):
    target_id: int
    target_label: str
    destination: DestinationSpec
    confidence: float


# ── Guided-decoding schema ────────────────────────────────────────────────────
# Flat structure avoids oneOf/anyOf complexity with the outlines backend.
# dest_region and dest_relation are optional; only one will be used depending
# on dest_type.

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "label": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["TARGET", "DESTINATION", "OBSTACLE"],
                    },
                },
                "required": ["idx", "label", "classification"],
            },
        },
        "target": {
            "type": "object",
            "properties": {
                "idx": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": ["idx", "label"],
        },
        "destination": {
            "type": "object",
            "properties": {
                "idx": {"type": "integer"},
                "label": {"type": "string"},
                "dest_type": {
                    "type": "string",
                    "enum": ["container", "surface", "relation"],
                },
                "dest_region": {
                    "type": "string",
                    "enum": ["left_edge", "right_edge", "center", "far_end", "near_end"],
                },
                "dest_relation": {
                    "type": "string",
                    "enum": ["left_of", "right_of", "in_front_of", "behind", "on_top_of", "near"],
                },
            },
            "required": ["idx", "label", "dest_type"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["classifications", "target", "destination", "confidence"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_grounding_result(raw: dict) -> GroundingResult:
    dest_raw = raw["destination"]
    dest_type = dest_raw["dest_type"]
    ref_id = dest_raw["idx"]

    if dest_type == "container":
        dest: DestinationSpec = ContainerDest(type="container", reference_id=ref_id)
    elif dest_type == "surface":
        dest = SurfaceDest(
            type="surface",
            reference_id=ref_id,
            region=dest_raw.get("dest_region"),
        )
    else:
        dest = RelationDest(
            type="relation",
            reference_id=ref_id,
            relation=dest_raw.get("dest_relation", "near"),
        )

    target = raw["target"]
    return GroundingResult(
        target_id=target["idx"],
        target_label=target["label"],
        destination=dest,
        confidence=raw.get("confidence", 1.0),
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_detections(
    detections: list[dict],
    instruction: str,
    endpoint_url: str,
    model: str,
) -> tuple[list[dict], GroundingResult]:
    """Classify GSAM detections and extract a structured grounding result.

    Parameters
    ----------
    detections:
        Each dict must have: idx, label, confidence, bbox_xyxy
    instruction:
        Natural-language robot command.
    endpoint_url:
        Base URL of the vLLM OpenAI-compat endpoint.
    model:
        Model name served by the endpoint.

    Returns
    -------
    (enriched_detections, grounding_result)
        enriched_detections — input dicts each with a "category" key added
        grounding_result    — structured target + destination with spatial relation
    """
    client = OpenAI(base_url=endpoint_url, api_key="EMPTY")

    detection_summary = "\n".join(
        f'  idx={d["idx"]} label="{d["label"]}" confidence={d["confidence"]:.2f}'
        for d in detections
    )

    prompt = (
        "You are a robot perception assistant.\n"
        "Given a list of detected objects and a manipulation instruction, do TWO things:\n\n"
        "1. Classify every object as exactly one of:\n"
        "     TARGET      — the object the robot should pick up or act on\n"
        "     DESTINATION — where the robot should place / move the target\n"
        "     OBSTACLE    — everything else\n\n"
        "2. Identify the target and its destination with a spatial relation:\n"
        "   - dest_type 'container': place inside (e.g. a box, bowl, bin)\n"
        "   - dest_type 'surface': place on a surface with optional region\n"
        "     (left_edge, right_edge, center, far_end, near_end)\n"
        "   - dest_type 'relation': place relative to another object\n"
        "     (left_of, right_of, in_front_of, behind, on_top_of, near)\n\n"
        f"Instruction: {instruction}\n\n"
        "Detected objects:\n"
        f"{detection_summary}\n\n"
        "Return a JSON object. Every idx must appear in classifications exactly once. "
        "dest_reference_id must be the idx of the DESTINATION object."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Decompose manipulation commands into structured JSON. "
                    "Respond only with valid JSON matching the provided schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "grounding_result", "schema": _SCHEMA},
        },
        temperature=0.0,
        max_tokens=1024,
    )

    if not response.choices:
        raise RuntimeError(f"vLLM returned no choices: {response}")

    content = response.choices[0].message.content
    if content is None:
        # thinking mode fired despite enable_thinking=False — content lives in
        # reasoning_content, actual answer was never emitted
        raise RuntimeError(
            "Model returned null content (thinking mode active?). "
            f"reasoning_content={response.choices[0].message.reasoning_content!r:.200}"
        )

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"VLM output is not valid JSON: {exc}\nRaw: {content!r:.500}") from exc

    import sys
    #print(f"[DEBUG] raw VLM output: {json.dumps(raw, indent=2)}", file=sys.stderr)

    category_map = {c["idx"]: c.get("classification", "OBSTACLE") for c in raw["classifications"]}
    enriched = [{**det, "category": category_map.get(det["idx"], "OBSTACLE")} for det in detections]

    grounding = _build_grounding_result(raw)

    return enriched, grounding
