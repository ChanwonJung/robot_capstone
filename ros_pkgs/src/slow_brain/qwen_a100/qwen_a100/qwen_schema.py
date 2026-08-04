"""Structured-output schema for the Qwen A100 grounding call.

ROS-free.  Single source of truth for the JSON contract between the VLM and
every downstream consumer.  Two distinct things live here:

  * ``VLM_SCHEMA``      — guided-decoding schema handed to vLLM.  Constrains
                          what the model is allowed to emit.
  * pydantic models     — validate + normalise the reply before publishing.

Downstream contracts these feed.  Changing a field name here silently breaks a
consumer, because every consumer parses JSON out of a ``std_msgs/String`` and
falls back to a default rather than erroring:

  /qwen/labeled_detections -> mask_projection_pkg : reads "label", "category"
                           -> sam_a100            : reads "bbox_xyxy", "category"
  /qwen/grounding_result   -> bt_pkg              : reads destination.{type,
                                                    reference_label, relation,
                                                    region}

Two invariants that are load-bearing and easy to break:

1. ``labeled_detections`` array order defines the mask label values.
   ``mask_projection_pkg`` joins by ARRAY POSITION, not by any id field:
   mask pixel value ``v`` -> ``detections[v - 1]``.  We emit TARGET at index 0
   and DESTINATION at index 1, so sam_a100 paints target=1, destination=2.

2. ``destination.reference_label`` is a STRING class name, not an id.  bt_pkg
   matches it against ``/yolo/world_map`` ``class_name`` to re-localise the
   destination at place time.  The legacy qwen_pkg emitted ``reference_id``
   (an int) here, which is why that path never worked.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import pydantic
from pydantic import BaseModel

# This package must import under BOTH pydantic majors: the system interpreter
# ships 1.10 while gsam_venv ships 2.x, and which one wins depends on whether
# launch_env.bash has injected the venv into PYTHONPATH.  Node shebangs are
# rewritten to system python by colcon, so v1 is a real runtime possibility.
# Avoid v2-only APIs (model_copy, model_dump_json, Field(min_length=...) on
# sequences) and go through the shims below.
_PYDANTIC_V2 = pydantic.VERSION.startswith("2")


def to_json(model: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialise a model to JSON on either pydantic major.

    exclude_none matters for the destination: bt_pkg checks
    ``j.contains("destination")`` and then calls .value() on it, so emitting an
    explicit null would throw inside its parser rather than be treated as
    absent.  The key has to be omitted entirely.
    """
    if _PYDANTIC_V2:
        return model.model_dump_json(exclude_none=exclude_none)
    return model.json(exclude_none=exclude_none)


def to_dict(model: BaseModel) -> dict[str, Any]:
    """Dump a model to a plain dict on either pydantic major."""
    return model.model_dump() if _PYDANTIC_V2 else model.dict()

# ── Vocabularies ──────────────────────────────────────────────────────────────
# These MUST stay in sync with bt_pkg/src/destination_calculator.cpp, which
# dispatches on the raw strings.  Unknown values there are silently tolerated
# (they degrade to "lift place_height_m above the destination centroid"), so a
# typo produces a plausible-but-wrong place pose rather than an error.


CATEGORIES = ("TARGET", "DESTINATION", "OBSTACLE")
DEST_TYPES = ("container", "surface", "relation")
DEST_REGIONS = ("left_edge", "right_edge", "center", "far_end", "near_end")
DEST_RELATIONS = ("left_of", "right_of", "in_front_of", "behind", "on_top_of", "near")


# ── Guided-decoding schema ────────────────────────────────────────────────────
# Kept flat — no oneOf/anyOf.  The outlines backend vLLM uses for guided_json
# does not handle discriminated unions, so destination subtypes are merged into
# one object with optional fields.

VLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "bbox_xyxy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["label", "category", "bbox_xyxy"],
            },
        },
        "destination": {
            "type": "object",
            "properties": {
                "reference_label": {"type": "string"},
                "type": {"type": "string", "enum": list(DEST_TYPES)},
                "relation": {"type": "string", "enum": list(DEST_RELATIONS)},
                "region": {"type": "string", "enum": list(DEST_REGIONS)},
            },
            "required": ["reference_label", "type"],
        },
        "target_label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        # Ambiguity signalling. Optional, so a confident scan omits both.
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": "string"},
    },
    # "destination" is deliberately NOT required: a pick-only instruction
    # ("pick up the book") has no destination, and forcing the field makes the
    # model invent one.  See order_detections for the matching category rule.
    "required": ["objects", "target_label", "confidence"],
}


# ── Validated models ──────────────────────────────────────────────────────────

class Detection(BaseModel):
    """One detected object.  Array position determines its mask label value.

    bbox_xyxy length is enforced by VLM_SCHEMA at decode time and re-checked in
    qwen_call.ground(), not by a Field constraint — the min_length/max_length
    kwargs are pydantic v2 only and raise at import under v1.
    """

    label: str
    category: Literal["TARGET", "DESTINATION", "OBSTACLE"]
    bbox_xyxy: list[float]
    confidence: float = 1.0

    def replaced(self, **changes: Any) -> "Detection":
        """Copy with fields replaced — works on both pydantic majors."""
        data = to_dict(self)
        data.update(changes)
        return Detection(**data)


class Destination(BaseModel):
    """Exactly the four strings bt_pkg reads.  Empty string == "not specified"."""

    type: Literal["container", "surface", "relation"]
    reference_label: str
    relation: str = ""
    region: str = ""


class GroundingResult(BaseModel):
    """Structured grounding.  ``destination`` is None for pick-only tasks.

    Serialise with ``to_json(result, exclude_none=True)`` so a None destination
    omits the key rather than emitting null — bt_pkg cannot handle the latter.
    """

    target_label: str
    destination: Optional[Destination] = None
    confidence: float = 1.0

    @property
    def has_destination(self) -> bool:
        return self.destination is not None


# ── Normalisation ─────────────────────────────────────────────────────────────

def target_candidates(objects: list[Detection]) -> list[Detection]:
    """Every detection the VLM marked TARGET, in the order it emitted them.

    Call this BEFORE order_detections, which collapses the list down to one and
    demotes the rest — that normalisation is what makes the mask contract work,
    but it also destroys the evidence that the instruction was ambiguous.
    """
    return [d for d in objects if d.category == "TARGET"]


def order_detections(objects: list[Detection]) -> list[Detection]:
    """Return detections ordered TARGET, DESTINATION, then everything else.

    This ordering IS the mask contract — sam_a100 paints ``index + 1`` as the
    pixel value, so TARGET becomes 1 and DESTINATION becomes 2.

    Extra TARGET/DESTINATION detections beyond the first are demoted to
    OBSTACLE.  ``mask_projection_pkg`` keys /world_map_result by category and
    overwrites on collision, so a second TARGET would silently replace the
    first rather than erroring.
    """
    target: Optional[Detection] = None
    destination: Optional[Detection] = None
    rest: list[Detection] = []

    for det in objects:
        if det.category == "TARGET" and target is None:
            target = det
        elif det.category == "DESTINATION" and destination is None:
            destination = det
        else:
            rest.append(det if det.category == "OBSTACLE"
                        else det.replaced(category="OBSTACLE"))

    ordered = [d for d in (target, destination) if d is not None]
    ordered.extend(rest)
    return ordered


def keep_only(objects: list[Detection], category: str) -> list[Detection]:
    """Demote every detection except the first ``category`` one to OBSTACLE.

    Used by the dual-view path: the overhead pass contributes ONLY the
    destination, and the wrist pass ONLY the target.  Without this both views
    would emit a TARGET, sam_a100 would segment both, and ``build_result_json``
    (which keys /world_map_result by category and overwrites on collision) would
    let the overhead target silently replace the wrist one — the wrist cloud is
    the one grasping depends on.

    Array positions are preserved on purpose.  The mask value is ``index + 1``,
    and ``mask_projection_pkg`` resolves a value back through
    ``detections[value - 1]["category"]``, so demoting in place keeps the join
    valid wherever the survivor happens to sit.
    """
    kept = False
    out: list[Detection] = []
    for det in objects:
        if det.category == category and not kept:
            kept = True
            out.append(det)
        else:
            out.append(det if det.category == "OBSTACLE"
                       else det.replaced(category="OBSTACLE"))
    return out


def scale_boxes(
    objects: list[Detection],
    convention: str,
    src_w: int,
    src_h: int,
) -> list[Detection]:
    """Rescale VLM box coordinates into original-image pixel space.

    VLMs resize before inference, so returned coordinates are usually NOT in
    your original image's pixel space.  Getting this wrong segments a
    confidently incorrect region with no error anywhere in the stack.

    VERIFY THIS EMPIRICALLY on your endpoint before trusting it: ask for one
    box on a known object and check whether the numbers land in original
    pixels, resized pixels, or a normalised 0-1000 grid.

    convention:
      "absolute"        — already in original-image pixels; passed through
      "normalized_1000" — 0..1000 grid (common for Qwen-VL family)
      "normalized_1"    — 0..1 floats
    """
    if convention == "absolute":
        return objects

    if convention == "normalized_1000":
        fx, fy = src_w / 1000.0, src_h / 1000.0
    elif convention == "normalized_1":
        fx, fy = float(src_w), float(src_h)
    else:
        raise ValueError(f"unknown bbox_convention: {convention!r}")

    scaled = []
    for det in objects:
        x1, y1, x2, y2 = det.bbox_xyxy
        scaled.append(det.replaced(bbox_xyxy=[x1 * fx, y1 * fy, x2 * fx, y2 * fy]))
    return scaled


def clamp_boxes(objects: list[Detection], w: int, h: int) -> list[Detection]:
    """Clip boxes to image bounds and drop degenerate ones.

    SAM raises on a box with zero area or one that falls outside the image.
    """
    kept = []
    for det in objects:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1, x2 = sorted((max(0.0, min(x1, w - 1.0)), max(0.0, min(x2, w - 1.0))))
        y1, y2 = sorted((max(0.0, min(y1, h - 1.0)), max(0.0, min(y2, h - 1.0))))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        kept.append(det.replaced(bbox_xyxy=[x1, y1, x2, y2]))
    return kept


def build_labeled_detections(
    objects: list[Detection],
    target_label: str = "",
) -> list[dict[str, Any]]:
    """Serialise to the /qwen/labeled_detections payload.

    ``idx`` is emitted for humans and logs only.  Nothing downstream joins on
    it — mask_projection_pkg and sam_a100 both use array position.  Do not
    reintroduce it as a join key.

    ``target_label`` (the instruction-derived name from GroundingResult) is
    attached to the TARGET entry as a second name.  The two genuinely differ:
    asked for "the glass cup", the VLM describes what it sees as ``label``
    "white cylinder" — accurate, and useless to graspgen_pkg, which keyword-
    matches the TARGET text to decide whether to run transparent depth
    restoration and which grasp profile to use.  Carrying both names means the
    match sees the word the operator actually used.
    """
    out: list[dict[str, Any]] = []
    for i, det in enumerate(objects):
        entry: dict[str, Any] = {
            "idx": i,
            "label": det.label,
            "category": det.category,
            "bbox_xyxy": [round(v, 2) for v in det.bbox_xyxy],
            "confidence": round(det.confidence, 4),
        }
        if det.category == "TARGET" and target_label:
            entry["target_label"] = target_label
        out.append(entry)
    return out
