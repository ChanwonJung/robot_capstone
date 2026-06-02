#!/usr/bin/env python3
"""
test_qwen_call.py  —  smoke-test for classify_detections() without ROS.

Usage:
  python3 test_qwen_call.py                          # run all scenarios
  python3 test_qwen_call.py --scenario bowl          # single scenario
  python3 test_qwen_call.py --endpoint http://localhost:8000/v1 --model Qwen/Qwen3.5-9B

The script resolves qwen_call.py from the repo layout, so run it from the
repo root or any subdirectory (it walks up until it finds qwen_call.py).
"""

import argparse
import json
import sys
from pathlib import Path


# ── Resolve qwen_call on sys.path without needing a ROS workspace ─────────────

def _find_qwen_pkg() -> Path:
    """Walk up from __file__ until we find qwen_call.py, then return its parent."""
    candidate = Path(__file__).resolve().parent
    for _ in range(8):  # guard against runaway
        target = candidate / "ros_pkgs/src/qwen_pkg/qwen_pkg/qwen_call.py"
        if target.exists():
            return target.parent
        candidate = candidate.parent
    raise FileNotFoundError(
        "Could not locate ros_pkgs/src/qwen_pkg/qwen_pkg/qwen_call.py "
        "relative to this script. Run from the repo root."
    )

sys.path.insert(0, str(_find_qwen_pkg()))
from qwen_call import classify_detections  # noqa: E402  (path set above)


# ── Fake detection scenarios ───────────────────────────────────────────────────
# Each entry: (scenario_name, instruction, detections_list)
# bbox_xyxy format: [x1, y1, x2, y2] in image pixels (arbitrary for this test)

SCENARIOS: dict[str, tuple[str, list[dict]]] = {
    "bowl": (
        "Pick up the apple and place it in the bowl.",
        [
            {"idx": 0, "label": "apple",  "confidence": 0.91, "bbox_xyxy": [120, 80,  200, 160]},
            {"idx": 1, "label": "bowl",   "confidence": 0.87, "bbox_xyxy": [300, 200, 420, 300]},
            {"idx": 2, "label": "bottle", "confidence": 0.78, "bbox_xyxy": [500, 100, 560, 240]},
            {"idx": 3, "label": "book",   "confidence": 0.65, "bbox_xyxy": [50,  300, 180, 380]},
        ],
    ),
    "shelf": (
        "Move the red cube to the left edge of the shelf.",
        [
            {"idx": 0, "label": "red cube",  "confidence": 0.93, "bbox_xyxy": [200, 150, 280, 230]},
            {"idx": 1, "label": "shelf",     "confidence": 0.85, "bbox_xyxy": [50,  350, 600, 420]},
            {"idx": 2, "label": "blue cube", "confidence": 0.80, "bbox_xyxy": [330, 150, 410, 230]},
        ],
    ),
    "relation": (
        "Put the orange next to the mug.",
        [
            {"idx": 0, "label": "orange", "confidence": 0.88, "bbox_xyxy": [100, 100, 180, 180]},
            {"idx": 1, "label": "mug",    "confidence": 0.82, "bbox_xyxy": [350, 120, 430, 220]},
            {"idx": 2, "label": "plate",  "confidence": 0.71, "bbox_xyxy": [200, 300, 400, 380]},
            {"idx": 3, "label": "spoon",  "confidence": 0.60, "bbox_xyxy": [450, 300, 500, 390]},
        ],
    ),
    "ambiguous": (
        "Stack the small box on top of the big box.",
        [
            {"idx": 0, "label": "small box", "confidence": 0.89, "bbox_xyxy": [150, 200, 220, 270]},
            {"idx": 1, "label": "big box",   "confidence": 0.91, "bbox_xyxy": [250, 180, 400, 350]},
            {"idx": 2, "label": "pen",       "confidence": 0.55, "bbox_xyxy": [50,  100, 70,  160]},
        ],
    ),
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_scenario(
    name: str,
    instruction: str,
    detections: list[dict],
    endpoint: str,
    model: str,
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"SCENARIO : {name}")
    print(f"INSTRUCTION : {instruction}")
    print(f"INPUT DETECTIONS ({len(detections)}):")
    print(json.dumps(detections, indent=2))
    print(sep)

    try:
        enriched, grounding = classify_detections(
            detections=detections,
            instruction=instruction,
            endpoint_url=endpoint,
            model=model,
        )
    except Exception as exc:
        import traceback
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return

    grounding_dict = (
        grounding.model_dump()
        if hasattr(grounding, "model_dump")
        else json.loads(grounding.json())
    )

    print("ENRICHED DETECTIONS (with 'category' added):")
    print(json.dumps(enriched, indent=2))
    print()
    print("GROUNDING RESULT:")
    print(json.dumps(grounding_dict, indent=2))

    # Quick sanity checks — warn but don't crash
    target_det = next((d for d in enriched if d["idx"] == grounding.target_id), None)
    if target_det is None:
        print(f"[WARN] target_id={grounding.target_id} not found in enriched detections")
    elif target_det.get("category") != "TARGET":
        print(
            f"[WARN] target_id={grounding.target_id} has category="
            f"'{target_det.get('category')}', expected 'TARGET'"
        )

    dest_ref = grounding.destination.reference_id
    dest_det = next((d for d in enriched if d["idx"] == dest_ref), None)
    if dest_det is None:
        print(f"[WARN] dest_reference_id={dest_ref} not found in enriched detections")
    elif dest_det.get("category") != "DESTINATION":
        print(
            f"[WARN] dest_reference_id={dest_ref} has category="
            f"'{dest_det.get('category')}', expected 'DESTINATION'"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test classify_detections()")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1",
        help="vLLM base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Model name registered in the vLLM server",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Run a single scenario (default: run all)",
    )
    args = parser.parse_args()

    targets = (
        {args.scenario: SCENARIOS[args.scenario]}
        if args.scenario
        else SCENARIOS
    )

    print(f"Endpoint : {args.endpoint}")
    print(f"Model    : {args.model}")
    print(f"Scenarios: {list(targets.keys())}")

    for name, (instruction, detections) in targets.items():
        run_scenario(name, instruction, detections, args.endpoint, args.model)

    print("\n" + "─" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
