"""Offline smoke test for the Qwen A100 endpoint — no ROS required.

Mirrors graspgen_cli.py.  This is the tool you reach for when the ROS pipeline
is silent and you need to know whether the fault is the tunnel, the server, the
prompt, or the box convention.

    python3 qwen_a100/qwen_cli.py --image demo/resources/ee_raw.png \
        --instruction "put the book in the box"

Verify the box convention with --annotate: if the drawn boxes do not land on
the objects, try --bbox-convention normalized_1000.  Getting this wrong is
silent everywhere else in the stack.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2

# Allow running this file directly (`python3 .../qwen_cli.py`) as well as via
# `python3 -m qwen_a100.qwen_cli`.  qwen_call imports schema relatively, so the
# package context has to exist either way — bootstrap it when absent.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "qwen_a100"

from .qwen_call import ground  # noqa: E402
from .qwen_schema import to_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="path to an EE camera frame")
    ap.add_argument("--instruction", required=True, help="natural-language command")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen35-local",
                    help="served model id on the vLLM endpoint (Qwen3.5-27B)")
    ap.add_argument("--bbox-convention", default="absolute",
                    choices=["absolute", "normalized_1000", "normalized_1"])
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--annotate", metavar="PATH",
                    help="write a box-overlay PNG here to eyeball the coordinates")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1

    try:
        objects, grounding, meta = ground(
            image,
            args.instruction,
            endpoint_url=args.endpoint,
            model=args.model,
            bbox_convention=args.bbox_convention,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print("\nIf this is a connection error, the tunnel is probably down:",
              file=sys.stderr)
        print("  ss -tln | grep 8000", file=sys.stderr)
        return 1

    print("=" * 62)
    print("Qwen grounding result")
    print("=" * 62)
    print(f"source        : {meta['source_size'][0]}x{meta['source_size'][1]}")
    print(f"instruction   : {args.instruction}")
    print(f"target_label  : {grounding.target_label}")
    d = grounding.destination
    print(f"destination   : {d.reference_label}  type={d.type} "
          f"relation={d.relation or '-'} region={d.region or '-'}")
    print(f"confidence    : {grounding.confidence:.3f}")
    if meta["warning"]:
        print(f"WARNING       : {meta['warning']}")
    print()
    print(f"{len(objects)} objects (array order == mask label value):")
    for i, det in enumerate(objects):
        x1, y1, x2, y2 = (round(v) for v in det.bbox_xyxy)
        print(f"  [mask={i + 1}] {det.category:<11} {det.label:<20} "
              f"box=({x1},{y1})-({x2},{y2}) conf={det.confidence:.2f}")

    if args.annotate:
        vis = image.copy()
        for i, det in enumerate(objects):
            x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
            color = {"TARGET": (0, 200, 80), "DESTINATION": (0, 220, 255)}.get(
                det.category, (40, 40, 220))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{i + 1}:{det.label}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite(args.annotate, vis)
        print(f"\nannotated overlay written to {args.annotate}")

    print()
    print("payload that would go to /qwen/grounding_result:")
    print(json.dumps(json.loads(to_json(grounding)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
