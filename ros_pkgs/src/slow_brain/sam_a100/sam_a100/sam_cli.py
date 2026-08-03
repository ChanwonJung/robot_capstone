"""Offline smoke test for the SAM 2.1 A100 server — no ROS required.

Mirrors graspgen_cli.py and qwen_cli.py.  Reach for this when the ROS pipeline
produces an empty or wrong mask and you need to know whether the fault is the
tunnel, the server, the boxes, or the label-map composition.

Boxes come either from a Qwen detections JSON (the realistic path) or straight
from the command line:

    python3 -m sam_a100.sam_cli --image demo/resources/ee_raw.png \
        --detections /tmp/dets.json --out /tmp/mask.png --overlay /tmp/overlay.png

    python3 -m sam_a100.sam_cli --image demo/resources/ee_raw.png \
        --box 210 150 330 260 --box 410 180 560 340 --out /tmp/mask.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "sam_a100"

from .label_map import compose_label_map, mask_stats  # noqa: E402
from .sam_client import SamClient  # noqa: E402

SEGMENT_CATEGORIES = ("TARGET", "DESTINATION")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--detections", help="Qwen /qwen/labeled_detections JSON file")
    ap.add_argument("--box", nargs=4, type=float, action="append",
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="explicit box; repeatable. First is treated as TARGET")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--timeout-ms", type=int, default=30000)
    ap.add_argument("--multimask", action="store_true",
                    help="let SAM return 3 candidates per box and keep the best")
    ap.add_argument("--mask-size", nargs=2, type=int, metavar=("W", "H"),
                    help="resize the label map to the depth resolution")
    ap.add_argument("--out", help="write the mono8 label map here")
    ap.add_argument("--overlay", help="write a tinted overlay here")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1
    h, w = image.shape[:2]

    if args.detections:
        dets = json.loads(pathlib.Path(args.detections).read_text())
        picks = [(i, d) for i, d in enumerate(dets)
                 if d.get("category", "").upper() in SEGMENT_CATEGORIES]
        if not picks:
            print("no TARGET/DESTINATION in detections", file=sys.stderr)
            return 1
        indices = [i for i, _ in picks]
        boxes = np.array([d["bbox_xyxy"] for _, d in picks], dtype=np.float32)
        labels = [d.get("label", "?") for _, d in picks]
    elif args.box:
        indices = list(range(len(args.box)))
        boxes = np.array(args.box, dtype=np.float32)
        labels = [f"box{i}" for i in indices]
    else:
        print("need --detections or at least one --box", file=sys.stderr)
        return 1

    try:
        client = SamClient(args.host, args.port, args.timeout_ms)
        masks, scores = client.segment(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes, multimask=args.multimask)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print("\nIf this is a connection error the tunnel is probably down:",
              file=sys.stderr)
        print(f"  ss -tln | grep {args.port}", file=sys.stderr)
        return 1

    out_hw = (args.mask_size[1], args.mask_size[0]) if args.mask_size else (h, w)
    label_map = compose_label_map(masks, indices, out_hw)
    stats = mask_stats(label_map)

    print("=" * 62)
    print("SAM 2.1 segmentation")
    print("=" * 62)
    print(f"source     : {w}x{h}")
    print(f"label map  : {out_hw[1]}x{out_hw[0]}")
    print()
    for n, (idx, label) in enumerate(zip(indices, labels)):
        px = stats.get(idx + 1, 0)
        flag = "  <-- SUSPICIOUSLY SMALL" if px < 50 else ""
        print(f"  [mask={idx + 1}] {label:<20} {px:>8} px  score={scores[n]:.3f}{flag}")
    print()
    print(f"label values present: {sorted(stats)}  (0 = background)")

    if args.out:
        cv2.imwrite(args.out, label_map)
        print(f"\nlabel map -> {args.out}")
        print("  NOTE: values are 1,2,... so it looks almost black. Inspect with:")
        print(f"    python3 -c \"import cv2,numpy;"
              f"print(numpy.unique(cv2.imread('{args.out}',0)))\"")

    if args.overlay:
        vis = image.copy()
        lm = (cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST)
              if label_map.shape != (h, w) else label_map)
        for n, idx in enumerate(indices):
            sel = lm == idx + 1
            if not sel.any():
                continue
            tint = np.zeros_like(vis)
            tint[:] = {1: (0, 200, 80), 2: (0, 220, 255)}.get(idx + 1, (200, 80, 200))
            vis[sel] = cv2.addWeighted(vis, 0.5, tint, 0.5, 0)[sel]
        for (x1, y1, x2, y2) in boxes.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.imwrite(args.overlay, vis)
        print(f"overlay   -> {args.overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
