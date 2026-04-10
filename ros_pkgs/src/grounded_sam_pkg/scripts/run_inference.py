#!/usr/bin/env python3
"""
Standalone inference script — no ROS2 required.

Usage:
    python scripts/run_inference.py --image /path/to/image.jpg --prompt "bottle"
    python scripts/run_inference.py --image /path/to/image.jpg --prompt "bottle, cup"

Output (saved to gsam_ws/output/{initials}/):
    rgb.png           — original input image
    result.jpg        — annotated image (bbox + mask overlay)
    mask_image.png    — uint8 label map (pixel value = 1-based detection index, 0 = background)
    detections.json   — [{index, label, confidence, bbox_xyxy}, ...]

To change weights, edit config/model_paths.yaml (or pass --config).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_sam_pkg.pipeline import GroundedSAMPipeline
from grounded_sam_pkg.postprocess import build_label_map, format_detections, format_masks
from grounded_sam_pkg.prompt_adapter import PromptAdapter
from grounded_sam_pkg.visualizer import draw_bboxes, draw_masks, save_result

DEFAULT_CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "model_paths.yaml")
_WS_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_BASE = _WS_ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description="Grounded SAM standalone inference")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Object prompt, e.g. 'bottle' or 'bottle, cup'")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to model_paths.yaml")
    args = parser.parse_args()

    # initials from prompt nouns (e.g. "glass, cup, bottle" → "gcb")
    initials = "".join(p.strip()[0] for p in args.prompt.split(",") if p.strip())
    out_dir = OUTPUT_BASE / initials
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = PromptAdapter()
    adapted_prompt = adapter.adapt(args.prompt)
    print(f"[prompt]   '{args.prompt}'  →  '{adapted_prompt}'")
    print(f"[output]   {out_dir}/")

    print("[pipeline] Loading models (this takes a moment on CPU)...")
    pipeline = GroundedSAMPipeline(args.config)

    print(f"[pipeline] Running inference on: {args.image}")
    result = pipeline.run(image=args.image, prompt=adapted_prompt)

    det_list = format_detections(result["detections"], result["phrases"])
    print(f"\n[result]   {len(det_list)} detection(s):")
    for i, d in enumerate(det_list, start=1):
        print(f"  [{i}] label={d['label']!r}  conf={d['confidence']:.3f}  bbox={d['bbox_xyxy']}")

    # --- rgb.png : original image ---
    cv2.imwrite(str(out_dir / "rgb.png"), result["image_bgr"])
    print(f"[saved]    rgb.png")

    if result["masks"] is not None:
        mask_list = format_masks(result["masks"], result["mask_scores"])

        # --- result.jpg : annotated visualization (class-colored masks) ---
        mask_labels = [d["label"] for d in det_list]
        vis = draw_bboxes(result["image_bgr"], det_list)
        vis = draw_masks(vis, mask_list, labels=mask_labels)
        save_result(vis, str(out_dir / "result.jpg"))
        print(f"[saved]    result.jpg")

        # --- mask_image.png : uint8 label map (pixel = detection index, 0=bg) ---
        label_map = build_label_map(result["image_bgr"].shape[:2], mask_list)
        cv2.imwrite(str(out_dir / "mask_image.png"), label_map)
        print(f"[saved]    mask_image.png  (values: 0=bg, 1~{len(mask_list)}=objects)")

        # --- detections.json : index → label mapping ---
        det_with_index = [{"index": i, **d} for i, d in enumerate(det_list, start=1)]
        with open(out_dir / "detections.json", "w") as f:
            json.dump(det_with_index, f, indent=2)
        print(f"[saved]    detections.json")

    else:
        print("\n[result]   No detections — nothing to save.")


if __name__ == "__main__":
    main()
