#!/usr/bin/env python3
"""Phase 3 — train a YOLO26-seg instance-segmentation model on the merged dataset.

Uses ultralytics' Python API. Augmentation params follow the Roboflow Agent
recommendation for our hazard-detection scenario: horizontal flip on, vertical
flip / 90° rotation / mosaic off, mild hue/saturation, ±10° rotation.

Outputs:
  runs/segment/<run-name>/weights/best.pt    <-- copy this to models/yolo26/best.pt
  runs/segment/<run-name>/weights/last.pt
  runs/segment/<run-name>/...                (metrics, plots, confusion matrix, etc.)

Optional TensorRT export with --export produces an .engine alongside best.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_DIR = SCRIPT_DIR / "merged"
# Default points at the current active run setup (v3 IsaacSim-captured dataset).
# Bump these per training campaign — CLI args still override everything.
DEFAULT_DATA_YAML = SCRIPT_DIR / "datasets" / "isaacsim_v3" / "data.yaml"
RUNS_DIR = SCRIPT_DIR / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26-seg on the merged hazard dataset.")
    p.add_argument("--model", default="yolo26m-seg.pt",
                   help="Starting weights. Hub names auto-download (yolo26n/s/m/l-seg.pt). "
                        "Fall back to yolo11s-seg.pt or yolov8s-seg.pt if yolo26 not available in your ultralytics version.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=-1,
                   help="Batch size. -1 = AutoBatch (recommended). 16 fits ~640px on 12 GB VRAM for s-seg.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0",
                   help="CUDA device id ('0', '0,1') or 'cpu'.")
    p.add_argument("--name", default="yolo26m-seg-capstone-v3",
                   help="Subdirectory name under runs/segment/. Bump the suffix (v3, v4, …) per run.")
    p.add_argument("--data", default=str(DEFAULT_DATA_YAML),
                   help="Path to data.yaml. Defaults to the v3 IsaacSim-captured dataset.")
    p.add_argument("--patience", type=int, default=50,
                   help="Early-stop patience in epochs.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="ram",
                   help="'ram', 'disk', or 'False'. Set to 'False' if RAM is tight.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last checkpoint of --name.")
    p.add_argument("--export", action="store_true",
                   help="After training, export TensorRT engine for fast inference.")
    p.add_argument("--no-train", action="store_true",
                   help="Skip training (useful with --export to just convert an existing best.pt).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    DATA_YAML = Path(args.data).resolve()
    if not DATA_YAML.is_file():
        print(f"ERROR: data.yaml not found: {DATA_YAML}",
              file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: 'ultralytics' not installed. Activate your venv and run:", file=sys.stderr)
        print("  pip install ultralytics torch", file=sys.stderr)
        return 2

    print(f"\n{'=' * 78}\nYOLO26-seg training\n{'=' * 78}")
    try:
        data_disp = DATA_YAML.relative_to(SCRIPT_DIR)
    except ValueError:
        data_disp = DATA_YAML
    print(f"  data:      {data_disp}")
    print(f"  model:     {args.model}")
    print(f"  epochs:    {args.epochs}")
    print(f"  batch:     {args.batch}")
    print(f"  imgsz:     {args.imgsz}")
    print(f"  device:    {args.device}")
    print(f"  output:    {RUNS_DIR.relative_to(SCRIPT_DIR)}/segment/{args.name}/")
    print(f"{'=' * 78}\n")

    model = YOLO(args.model)

    if not args.no_train:
        cache_val = args.cache
        if isinstance(cache_val, str) and cache_val.lower() in ("false", "no", "0"):
            cache_val = False

        model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=str(RUNS_DIR / "segment"),
            name=args.name,
            patience=args.patience,
            workers=args.workers,
            cache=cache_val,
            resume=args.resume,
            cos_lr=True,
            save_period=10,
            # Augmentation — v3 IsaacSim-domain capture (small dataset, strong aug).
            # Bumped mosaic/copy_paste/scale/degrees vs v2; mixup/flipud/shear stay off.
            mosaic=0.5,
            close_mosaic=10,      # mosaic OFF for the last 10 epochs (stabilises convergence)
            copy_paste=0.3,       # IS-specific aug — paste instances across images
            mixup=0.0,
            degrees=15.0,
            translate=0.15,
            scale=0.7,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.4,
        )

        print("\nValidating best.pt on test split...")
        model.val(data=str(DATA_YAML), split="test")

    run_dir = RUNS_DIR / "segment" / args.name
    best_pt = run_dir / "weights" / "best.pt"

    if args.export:
        if not best_pt.is_file():
            print(f"ERROR: cannot export — {best_pt} not found", file=sys.stderr)
            return 2
        print(f"\nExporting TensorRT engine from {best_pt.relative_to(SCRIPT_DIR)} ...")
        export_model = YOLO(str(best_pt))
        try:
            export_model.export(format="engine", half=True, imgsz=args.imgsz, device=args.device)
            print("Export done.")
        except Exception as exc:
            print(f"TensorRT export failed: {exc}", file=sys.stderr)
            print("  Falling back to ONNX export...", file=sys.stderr)
            try:
                export_model.export(format="onnx", imgsz=args.imgsz)
            except Exception as exc2:
                print(f"  ONNX export also failed: {exc2}", file=sys.stderr)
                return 1

    print(f"\n{'=' * 78}")
    print("Training done.")
    print(f"{'=' * 78}")
    if best_pt.is_file():
        print(f"\nBest weights: {best_pt.relative_to(SCRIPT_DIR)}")
        print()
        print("Next step — wire into ROS:")
        print(f"  1. Retarget the release symlink to this run's best.pt:")
        print(f"       ln -sfn ../training/runs/segment/{args.name}/weights/best.pt \\")
        print(f"               models/yolo26/release/capstone-hazard-seg-best.pt")
        print(f"     (model_paths.yaml already points at the symlink — no yaml edit needed)")
        print(f"  2. Confirm ros_pkgs/src/yolo_hazard_pkg/config/runtime.yaml:")
        print(f"       filter_by_class: true")
        print(f"       class_allowlist: [0, 1, 2]   # arm, bottle, box")
        print(f"  3. colcon build --packages-select yolo_hazard_pkg && source install/setup.bash")
        print(f"  4. ros2 launch yolo_hazard_pkg yolo_hazard_both.launch.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
