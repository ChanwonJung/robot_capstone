# Fast Brain YOLO26-seg training pipeline

End-to-end recipe for training the custom 4-class instance-segmentation model used by `yolo_hazard_pkg`.

Target classes: `hand`, `forearm`, `pet_bottle`, `small_box`.

## Phases

1. **Verify** — download each Roboflow source dataset, confirm it is instance segmentation with polygon labels (`check_datasets.py`)
2. **Merge** — combine all sources into one YOLO-format dataset, remap class IDs (`merge_datasets.py`, added after Phase 1 passes)
3. **Train** — local ultralytics training producing `best.pt` (`train.py`, added after Phase 2)
4. **Export** — optional TensorRT engine for >30 FPS deployment
5. **Swap** — drop the trained file into `models/yolo26/`, point `yolo_hazard_pkg/config/model_paths.yaml` at it

## Phase 1 — Verification (run this now)

### Setup

Use a dedicated Python environment — do **not** install into the perception venv (`gsam_venv`).

```bash
# from the project root
python -m venv .venv-yolo
source .venv-yolo/bin/activate
pip install -r models/yolo26/training/requirements.txt
```

Set your Roboflow API key (do not commit it):

```bash
export ROBOFLOW_API_KEY=...   # from https://app.roboflow.com/settings/api
```

### Run

```bash
cd models/yolo26/training
python check_datasets.py
```

The script will:
- Download each of the 4 datasets into `raw_downloads/`
- Print classes, image counts per split, and the detected label format for each
- Mark each dataset `OK` (polygon / instance-seg) or `FAIL` (bbox / semantic / missing)
- Write `verification_report.json` for later reference

A `FAIL` on any dataset stops the pipeline — replace or fix that dataset in `config.yaml`, then re-run. Already-downloaded datasets are skipped on subsequent runs (delete the corresponding folder under `raw_downloads/` to force a fresh download).

### After verification passes

1. Look at the printed class names per dataset
2. Edit `config.yaml` and fill in `class_map` for each entry, mapping each source class name to one of `hand` / `forearm` / `pet_bottle` / `small_box`. Omit classes you want to discard.
3. Proceed to Phase 2 (`merge_datasets.py`).

## Phase 2 — Merge

Pools every source dataset into one unified YOLO-seg directory, remaps class IDs to the `target_classes` indices, and re-splits 70/20/10 (source splits are discarded because they are often imbalanced).

```bash
cd models/yolo26/training
python merge_datasets.py
```

Output:

```
merged/
  train/{images,labels}/   # ~70 %
  valid/{images,labels}/   # ~20 %
  test/{images,labels}/    # ~10 %
  data.yaml                # ultralytics-compatible
```

The script wipes any previous `merged/` content before writing. Images are renamed `<role>__<original_filename>` to avoid collisions across sources.

Inspect the final per-class counts that the script prints. If any class has 0 samples in a split, fix the source data and re-run.

## Phase 3 — Train

```bash
cd models/yolo26/training
pip install -r requirements.txt          # adds ultralytics + torch
python train.py                          # uses defaults
```

Defaults (override with flags):

| flag | default | notes |
|------|---------|-------|
| `--model` | `yolo26s-seg.pt` | Hub name auto-downloads. If yolo26 not available in your ultralytics build, fall back to `yolo11s-seg.pt` or `yolov8s-seg.pt`. |
| `--epochs` | 100 | |
| `--batch` | 16 | Fits ~640px on 12 GB. Use `-1` for auto. |
| `--imgsz` | 640 | |
| `--device` | `0` | `0,1` for multi-GPU, `cpu` to force CPU. |
| `--name` | `yolo26s-seg-capstone-v1` | run subdirectory under `runs/segment/`. |
| `--patience` | 30 | early stop. |
| `--export` | off | also export TensorRT `.engine` after training. |

Augmentation is hardcoded to follow the Roboflow Agent recommendation for this scenario:
- horizontal flip on, vertical flip / mosaic / mixup off
- ±10° rotation, mild HSV jitter, 10 % translate, ±50 % scale

After training:
- Best weights at `runs/segment/<name>/weights/best.pt`
- Final test-split metrics printed automatically
- Retarget the release symlink to the new run, e.g.
  `ln -sf ../training/runs/segment/<name>/weights/best.pt models/yolo26/release/yolo26s-seg-capstone-best.pt`.
  `yolo_hazard_pkg/config/model_paths.yaml` already points at that symlink, so no yaml edit is needed.

### Export to TensorRT for >30 FPS deployment

```bash
python train.py --no-train --export --name yolo26s-seg-capstone-v2
```

Produces `runs/segment/yolo26s-seg-capstone-v2/weights/best.engine`. Falls back to ONNX if TensorRT is not installed. Retarget the release symlink at the `.engine` file (or update `model_paths.yaml`) to use it.

## Files

| file | purpose |
|------|---------|
| `config.yaml` | dataset registry + class mapping (filled in after Phase 1) |
| `check_datasets.py` | Phase 1 verification |
| `requirements.txt` | Python deps |
| `.gitignore` | excludes `raw_downloads/`, `merged/`, `runs/`, the report, etc. |
| `raw_downloads/` | (generated) per-dataset YOLO exports |
| `verification_report.json` | (generated) machine-readable verification result |
