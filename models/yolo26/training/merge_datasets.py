#!/usr/bin/env python3
"""Phase 2 — merge the per-source Roboflow downloads under raw_downloads/
into a single YOLO-seg dataset under merged/, with unified class IDs.

For each source dataset:
  1. Reads its data.yaml to map source class_id -> source class_name
  2. Remaps each label line to the unified class_id using config.yaml's
     `class_map` for that dataset and `target_classes` for the index
  3. Discards annotations whose source class is not in class_map
  4. Renames images to `<role>_<orig_filename>` to avoid cross-source collisions

After collecting every (image, remapped-labels) pair the script pools them,
discards the source train/valid/test split (often very imbalanced), and
re-splits 70/20/10 with a fixed random seed for reproducibility.

Writes:
  merged/{train,valid,test}/images/*
  merged/{train,valid,test}/labels/*
  merged/data.yaml          (ultralytics-compatible)
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
RAW_DIR = SCRIPT_DIR / "raw_downloads"
MERGED_DIR = SCRIPT_DIR / "merged"
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_source_names(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    cfg = yaml.safe_load(data_yaml.read_text()) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def _remap_label_text(
    text: str,
    src_names: list[str],
    class_map: dict[str, str],
    target_name_to_id: dict[str, int],
) -> list[str]:
    """Return remapped polygon lines. Annotations of unmapped classes are dropped."""
    kept: list[str] = []
    for line in text.strip().splitlines():
        toks = line.split()
        if len(toks) < 2:
            continue
        try:
            src_id = int(float(toks[0]))
        except ValueError:
            continue
        if src_id < 0 or src_id >= len(src_names):
            continue
        src_name = src_names[src_id]
        target_name = class_map.get(src_name)
        if target_name is None:
            continue
        if target_name not in target_name_to_id:
            print(
                f"  WARNING: class_map maps to '{target_name}' but it is not in target_classes",
                file=sys.stderr,
            )
            continue
        new_id = target_name_to_id[target_name]
        kept.append(f"{new_id} " + " ".join(toks[1:]))
    return kept


def _iter_split_dirs(dataset_dir: Path):
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if img_dir.exists():
            yield img_dir, lbl_dir


def _format_count_table(rows: list[tuple], headers: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * widths[i] for i in range(len(headers)))]
    for row in rows:
        out.append("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def main() -> int:
    if not CONFIG_PATH.is_file():
        print(f"ERROR: config not found at {CONFIG_PATH}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    datasets_cfg = cfg.get("datasets", [])
    target_classes_raw = cfg.get("target_classes", {}) or {}
    target_id_to_name = {int(k): v for k, v in target_classes_raw.items()}
    target_name_to_id = {v: k for k, v in target_id_to_name.items()}

    if not target_id_to_name:
        print("ERROR: config.yaml is missing target_classes", file=sys.stderr)
        return 2

    if MERGED_DIR.exists():
        print(f"Wiping previous merged/ at {MERGED_DIR.relative_to(SCRIPT_DIR)}/")
        shutil.rmtree(MERGED_DIR)

    pool: list[tuple[Path, list[str], str]] = []
    per_dataset_counts: dict[str, int] = {}
    per_class_total: dict[str, int] = {n: 0 for n in target_name_to_id}

    print(f"\n{'=' * 78}\nMerging source datasets\n{'=' * 78}")

    for d in datasets_cfg:
        role = d["role"]
        ws = d["workspace"]
        slug = d["project"]
        class_map = d.get("class_map", {}) or {}
        max_images = d.get("max_images")  # optional per-dataset cap
        src_dir = RAW_DIR / f"{ws}__{slug}"

        print(f"\n>>> [{role}] {ws}/{slug}")
        if not src_dir.exists():
            print(f"  SKIP: {src_dir.relative_to(SCRIPT_DIR)}/ not found — run check_datasets.py first")
            continue
        if not class_map:
            print("  SKIP: class_map is empty in config.yaml — fill it in then re-run")
            continue

        src_names = _load_source_names(src_dir)
        print(f"  source classes: {src_names}")
        print(f"  class_map: {class_map}")
        if max_images:
            print(f"  max_images cap: {max_images}")

        # Collect (img_path, new_lines, new_filename, src_anno_count) candidates first.
        candidates: list[tuple[Path, list[str], str, int]] = []
        for img_dir, lbl_dir in _iter_split_dirs(src_dir):
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    src_text = lbl_path.read_text()
                    src_anno_count = len([l for l in src_text.strip().splitlines() if l.strip()])
                    new_lines = _remap_label_text(src_text, src_names, class_map, target_name_to_id)
                else:
                    src_anno_count = 0
                    new_lines = []
                new_filename = f"{role}__{img_path.name}"
                candidates.append((img_path, new_lines, new_filename, src_anno_count))

        # Apply per-dataset cap with deterministic random sampling.
        if max_images and len(candidates) > max_images:
            rng = random.Random(SEED + hash(role) % 1000)
            rng.shuffle(candidates)
            print(f"  capping {len(candidates)} -> {max_images} images")
            candidates = candidates[:max_images]

        n_kept_imgs = 0
        n_kept_annos = 0
        n_dropped_annos = 0
        for img_path, new_lines, new_filename, src_anno_count in candidates:
            n_dropped_annos += src_anno_count - len(new_lines)
            n_kept_annos += len(new_lines)
            for ln in new_lines:
                cid = int(ln.split()[0])
                per_class_total[target_id_to_name[cid]] += 1
            pool.append((img_path, new_lines, new_filename))
            n_kept_imgs += 1

        per_dataset_counts[role] = n_kept_imgs
        print(f"  pooled images: {n_kept_imgs}  kept annotations: {n_kept_annos}  dropped: {n_dropped_annos}")

    if not pool:
        print("\nERROR: no images pooled. Check that raw_downloads/ exists and class_map is filled.", file=sys.stderr)
        return 2

    # Shuffle + split.
    random.seed(SEED)
    random.shuffle(pool)
    n = len(pool)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_valid = int(n * SPLIT_RATIOS["valid"])
    split_buckets = {
        "train": pool[:n_train],
        "valid": pool[n_train:n_train + n_valid],
        "test":  pool[n_train + n_valid:],
    }

    print(f"\n{'=' * 78}\nWriting merged dataset to {MERGED_DIR.relative_to(SCRIPT_DIR)}/\n{'=' * 78}")
    per_split_class_counts = {s: {n: 0 for n in target_name_to_id} for s in split_buckets}
    for split_name, items in split_buckets.items():
        img_out = MERGED_DIR / split_name / "images"
        lbl_out = MERGED_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_src, lines, filename in items:
            shutil.copy2(img_src, img_out / filename)
            label_text = "\n".join(lines)
            if label_text:
                label_text += "\n"
            (lbl_out / (Path(filename).stem + ".txt")).write_text(label_text)
            for ln in lines:
                cid = int(ln.split()[0])
                per_split_class_counts[split_name][target_id_to_name[cid]] += 1

    # Write data.yaml for ultralytics.
    data_yaml = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(target_id_to_name),
        "names": {i: target_id_to_name[i] for i in sorted(target_id_to_name)},
    }
    (MERGED_DIR / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))

    # Summary tables.
    print()
    print("Per-dataset image contribution:")
    print(_format_count_table(
        rows=[(role, n) for role, n in per_dataset_counts.items()],
        headers=["role", "images"],
    ))
    print()
    print(f"Pool total: {len(pool)} images")
    print(f"Splits: train={len(split_buckets['train'])}  valid={len(split_buckets['valid'])}  test={len(split_buckets['test'])}")
    print()
    print("Per-class annotation count by split:")
    headers = ["class"] + list(split_buckets) + ["total"]
    rows = []
    for cname in target_name_to_id:
        per_split = [per_split_class_counts[s][cname] for s in split_buckets]
        rows.append((cname, *per_split, sum(per_split)))
    print(_format_count_table(rows=rows, headers=headers))

    # Sanity warnings.
    print()
    for cname, total in per_class_total.items():
        if total == 0:
            print(f"  WARNING: class '{cname}' has 0 annotations — class_map or source data is wrong")
        elif total < 100:
            print(f"  WARNING: class '{cname}' only has {total} annotations — likely underrepresented")
    for split_name, counts in per_split_class_counts.items():
        empty = [c for c, n in counts.items() if n == 0]
        if empty:
            print(f"  WARNING: split '{split_name}' has zero samples for: {empty}")

    print(f"\nDone. Next step: review merged/data.yaml, then run train.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
