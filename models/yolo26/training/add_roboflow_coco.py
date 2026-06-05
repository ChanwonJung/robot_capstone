#!/usr/bin/env python3
"""Convert a Roboflow COCO-segmentation export to YOLOv8 polygon labels and merge into a target dataset (80/10/10 split, deterministic)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import yaml


def read_target_names(data_yaml_path: Path) -> dict:
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return {v: int(k) for k, v in names.items()}
    if isinstance(names, list):
        return {n: i for i, n in enumerate(names)}
    raise ValueError(f"unsupported names format in {data_yaml_path}")


def write_data_yaml(target: Path, names_by_id: dict) -> None:
    payload = {
        "path": str(target.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names_by_id),
        "names": dict(sorted(names_by_id.items())),
    }
    (target / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def convert_split(
    coco_json: Path,
    img_dir: Path,
    target_dir: Path,
    class_remap: dict,
    prefix: str,
    split_ratios: dict,
    seed: int,
) -> dict:
    with open(coco_json) as f:
        coco = json.load(f)

    ann_by_img: dict = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    images = list(coco["images"])
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * split_ratios["train"])
    n_valid = int(n * split_ratios["valid"])

    counts = {"train": 0, "valid": 0, "test": 0}
    class_counts: dict = {}
    skipped_empty = 0

    for i, img in enumerate(images):
        if i < n_train:
            split = "train"
        elif i < n_train + n_valid:
            split = "valid"
        else:
            split = "test"

        w, h = img["width"], img["height"]
        fname = img["file_name"]
        src_img = img_dir / fname
        if not src_img.is_file():
            print(f"WARN: image not found: {src_img}", file=sys.stderr)
            continue

        stem = Path(fname).stem
        new_name = f"{prefix}{stem}{Path(fname).suffix}"
        out_img_dir = target_dir / split / "images"
        out_lbl_dir = target_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_img, out_img_dir / new_name)

        lines = []
        for ann in ann_by_img.get(img["id"], []):
            cid = ann["category_id"]
            if cid not in class_remap:
                continue
            target_cid = class_remap[cid]
            seg = ann.get("segmentation")
            if not isinstance(seg, list):
                continue
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                coords = [
                    f"{(v / w if k % 2 == 0 else v / h):.6f}"
                    for k, v in enumerate(poly)
                ]
                lines.append(f"{target_cid} " + " ".join(coords))
                class_counts[target_cid] = class_counts.get(target_cid, 0) + 1

        (out_lbl_dir / f"{prefix}{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        if not lines:
            skipped_empty += 1
        counts[split] += 1

    return {"split_counts": counts, "class_counts": class_counts, "skipped_empty": skipped_empty}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, type=Path, help="Roboflow COCO-seg export zip")
    p.add_argument("--target", required=True, type=Path, help="Target dataset dir")
    p.add_argument("--copy-from", type=Path, default=None,
                   help="If --target missing, copy this dataset there first (e.g. isaacsim_v3)")
    p.add_argument("--prefix", default="v4_", help="Filename prefix for new samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VALID", "TEST"))
    args = p.parse_args(argv)

    target = args.target.resolve()
    if not target.is_dir():
        if args.copy_from is None:
            print(f"ERROR: target {target} missing and --copy-from not given", file=sys.stderr)
            return 2
        src = args.copy_from.resolve()
        if not src.is_dir():
            print(f"ERROR: --copy-from {src} not found", file=sys.stderr)
            return 2
        print(f"copying {src.name} -> {target.name} ...")
        shutil.copytree(src, target)
        # rewrite data.yaml so path: points at the new dir (v3's path: is absolute)
        names = read_target_names(target / "data.yaml")
        write_data_yaml(target, {v: k for k, v in names.items()})

    data_yaml = target / "data.yaml"
    if not data_yaml.is_file():
        print(f"ERROR: {data_yaml} missing", file=sys.stderr)
        return 2
    target_names = read_target_names(data_yaml)
    print(f"target classes (name -> id): {target_names}")

    stage = target.parent / f".staging_{args.zip.stem}"
    if stage.exists():
        shutil.rmtree(stage)
    print(f"extracting {args.zip.name} -> {stage.name}/ ...")
    stage.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(stage)

    coco_jsons = sorted(stage.rglob("_annotations.coco.json"))
    if not coco_jsons:
        print(f"ERROR: no _annotations.coco.json found under {stage}", file=sys.stderr)
        return 2

    with open(coco_jsons[0]) as f:
        coco_cats = json.load(f)["categories"]
    class_remap = {c["id"]: target_names[c["name"]] for c in coco_cats if c["name"] in target_names}
    unmapped = [c["name"] for c in coco_cats if c["name"] not in target_names]
    if not class_remap:
        print(f"ERROR: no overlapping classes. COCO has {[c['name'] for c in coco_cats]}, "
              f"target has {list(target_names.keys())}", file=sys.stderr)
        return 2
    print(f"class remap (coco_id -> target_id): {class_remap}")
    if unmapped:
        print(f"skipping unmapped COCO classes: {unmapped}")

    ratios = {"train": args.split[0], "valid": args.split[1], "test": args.split[2]}
    overall = {"split_counts": {"train": 0, "valid": 0, "test": 0}, "class_counts": {}, "skipped_empty": 0}
    for cj in coco_jsons:
        print(f"\nprocessing {cj.relative_to(stage)} ...")
        stats = convert_split(cj, cj.parent, target, class_remap, args.prefix, ratios, args.seed)
        for k, v in stats["split_counts"].items():
            overall["split_counts"][k] += v
        for k, v in stats["class_counts"].items():
            overall["class_counts"][k] = overall["class_counts"].get(k, 0) + v
        overall["skipped_empty"] += stats["skipped_empty"]

    shutil.rmtree(stage)

    name_by_id = {v: k for k, v in target_names.items()}
    print("\n=== summary ===")
    print(f"images added per split: {overall['split_counts']}")
    print("annotations per class: " + ", ".join(
        f"{name_by_id[k]}({k}):{v}" for k, v in sorted(overall["class_counts"].items())
    ))
    print(f"images with no labels written: {overall['skipped_empty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
