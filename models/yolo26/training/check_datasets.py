#!/usr/bin/env python3
"""Phase 1 — download each Roboflow source dataset and verify it is usable
for instance-segmentation training.

For every dataset listed in config.yaml the script:
  1. Connects to Roboflow with the API key from $ROBOFLOW_API_KEY
  2. Resolves the latest dataset version (or the explicit version pinned in config)
  3. Downloads it in YOLOv8 format into raw_downloads/<workspace>__<project>/
  4. Reads data.yaml to extract class names and per-split image counts
  5. Inspects one label file to detect format: POLYGON (IS) vs BBOX (OD)
  6. Prints a per-dataset report and an overall pass/fail summary
  7. Writes verification_report.json next to this script

A dataset is GOOD only if its labels are polygons (instance segmentation).
Object-detection or semantic-segmentation exports will be flagged FAIL.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
RAW_DIR = SCRIPT_DIR / "raw_downloads"
REPORT_PATH = SCRIPT_DIR / "verification_report.json"


@dataclass
class DatasetReport:
    role: str
    workspace: str
    project: str
    version: Optional[int]
    project_type: Optional[str] = None
    classes: Optional[list] = None
    train_images: int = 0
    valid_images: int = 0
    test_images: int = 0
    label_format: str = "unknown"
    polygon_points_sample: Optional[int] = None
    download_path: Optional[str] = None
    status: str = "PENDING"
    notes: str = ""

    @property
    def is_good(self) -> bool:
        return self.status == "OK"


def _detect_label_format(label_path: Path) -> tuple[str, Optional[int]]:
    """Return ('POLYGON'|'BBOX'|'EMPTY'|'UNKNOWN', n_points_if_polygon)."""
    try:
        text = label_path.read_text().strip()
    except Exception as exc:
        return f"UNREADABLE ({exc})", None
    if not text:
        return "EMPTY", None
    first_line = text.splitlines()[0].split()
    n_tokens = len(first_line)
    if n_tokens == 5:
        return "BBOX (OD)", None
    if n_tokens > 5 and (n_tokens - 1) % 2 == 0:
        return "POLYGON (IS)", (n_tokens - 1) // 2
    return f"UNKNOWN ({n_tokens} tokens)", None


def _count_images(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
        n += len(list(split_dir.glob(ext)))
    return n


def _first_label(target_dir: Path) -> Optional[Path]:
    for split in ("train", "valid", "test"):
        labels = list((target_dir / split / "labels").glob("*.txt")) if (target_dir / split / "labels").exists() else []
        if labels:
            return labels[0]
    return None


def _inspect(target_dir: Path, report: DatasetReport) -> None:
    data_yaml = target_dir / "data.yaml"
    if not data_yaml.is_file():
        report.status = "FAIL"
        report.notes = "data.yaml not found in downloaded dataset"
        return
    try:
        cfg = yaml.safe_load(data_yaml.read_text()) or {}
    except Exception as exc:
        report.status = "FAIL"
        report.notes = f"failed to parse data.yaml: {exc}"
        return

    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    report.classes = list(names)

    report.train_images = _count_images(target_dir / "train" / "images")
    report.valid_images = _count_images(target_dir / "valid" / "images")
    report.test_images = _count_images(target_dir / "test" / "images")

    sample_label = _first_label(target_dir)
    if sample_label is None:
        report.status = "FAIL"
        report.notes = "no label file found in train/valid/test"
        return

    fmt, n_pts = _detect_label_format(sample_label)
    report.label_format = fmt
    report.polygon_points_sample = n_pts

    if fmt.startswith("POLYGON"):
        report.status = "OK"
        report.notes = f"polygon labels detected ({n_pts} points in first annotation)"
    elif fmt.startswith("BBOX"):
        report.status = "FAIL"
        report.notes = "object-detection labels — not usable for instance segmentation"
    elif fmt == "EMPTY":
        report.status = "FAIL"
        report.notes = "first label file is empty — re-check on Roboflow"
    else:
        report.status = "FAIL"
        report.notes = f"unexpected label format: {fmt}"


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY environment variable is not set.", file=sys.stderr)
        print("  Get your key from https://app.roboflow.com/settings/api", file=sys.stderr)
        print("  Then:   export ROBOFLOW_API_KEY=...", file=sys.stderr)
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: 'roboflow' package not installed. Run:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    export_format = cfg.get("export_format", "yolov8")
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        print("ERROR: no datasets listed in config.yaml", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    reports: list[DatasetReport] = []
    print(f"\n{'=' * 78}")
    print("Roboflow dataset verification")
    print(f"{'=' * 78}")

    for d in datasets_cfg:
        role = d["role"]
        ws = d["workspace"]
        slug = d["project"]
        pinned_version = d.get("version")
        target_dir = RAW_DIR / f"{ws}__{slug}"

        report = DatasetReport(role=role, workspace=ws, project=slug, version=pinned_version)
        print(f"\n>>> [{role}] {ws}/{slug}")

        try:
            project = rf.workspace(ws).project(slug)
            report.project_type = getattr(project, "type", None)
            print(f"  project_type: {report.project_type}")

            versions = project.versions()
            if not versions:
                report.status = "FAIL"
                report.notes = "no dataset versions generated on Roboflow — open the project and click 'Generate New Version'"
                print(f"  FAIL: {report.notes}")
                reports.append(report)
                continue

            chosen = (
                project.version(pinned_version)
                if pinned_version is not None
                else max(versions, key=lambda v: int(getattr(v, "version", 0)))
            )
            chosen_num = int(getattr(chosen, "version", pinned_version or 0))
            report.version = chosen_num
            print(f"  version: v{chosen_num}")

            if not target_dir.exists():
                print(f"  downloading -> {target_dir.relative_to(SCRIPT_DIR)}/ ...")
                chosen.download(export_format, location=str(target_dir))
            else:
                print(f"  already downloaded at {target_dir.relative_to(SCRIPT_DIR)}/ (skip)")
            report.download_path = str(target_dir.relative_to(SCRIPT_DIR))

            _inspect(target_dir, report)

            print(f"  classes ({len(report.classes or [])}): {report.classes}")
            print(f"  images:  train={report.train_images}  valid={report.valid_images}  test={report.test_images}")
            print(f"  labels:  {report.label_format}")
            print(f"  status:  {report.status} — {report.notes}")

        except Exception as exc:
            report.status = "FAIL"
            report.notes = f"exception: {exc.__class__.__name__}: {exc}"
            print(f"  FAIL: {report.notes}")

        reports.append(report)

    print(f"\n{'=' * 78}")
    print("Summary")
    print(f"{'=' * 78}")
    good = [r for r in reports if r.is_good]
    bad = [r for r in reports if not r.is_good]
    for r in reports:
        mark = "OK  " if r.is_good else "FAIL"
        n_imgs = r.train_images + r.valid_images + r.test_images
        print(f"  [{mark}] {r.role:12s} {r.workspace}/{r.project}  ({n_imgs} imgs, classes={r.classes})")
    print(f"\n  Good: {len(good)} / {len(reports)}")

    REPORT_PATH.write_text(json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False))
    print(f"\nReport written to: {REPORT_PATH.relative_to(SCRIPT_DIR)}")

    if bad:
        print(f"\nNext step:")
        print("  - Replace or fix any FAILed datasets, then re-run this script.")
        print("  - For each OK dataset, fill in `class_map` in config.yaml using the printed class names.")
        return 1

    print("\nAll datasets passed verification.")
    print("Next step: fill in class_map for each dataset in config.yaml, then run merge_datasets.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
