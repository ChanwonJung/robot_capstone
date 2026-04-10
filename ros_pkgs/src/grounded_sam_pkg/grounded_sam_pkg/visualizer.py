from pathlib import Path
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np


def draw_bboxes(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of the image.
    Box color matches the mask color for the same class.
    """
    img = image_bgr.copy()
    class_index_map: dict = {}
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = det.get("label", "")
        conf = det.get("confidence") or 0.0
        color = tuple(int(c) for c in _class_color(label, class_index_map))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return img



# High-contrast BGR palette cycled per unique class name
_CLASS_PALETTE = [
    (  0, 255, 255),  # yellow
    (255,   0, 255),  # magenta
    (255, 255,   0),  # cyan
    (  0, 128, 255),  # orange
    (  0, 255,   0),  # green
    (255,   0,   0),  # blue
    (128,   0, 255),  # violet
    (  0, 255, 128),  # lime
]


def _class_color(label: str, class_index_map: dict) -> np.ndarray:
    if label not in class_index_map:
        class_index_map[label] = len(class_index_map)
    idx = class_index_map[label]
    return np.array(_CLASS_PALETTE[idx % len(_CLASS_PALETTE)], dtype=np.uint8)


def draw_masks(
    image_bgr: np.ndarray,
    masks: List[Dict[str, Any]],
    labels: List[str] = None,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay semi-transparent colored masks onto a copy of the image.

    Args:
        masks:  output of format_masks()
        labels: class label per mask (same order). If provided, same class → same color.
                If None, each mask gets a unique color.
    """
    img = image_bgr.copy()
    overlay = img.copy()
    class_index_map: dict = {}

    for i, m in enumerate(masks):
        if labels and i < len(labels):
            color = _class_color(labels[i], class_index_map)
        else:
            color = np.array(_CLASS_PALETTE[i % len(_CLASS_PALETTE)], dtype=np.uint8)

        mask = m["mask"].astype(bool)
        overlay[mask] = color

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def save_result(image_bgr: np.ndarray, output_path: str) -> None:
    """Save image to disk, creating parent directories if needed."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image_bgr)
