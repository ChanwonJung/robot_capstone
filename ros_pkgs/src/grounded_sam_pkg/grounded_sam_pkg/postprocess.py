from typing import List, Dict, Any, Tuple

import numpy as np
import torch


def format_detections(detections, phrases: List[str]) -> List[Dict[str, Any]]:
    """
    Convert sv.Detections + phrase labels into a plain list of dicts.

    Returns:
        List of dicts, each with:
            bbox_xyxy   : [x1, y1, x2, y2] in absolute pixel coords
            confidence  : float
            label       : str (matched phrase from GroundingDINO)
    """
    results = []
    confidences = (
        detections.confidence
        if detections.confidence is not None
        else [None] * len(detections.xyxy)
    )
    for i, box in enumerate(detections.xyxy):
        results.append({
            "bbox_xyxy": box.tolist(),
            "confidence": float(confidences[i]) if confidences[i] is not None else None,
            "label": phrases[i] if i < len(phrases) else "",
        })
    return results


def format_masks(masks: torch.Tensor, scores: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Convert SAM output tensors into a plain list of dicts.

    Args:
        masks : bool tensor (N, 1, H, W)
        scores: float tensor (N, 1)
    Returns:
        List of dicts, each with:
            mask  : bool numpy array (H, W)
            score : float
    """
    results = []
    for i in range(masks.shape[0]):
        results.append({
            "mask": masks[i, 0].cpu().numpy().astype(bool),
            "score": float(scores[i, 0]) if scores is not None else None,
        })
    return results


def build_label_map(shape_hw: Tuple[int, int], mask_list: List[Dict[str, Any]]) -> np.ndarray:
    """
    Build a uint8 label map from SAM masks.

    pixel value = 1-based detection index  (0 = background)
    Later detections overwrite earlier ones where masks overlap.

    Args:
        shape_hw  : (H, W) of the original image
        mask_list : output of format_masks()
    Returns:
        label_map : np.ndarray uint8 (H, W)
    """
    label_map = np.zeros(shape_hw, dtype=np.uint8)
    for i, m in enumerate(mask_list, start=1):
        label_map[m["mask"]] = i
    return label_map
