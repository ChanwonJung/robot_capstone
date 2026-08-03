"""Compose SAM binary masks into the mono8 label map mask_projection_pkg wants.

ROS-free.  This is the highest-risk conversion in the Slow Brain: every way of
getting it wrong fails SILENTLY downstream, so it lives here on its own and is
unit-testable without a graph.

The contract, from mask_projection_pkg/label_mapper.py:

  * ``mono8`` single channel.  The projector calls imgmsg_to_cv2 with
    desired_encoding='mono8'; publish bgr8 and cv_bridge silently runs a
    luma-weighted colour conversion that destroys the integer labels.
  * Pixel value ``v`` is a **1-based index into the detections array**:
    ``detections[v - 1]``.  0 is background.  The join is by ARRAY POSITION,
    never by any id field.
  * Shape must equal the **EE DEPTH** image, not the RGB.  The projector indexes
    the mask with pixel coordinates generated from the depth grid.  Too small
    raises IndexError inside its callback; too large silently mislabels.

qwen_a100 orders detections TARGET first, DESTINATION second, so painting
``index + 1`` gives target=1, destination=2 — which also happens to match the
projector's legacy positional fallback table.
"""
from __future__ import annotations

import cv2
import numpy as np


def compose_label_map(
    masks: np.ndarray,
    indices: list[int],
    out_hw: tuple[int, int],
    target_priority: bool = True,
) -> np.ndarray:
    """Paint binary masks into a single mono8 label map.

    Parameters
    ----------
    masks:
        (N, H, W) boolean or 0/1 array, one per segmented object, at the
        SOURCE image resolution.
    indices:
        For each mask, its 0-based position in the detections array.  Painted
        value is ``index + 1``.  Must be the same length as ``masks``.
    out_hw:
        (height, width) of the EE depth image.  Masks are resized to this with
        nearest-neighbour — any interpolating filter invents fractional label
        values and the 1/2 label map becomes meaningless.
    target_priority:
        Where two masks overlap, let the LOWER detection index win (i.e. TARGET
        beats DESTINATION).  This deliberately inverts GSAM's convention, where
        later detections overwrite earlier ones: a target occluded by its own
        destination is the common case (a book already over the box), and grasp
        quality depends on the target's mask being intact.  Set False to match
        GSAM byte-for-byte.

    Returns
    -------
    (H, W) uint8 label map, 0 = background.

    """
    if len(masks) != len(indices):
        raise ValueError(f"{len(masks)} masks but {len(indices)} indices")

    out_h, out_w = out_hw
    label_map = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(masks) == 0:
        return label_map

    # Paint highest index first so the lowest ends up on top.
    order = sorted(range(len(indices)), key=lambda i: indices[i],
                   reverse=target_priority)

    for i in order:
        value = indices[i] + 1
        if not 1 <= value <= 255:
            raise ValueError(
                f"detection index {indices[i]} maps to mask value {value}, "
                "outside the uint8 range a mono8 label map can carry")

        m = np.asarray(masks[i])
        if m.ndim != 2:
            raise ValueError(f"mask {i} has shape {m.shape}, expected 2-D")

        binary = (m > 0).astype(np.uint8)
        if binary.shape != (out_h, out_w):
            binary = cv2.resize(binary, (out_w, out_h),
                                interpolation=cv2.INTER_NEAREST)

        label_map[binary > 0] = value

    return label_map


def mask_stats(label_map: np.ndarray) -> dict[int, int]:
    """Pixel count per label value, excluding background.

    Empty or near-empty entries are the tell-tale of a wrong bbox convention
    upstream — SAM segments *something* inside a garbage box, just not the
    object you meant, and often only a handful of pixels.
    """
    values, counts = np.unique(label_map, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts) if v != 0}
