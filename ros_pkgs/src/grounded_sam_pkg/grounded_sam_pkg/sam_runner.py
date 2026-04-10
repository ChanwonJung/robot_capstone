from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


class SAMRunner:
    def __init__(self, model_type: str, checkpoint: str, device: str = "cpu"):
        self.model_type = model_type
        self.checkpoint = str(Path(checkpoint).expanduser())
        self.device = device

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def predict_masks_from_boxes(
        self, image_bgr: np.ndarray, boxes_xyxy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_bgr: BGR numpy array (H, W, 3) — already loaded, not a path
            boxes_xyxy: float32 tensor of shape (N, 4) in absolute pixel coords
        Returns:
            (masks, scores)
            masks: bool tensor (N, 1, H, W)
            scores: float tensor (N, 1)
        """
        import cv2
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_rgb.shape[:2]
        )

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks, scores
