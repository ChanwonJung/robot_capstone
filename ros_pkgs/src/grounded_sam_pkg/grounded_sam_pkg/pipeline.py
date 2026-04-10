from pathlib import Path
from typing import Dict, Any, Union

import cv2
import numpy as np
import torch
import yaml

from .gdino_runner import GroundingDINORunner
from .sam_runner import SAMRunner


class GroundedSAMPipeline:
    def __init__(self, model_config_path: str):
        model_config_path = str(Path(model_config_path).expanduser())
        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        gdino_cfg = cfg["grounding_dino"]
        sam_cfg = cfg["sam"]

        self.gdino = GroundingDINORunner(
            config_file=gdino_cfg["config_file"],
            checkpoint=gdino_cfg["checkpoint"],
            box_threshold=gdino_cfg["box_threshold"],
            text_threshold=gdino_cfg["text_threshold"],
            device=gdino_cfg["device"],
        )

        self.sam = SAMRunner(
            model_type=sam_cfg["model_type"],
            checkpoint=sam_cfg["checkpoint"],
            device=sam_cfg["device"],
        )

    def run(self, image: Union[str, np.ndarray], prompt: str) -> Dict[str, Any]:
        """
        Args:
            image: file path (str) or BGR numpy array (np.ndarray)
                   — str path for standalone use, ndarray when receiving from ROS2 topic
            prompt: GroundingDINO noun phrase, e.g. "bottle . cup"
        Returns:
            dict with keys: detections, phrases, image_bgr, masks, mask_scores
        """
        if isinstance(image, str):
            image_bgr = cv2.imread(str(Path(image).expanduser()))
            if image_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {image}")
        else:
            image_bgr = image

        detections, phrases = self.gdino.predict(image_bgr=image_bgr, prompt=prompt)

        if len(detections.xyxy) == 0:
            return {
                "detections": detections,
                "phrases": phrases,
                "image_bgr": image_bgr,
                "masks": None,
                "mask_scores": None,
            }

        boxes_torch = torch.tensor(
            detections.xyxy, dtype=torch.float32, device=self.sam.device
        )
        masks, mask_scores = self.sam.predict_masks_from_boxes(
            image_bgr=image_bgr,
            boxes_xyxy=boxes_torch,
        )

        return {
            "detections": detections,
            "phrases": phrases,
            "image_bgr": image_bgr,
            "masks": masks,
            "mask_scores": mask_scores,
        }
