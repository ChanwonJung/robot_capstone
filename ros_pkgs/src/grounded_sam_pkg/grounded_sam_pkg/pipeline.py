from pathlib import Path
from typing import Dict, Any, Union

import cv2
import numpy as np
import torch
import yaml

from .gdino_runner import GroundingDINORunner
from .sam_runner import SAMRunner


def _find_project_root(start_path: Path) -> Path:
    for parent in start_path.resolve().parents:
        if (parent / "ros_pkgs").exists() and (parent / "sim").exists():
            return parent
    return start_path.resolve().parents[4]


def _resolve_config_path(path_str: str, *, config_dir: Path, project_root: Path) -> str:
    raw_path = (path_str or "").strip()
    if not raw_path:
        return ""

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    if raw_path.startswith(("models/", "output/", "sim/", "ros_pkgs/")):
        return str((project_root / path).resolve())

    return str((config_dir / path).resolve())


class GroundedSAMPipeline:
    def __init__(self, model_config_path: str):
        model_config_path = Path(model_config_path).expanduser().resolve()
        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config_dir = model_config_path.parent
        project_root = _find_project_root(model_config_path)

        gdino_cfg = cfg["grounding_dino"]
        sam_cfg = cfg["sam"]

        self.gdino = GroundingDINORunner(
            config_file=_resolve_config_path(
                gdino_cfg["config_file"],
                config_dir=config_dir,
                project_root=project_root,
            ),
            checkpoint=_resolve_config_path(
                gdino_cfg["checkpoint"],
                config_dir=config_dir,
                project_root=project_root,
            ),
            box_threshold=gdino_cfg["box_threshold"],
            text_threshold=gdino_cfg["text_threshold"],
            device=gdino_cfg["device"],
        )

        self.sam = SAMRunner(
            model_type=sam_cfg["model_type"],
            checkpoint=_resolve_config_path(
                sam_cfg["checkpoint"],
                config_dir=config_dir,
                project_root=project_root,
            ),
            device=sam_cfg["device"],
        )
        self._logged_runtime_devices = False

    def describe_devices(self) -> Dict[str, Any]:
        gdino_model = getattr(self.gdino.model, "model", None)
        gdino_param = next(gdino_model.parameters(), None) if gdino_model is not None else None
        sam_model = getattr(self.sam.predictor, "model", None)
        sam_param = next(sam_model.parameters(), None) if sam_model is not None else None
        return {
            "torch_cuda_available": torch.cuda.is_available(),
            "gdino_config_device": self.gdino.device,
            "gdino_model_device": str(gdino_param.device) if gdino_param is not None else "unknown",
            "sam_config_device": self.sam.device,
            "sam_model_device": str(sam_param.device) if sam_param is not None else "unknown",
        }

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
        runtime_devices = None
        if not self._logged_runtime_devices:
            runtime_devices = {
                "boxes_device": str(boxes_torch.device),
                "sam_predictor_device": str(next(self.sam.predictor.model.parameters()).device),
            }
            self._logged_runtime_devices = True
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
            "runtime_devices": runtime_devices,
        }
