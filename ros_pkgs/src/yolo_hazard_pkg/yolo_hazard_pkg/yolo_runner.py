"""ultralytics YOLO26-seg wrapper for hazard detection."""

import numpy as np


class YOLORunner:
    """Loads a YOLO instance-segmentation model and runs inference on BGR frames."""

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        half: bool = True,
    ):
        from ultralytics import YOLO

        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.half = half

        self.model = YOLO(weights)

        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = self.model.predict(
            source=dummy,
            device=device,
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            half=half,
            verbose=False,
        )

    def infer(self, bgr_image: np.ndarray):
        """Run inference on a single BGR frame. Returns ultralytics Results[0]."""
        results = self.model.predict(
            source=bgr_image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            half=self.half,
            verbose=False,
        )
        return results[0]
