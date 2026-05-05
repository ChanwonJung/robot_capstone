from pathlib import Path
from typing import Tuple

import numpy as np
from groundingdino.util.inference import Model


class GroundingDINORunner:
    def __init__(
        self,
        config_file: str,
        checkpoint: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cpu",
    ):
        raw_config_file = (config_file or "").strip()
        self.config_file = str(Path(raw_config_file).expanduser()) if raw_config_file else ""
        if not self.config_file or not Path(self.config_file).exists() or Path(self.config_file).is_dir():
            try:
                import groundingdino as _gdino
                self.config_file = str(
                    Path(_gdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
                )
            except ImportError:
                pass
        self.checkpoint = str(Path(checkpoint).expanduser())
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        self.model = Model(
            model_config_path=self.config_file,
            model_checkpoint_path=self.checkpoint,
            device=self.device,
        )

    def predict(self, image_bgr: np.ndarray, prompt: str) -> Tuple:
        """
        Args:
            image_bgr: BGR numpy array (H, W, 3)
            prompt: GroundingDINO noun phrase, e.g. "bottle . cup"
        Returns:
            (sv.Detections, List[str]) — detections and matched phrase labels
        """
        detections, phrases = self.model.predict_with_caption(
            image=image_bgr,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        return detections, phrases
