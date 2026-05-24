"""Convert ultralytics Results to the hazard_detections JSON schema."""

from typing import Iterable, Optional


def results_to_hazard_dict(
    result,
    timestamp_ns: int,
    frame_id: str,
    image_width: int,
    image_height: int,
    class_allowlist: Optional[Iterable[int]] = None,
) -> dict:
    """Format an ultralytics Result into the hazard_detections envelope.

    Matches the schema used by the Roboflow workflow test output so downstream
    consumers can handle both interchangeably.
    """
    allow = set(int(c) for c in class_allowlist) if class_allowlist else None
    detections: list = []

    if result.boxes is None or len(result.boxes) == 0:
        return _envelope(timestamp_ns, frame_id, image_width, image_height, detections)

    names = result.names
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    if result.masks is not None and result.masks.xy is not None:
        polygons = [poly for poly in result.masks.xy]
    else:
        polygons = [None] * len(cls_ids)

    for xyxy, cls_id, conf, poly in zip(boxes_xyxy, cls_ids, confs, polygons):
        cls_int = int(cls_id)
        if allow is not None and cls_int not in allow:
            continue
        x1, y1, x2, y2 = xyxy
        det = {
            "class_id": cls_int,
            "class_name": names.get(cls_int, str(cls_int)) if isinstance(names, dict) else str(names[cls_int]),
            "confidence": round(float(conf), 4),
            "bbox": {
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            },
            "polygon": (
                [[int(p[0]), int(p[1])] for p in poly]
                if poly is not None
                else []
            ),
        }
        detections.append(det)

    return _envelope(timestamp_ns, frame_id, image_width, image_height, detections)


def _envelope(timestamp_ns, frame_id, w, h, detections):
    return {
        "hazard_detections": {
            "timestamp": int(timestamp_ns),
            "frame_id": frame_id,
            "image_width": int(w),
            "image_height": int(h),
            "detections": detections,
        }
    }
