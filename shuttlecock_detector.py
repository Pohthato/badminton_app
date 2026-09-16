"""
World-Class Shuttlecock Detector
Uses best.pt (YOLOv8m-pose trained on shuttlecock data) as primary detector.
Falls back to color-based detection if model unavailable.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

_shuttle_model = None


def init_shuttle_detector(model_path=None):
    """Initialize world-class shuttlecock detector using best.pt."""
    global _shuttle_model
    if _shuttle_model is not None:
        return _shuttle_model

    if model_path is None:
        model_path = Path(__file__).parent / "Models" / "best.pt"

    try:
        from ultralytics import YOLO
        import torch
        if Path(model_path).exists():
            _shuttle_model = YOLO(str(model_path))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _shuttle_model.to(device)
            logger.info(f"[SHUTTLE] World-class model loaded from {model_path} on {device}")
        else:
            logger.warning(f"[SHUTTLE] Model not found at {model_path}. Using fallback.")
            _shuttle_model = False
    except Exception as e:
        logger.error(f"[SHUTTLE] Detector init failed: {e}")
        _shuttle_model = False

    return _shuttle_model


def detect_shuttlecocks(frame, conf_threshold=0.25):
    """Detect shuttlecock in frame. Returns list of (x1, y1, x2, y2, conf) tuples."""
    global _shuttle_model

    if _shuttle_model is None:
        init_shuttle_detector()

    if _shuttle_model and _shuttle_model is not False:
        try:
            results = _shuttle_model(frame, conf=conf_threshold, verbose=False)
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0]) if box.conf is not None else 0.0
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = map(float, box.xyxy[0])
                            detections.append((x1, y1, x2, y2, conf))


def detect_shuttle_keypoints(frame, conf_threshold=0.25):
    """Detect shuttlecock with full keypoint data for advanced tracking."""
    global _shuttle_model

    if _shuttle_model is None:
        init_shuttle_detector()

    detections = []
    if _shuttle_model and _shuttle_model is not False:
        try:
            results = _shuttle_model(frame, conf=conf_threshold, verbose=False)
            for result in results:
                if result.keypoints is not None:
                    for i, kpts in enumerate(result.keypoints):
                        if kpts.conf is not None and len(kpts.conf) > 0:
                            kp_conf = float(kpts.conf[0][0])
                            kp_x = float(kpts.xy[0][0][0])
                            kp_y = float(kpts.xy[0][0][1])
                            bbox = None
                            conf = kp_conf
                            if result.boxes is not None and i < len(result.boxes):
                                bbox = tuple(map(float, result.boxes[i].xyxy[0]))
                                conf = float(result.boxes[i].conf[0])
                            if kp_conf > 0.15:
                                detections.append({
                                    'center': (kp_x, kp_y),
                                    'center_x': kp_x,
                                    'center_y': kp_y,
                                    'confidence': kp_conf,
                                    'bbox': bbox,
                                    'bbox_confidence': conf
                                })
        except Exception as e:
            logger.warning(f"[SHUTTLE] Keypoint detection failed: {e}")
    return detections


def _detect_by_color(frame):
    """Fallback color-based shuttlecock detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    lower_yellow = np.array([20, 50, 150])
    upper_yellow = np.array([40, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15 or area > 800:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / (w + 1e-6)
        if 0.3 < aspect < 3.0:
            detections.append((float(x), float(y), float(x+w), float(y+h), 0.4))
    return detections

                if result.keypoints is not None and not detections:
                    for kpts in result.keypoints:
                        if kpts.conf is not None and len(kpts.conf) > 0:
                            kp_conf = float(kpts.conf[0][0])
                            kp_x, kp_y = float(kpts.xy[0][0][0]), float(kpts.xy[0][0][1])
                            if kp_conf > 0.2:
                                r = 15
                                detections.append((kp_x - r, kp_y - r, kp_x + r, kp_y + r, kp_conf))
            return detections
        except Exception as e:
            logger.warning(f"[SHUTTLE] Model inference failed: {e}")

    return _detect_by_color(frame)
