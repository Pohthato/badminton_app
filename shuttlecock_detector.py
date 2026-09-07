"""
Shuttlecock detector with lightweight fallback.
"""
import cv2
import numpy as np

_shuttle_model = None


def init_shuttle_detector(model_path=None):
    """Initialize shuttlecock detector if model available."""
    global _shuttle_model
    if _shuttle_model is not None:
        return _shuttle_model

    if model_path is None:
        from pathlib import Path
        model_path = Path(__file__).parent / "models" / "shuttlecock_yolov8n.pt"

    try:
        from ultralytics import YOLO
        if Path(model_path).exists():
            _shuttle_model = YOLO(str(model_path))
            print(f"[INFO] Shuttle model loaded from {model_path}")
        else:
            print(f"[WARNING] Shuttle model not found at {model_path}. Using fallback.")
            _shuttle_model = False
    except Exception as e:
        print(f"[WARNING] Shuttle detector init failed: {e}")
        _shuttle_model = False

    return _shuttle_model


def detect_shuttlecocks(frame, conf_threshold=0.25):
    """
    Detect shuttlecock in frame.
    Returns list of (x1, y1, x2, y2, conf) tuples.
    """
    global _shuttle_model

    # Try model-based detection first
    if _shuttle_model is None:
        init_shuttle_detector()

    if _shuttle_model and _shuttle_model is not False:
        try:
            results = _shuttle_model(frame, conf=conf_threshold, verbose=False)
            detections = []
            for box in results[0].boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, conf))
            return detections
        except Exception as e:
            print(f"[WARNING] Shuttle model inference failed: {e}")

    # Fallback: color-based detection (white/yellow blob)
    return _detect_by_color(frame)


def _detect_by_color(frame):
    """Fallback color-based shuttlecock detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White/yellow shuttlecock color range
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 500:  # Shuttlecock size range
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / (w + 1e-6)
        if 0.5 < aspect < 2.0:  # Roughly circular/oval
            detections.append((x, y, x+w, y+h, 0.5))

    return detections
