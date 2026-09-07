"""
Data processing utilities for badminton detection and tracking.
"""
import numpy as np


def apply_nms(boxes, overlap_threshold=0.35):
    """Apply Non-Maximum Suppression to bounding boxes."""
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=np.float64)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= overlap_threshold)[0]
        order = order[inds + 1]

    return [tuple(map(int, boxes_arr[k])) for k in keep]


def box_iou(box_a, box_b):
    """Compute IoU between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / (union + 1e-6)


def center_distance(box_a, box_b):
    """Compute distance between box centers."""
    ca_x = (box_a[0] + box_a[2]) / 2
    ca_y = (box_a[1] + box_a[3]) / 2
    cb_x = (box_b[0] + box_b[2]) / 2
    cb_y = (box_b[1] + box_b[3]) / 2
    return np.sqrt((ca_x - cb_x)**2 + (ca_y - cb_y)**2)


def filter_close_detections(detections, min_distance_px=25):
    """Remove detections that are too close to each other."""
    if not detections:
        return detections

    filtered = []
    for det in detections:
        too_close = False
        for existing in filtered:
            if center_distance(det, existing) < min_distance_px:
                too_close = True
                break
        if not too_close:
            filtered.append(det)
    return filtered


def merge_additional_detections(existing, new_dets, iou_threshold=0.5):
    """Merge new detections with existing ones."""
    merged = list(existing)
    for new_det in new_dets:
        is_new = True
        for existing_det in merged:
            if box_iou(new_det, existing_det) > iou_threshold:
                is_new = False
                break
        if is_new:
            merged.append(new_det)
    return merged


def assign_detections_to_tracks(detections, tracks, cost_threshold=100):
    """Assign detections to existing tracks using greedy matching."""
    if not tracks or not detections:
        return {}, list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            cost_matrix[t_idx, d_idx] = center_distance(track.box, det["box"])

    matched = {}
    unmatched_dets = list(range(len(detections)))

    while cost_matrix.size > 0:
        min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
        t_idx, d_idx = min_idx

        if cost_matrix[t_idx, d_idx] > cost_threshold:
            break

        track_id = tracks[t_idx].track_id
        matched[track_id] = d_idx

        # Remove row and column
        cost_matrix = np.delete(cost_matrix, t_idx, axis=0)
        tracks.pop(t_idx)
        unmatched_dets.remove(d_idx)

    return matched, unmatched_dets


def compute_box_metrics(boxes):
    """Compute movement metrics from a sequence of bounding boxes."""
    if not boxes or len(boxes) < 2:
        return {
            "total_movement_px": 0,
            "avg_step_px": 0,
            "movement_consistency": 0,
            "max_speed_px": 0,
        }

    boxes_arr = np.array(boxes, dtype=np.float64)
    centers = np.array([
        [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        for b in boxes_arr
    ])

    # Compute step distances
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    total_movement = float(np.sum(steps))
    avg_step = float(np.mean(steps))
    max_speed = float(np.max(steps))

    # Consistency: coefficient of variation (lower = more consistent)
    if avg_step > 0:
        consistency = float(1 - min(np.std(steps) / avg_step, 1.0))
    else:
        consistency = 0.0

    return {
        "total_movement_px": round(total_movement, 1),
        "avg_step_px": round(avg_step, 2),
        "movement_consistency": round(consistency, 3),
        "max_speed_px": round(max_speed, 2),
        "frame_count": len(boxes),
    }
