"""
Visualization utilities for drawing skeletons, boxes, and court overlays.
"""
import cv2
import numpy as np


SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, keypoints, keypoint_scores, score_thresh=0.25, 
                  color=(0, 220, 255), thickness=2):
    """
    Draw COCO skeleton on frame.
    keypoints: (17, 2) array
    keypoint_scores: (17,) array
    """
    keypoints = np.array(keypoints, dtype=np.float64)
    keypoint_scores = np.array(keypoint_scores, dtype=np.float64)

    if keypoints.shape != (17, 2):
        return frame

    h, w = frame.shape[:2]

    # Draw edges
    for edge in SKELETON_EDGES:
        i, j = edge
        if keypoint_scores[i] >= score_thresh and keypoint_scores[j] >= score_thresh:
            pt1 = tuple(map(int, keypoints[i]))
            pt2 = tuple(map(int, keypoints[j]))
            # Clip to frame bounds
            pt1 = (max(0, min(pt1[0], w-1)), max(0, min(pt1[1], h-1)))
            pt2 = (max(0, min(pt2[0], w-1)), max(0, min(pt2[1], h-1)))
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for i in range(17):
        if keypoint_scores[i] >= score_thresh:
            pt = tuple(map(int, keypoints[i]))
            pt = (max(0, min(pt[0], w-1)), max(0, min(pt[1], h-1)))
            cv2.circle(frame, pt, 3, color, -1)

    return frame


def draw_box(frame, box, label=None, color=(0, 255, 0), thickness=2):
    """Draw bounding box with optional label."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), font, scale, (0, 0, 0), thickness)

    return frame


def draw_trajectory(frame, points, color=(0, 255, 0), thickness=2):
    """Draw a trajectory path from a list of points."""
    if len(points) < 2:
        return frame

    for i in range(1, len(points)):
        pt1 = tuple(map(int, points[i-1]))
        pt2 = tuple(map(int, points[i]))
        cv2.line(frame, pt1, pt2, color, thickness)

    return frame


def draw_action_label(frame, box, action, conf, color=(255, 0, 128)):
    """Draw action classification label above box."""
    x1, y1, x2, y2 = map(int, box)
    label = f"{action} ({conf:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

    # Background
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), font, scale, (255, 255, 255), thickness)

    return frame
