"""
Player analysis utilities with safe array handling for NumPy 2.x compatibility.
"""
import numpy as np


def compute_pose_features_from_keypoints(keypoints, keypoint_scores, score_thresh=0.3):
    """
    Compute posture, balance, stance, and height ratio from 17 COCO keypoints.
    keypoints: (17, 2) array
    keypoint_scores: (17,) array
    """
    # Validate input shape
    keypoints = np.array(keypoints, dtype=np.float64)
    keypoint_scores = np.array(keypoint_scores, dtype=np.float64)

    if keypoints.shape != (17, 2):
        # Pad or reshape to expected size
        padded = np.zeros((17, 2), dtype=np.float64)
        n = min(keypoints.shape[0], 17)
        padded[:n] = keypoints[:n]
        keypoints = padded

    if keypoint_scores.shape[0] != 17:
        padded_scores = np.zeros(17, dtype=np.float64)
        n = min(keypoint_scores.shape[0], 17)
        padded_scores[:n] = keypoint_scores[:n]
        keypoint_scores = padded_scores

    # COCO keypoint indices
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    def valid(idx):
        return keypoint_scores[idx] >= score_thresh

    def pt(idx):
        return keypoints[idx]

    # Posture: upright vs bent
    posture = "unknown"
    if valid(L_SHOULDER) and valid(L_HIP) and valid(L_ANKLE):
        torso_vec = pt(L_HIP) - pt(L_SHOULDER)
        leg_vec = pt(L_ANKLE) - pt(L_HIP)
        torso_leg_angle = _angle_between(torso_vec, leg_vec)
        if torso_leg_angle > 150:
            posture = "upright"
        elif torso_leg_angle > 120:
            posture = "slight_bend"
        else:
            posture = "bent"
    elif valid(R_SHOULDER) and valid(R_HIP) and valid(R_ANKLE):
        torso_vec = pt(R_HIP) - pt(R_SHOULDER)
        leg_vec = pt(R_ANKLE) - pt(R_HIP)
        torso_leg_angle = _angle_between(torso_vec, leg_vec)
        if torso_leg_angle > 150:
            posture = "upright"
        elif torso_leg_angle > 120:
            posture = "slight_bend"
        else:
            posture = "bent"

    # Balance: centered weight distribution
    balance = "unknown"
    if valid(L_ANKLE) and valid(R_ANKLE) and valid(NOSE):
        mid_feet = (pt(L_ANKLE) + pt(R_ANKLE)) / 2
        nose_x = pt(NOSE)[0]
        feet_width = abs(pt(R_ANKLE)[0] - pt(L_ANKLE)[0])
        offset = abs(nose_x - mid_feet[0])
        if feet_width > 0:
            if offset / feet_width < 0.15:
                balance = "centered"
            elif offset / feet_width < 0.35:
                balance = "slight_lean"
            else:
                balance = "off_balance"

    # Stance: wide vs narrow
    stance = "unknown"
    if valid(L_ANKLE) and valid(R_ANKLE) and valid(L_SHOULDER) and valid(R_SHOULDER):
        feet_width = abs(pt(R_ANKLE)[0] - pt(L_ANKLE)[0])
        shoulder_width = abs(pt(R_SHOULDER)[0] - pt(L_SHOULDER)[0])
        if shoulder_width > 0:
            ratio = feet_width / shoulder_width
            if ratio > 1.8:
                stance = "wide"
            elif ratio > 1.2:
                stance = "medium"
            else:
                stance = "narrow"

    # Height ratio
    height_ratio = 0.0
    if valid(NOSE) and valid(L_ANKLE):
        height = abs(pt(NOSE)[1] - pt(L_ANKLE)[1])
        # Normalize by a reference (roughly 200px for full height in 720p)
        height_ratio = min(height / 300.0, 1.0)

    return {
        "posture": posture,
        "balance": balance,
        "stance": stance,
        "height_ratio": round(float(height_ratio), 3),
        "keypoints": keypoints.tolist(),
        "keypoint_scores": keypoint_scores.tolist(),
    }


def _angle_between(v1, v2):
    """Compute angle between two 2D vectors in degrees."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def find_best_pose_match(box, pose_results, min_score=0.2, min_iou=0.1):
    """
    Find the pose detection that best matches the given bounding box.
    Returns index into pose_results or None.
    """
    if not pose_results:
        return None

    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return None

    best_idx = None
    best_score = -1

    for idx, pose in enumerate(pose_results):
        if pose.get("score", 0) < min_score:
            continue

        px1, py1, px2, py2 = pose["box"]
        pose_area = (px2 - px1) * (py2 - py1)
        if pose_area <= 0:
            continue

        # Compute IoU
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)

        if ix2 <= ix1 or iy2 <= iy1:
            iou = 0.0
        else:
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = box_area + pose_area - inter
            iou = inter / (union + 1e-6)

        # Center distance score
        bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        dist = np.sqrt((bcx - pcx)**2 + (bcy - pcy)**2)
        dist_score = max(0, 1 - dist / 200)  # Normalize by 200px

        score = iou * 2 + dist_score + pose.get("score", 0) * 0.5

        if iou >= min_iou and score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def summarize_pose_metrics(poses):
    """
    Summarize a list of pose feature dicts.
    SAFE: handles empty lists and inconsistent shapes.
    """
    if not poses:
        return {
            "upright_ratio": 0.0,
            "centred_ratio": 0.0,
            "athletic_ready_ratio": 0.0,
            "avg_height_ratio": 0.0,
        }

    # Filter out invalid entries
    valid_poses = []
    for p in poses:
        if isinstance(p, dict) and "posture" in p:
            valid_poses.append(p)

    if not valid_poses:
        return {
            "upright_ratio": 0.0,
            "centred_ratio": 0.0,
            "athletic_ready_ratio": 0.0,
            "avg_height_ratio": 0.0,
        }

    total = len(valid_poses)
    upright = sum(1 for p in valid_poses if p.get("posture") == "upright")
    centred = sum(1 for p in valid_poses if p.get("balance") == "centered")
    athletic = sum(1 for p in valid_poses if p.get("stance") in ["wide", "medium"])
    heights = [p.get("height_ratio", 0) for p in valid_poses if p.get("height_ratio") is not None]

    return {
        "upright_ratio": round(upright / total, 3) if total > 0 else 0.0,
        "centred_ratio": round(centred / total, 3) if total > 0 else 0.0,
        "athletic_ready_ratio": round(athletic / total, 3) if total > 0 else 0.0,
        "avg_height_ratio": round(float(np.mean(heights)), 3) if heights else 0.0,
    }


def build_rule_based_player_feedback(player, posture, balance, stance, height_ratio, pose_metrics, box_metrics):
    """Generate rule-based coaching feedback."""
    strengths = []
    weaknesses = []
    drills = []
    improvements = []

    if pose_metrics.get("upright_ratio", 0) > 0.6:
        strengths.append("Good upright posture maintained during rally")
    else:
        weaknesses.append("Posture tends to break down under pressure")
        drills.append("Wall sits: 3x30s to build core stability")
        improvements.append("Better posture will improve shot power by 15-20%")

    if pose_metrics.get("centred_ratio", 0) > 0.5:
        strengths.append("Solid balance and weight distribution")
    else:
        weaknesses.append("Weight shifts too far during movement")
        drills.append("Single-leg balance drills: 3x30s each leg")
        improvements.append("Improved balance reduces recovery time by 0.3s per shot")

    if stance in ["wide", "medium"]:
        strengths.append("Athletic ready stance detected")
    else:
        weaknesses.append("Stance too narrow — limited power base")
        drills.append("Shadow footwork with emphasis on wide base")
        improvements.append("Wider stance increases smash power by 10%")

    total_movement = box_metrics.get("total_movement_px", 0)
    if total_movement > 500:
        strengths.append(f"High court coverage: {total_movement:.0f}px total movement")
    else:
        weaknesses.append("Limited court coverage detected")
        drills.append("Four-corner footwork drill: 10 reps each corner")
        improvements.append("Better coverage = more reachable shots")

    if not strengths:
        strengths.append("Player tracking successful — baseline established")
    if not weaknesses:
        weaknesses.append("No major technical flaws detected in sample")
    if not drills:
        drills.append("Continue current training regimen")
    if not improvements:
        improvements.append("Maintain current form")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "drills": drills,
        "improvements": improvements,
        "metrics": box_metrics,
    }


def build_player_analysis_report(player, posture, balance, stance, height_ratio, pose_metrics, box_metrics, feedback):
    """Build a detailed text report."""
    lines = [
        f"PLAYER ANALYSIS: {player}",
        "=" * 50,
        "",
        f"Dominant Posture: {posture}",
        f"Dominant Balance: {balance}",
        f"Dominant Stance: {stance}",
        f"Average Height Ratio: {height_ratio:.3f}",
        "",
        "POSE METRICS:",
        f"  Upright Ratio: {pose_metrics.get('upright_ratio', 0):.1%}",
        f"  Centered Ratio: {pose_metrics.get('centred_ratio', 0):.1%}",
        f"  Athletic Ready: {pose_metrics.get('athletic_ready_ratio', 0):.1%}",
        "",
        "MOVEMENT METRICS:",
    ]

    for k, v in box_metrics.items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        "FEEDBACK SUMMARY:",
        f"  Strengths: {len(feedback.get('strengths', []))} identified",
        f"  Weaknesses: {len(feedback.get('weaknesses', []))} identified",
        f"  Recommended Drills: {len(feedback.get('drills', []))}",
    ])

    return "\n".join(lines)


def format_percent(val):
    return f"{val:.1%}"


def format_optional_metric(val, suffix=""):
    if val is None:
        return "N/A"
    return f"{val:.2f}{suffix}"
