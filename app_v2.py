"""
OmniCourt v4 - Single player, click-to-mark court corners, skeleton video overlay.
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import base64
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from flask_cors import CORS
from celery.result import AsyncResult
from dotenv import load_dotenv

from shuttlecock_detector import detect_shuttlecocks
from court_homography import CourtHomography
from pose_tracker import MultiPoseTracker
from llm_feedback import (
    generate_structured_feedback, generate_chat_reply,
    is_valid_coaching_feedback, has_ai_analyzer, get_ai_model_name,
)
from tasks import celery_app, process_video_analysis, cleanup_old_files
from data_processing import (
    apply_nms, center_distance, compute_box_metrics, filter_close_detections,
)
from player_analysis import (
    build_player_analysis_report, build_rule_based_player_feedback,
    compute_pose_features_from_keypoints, find_best_pose_match,
    summarize_pose_metrics,
)
from visualization_utils import draw_skeleton

load_dotenv()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.errorhandler(Exception)
def handle_error(error):
    import traceback
    traceback.print_exc()
    return jsonify({"success": False, "error": str(error)}), 500

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

BADMINTON_CLASSES = {
    0: "player", 1: "player_smash", 2: "player_clear", 3: "player_drop",
    4: "player_net_shot", 5: "player_lift", 6: "player_push",
    7: "player_drive", 8: "player_cross", 9: "player_net_kill"
}

yolo_model = None
pose_model = None
pose_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_video_path = None
detected_players = {}
player_analysis_context = {}
last_detected_frame = None
court_homography = None
annotated_video_url = None

# ============================================================
# MODELS
# ============================================================

def init_yolo():
    global yolo_model
    if yolo_model is None:
        try:
            model_path = BASE_DIR / "models" / "badminton_actions_v1.pt"
            if not model_path.exists():
                model_path = "yolov8n.pt"
            yolo_model = YOLO(str(model_path))
            yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"[ERROR] YOLO load failed: {e}")
            yolo_model = False

def init_pose_model():
    global pose_model, pose_model_device
    if pose_model is None:
        try:
            pose_model = YOLO("yolov8s-pose.pt")
            pose_model.to(pose_model_device)
        except Exception as e:
            print(f"[WARNING] Pose model failed: {e}")
            pose_model = False
    return pose_model

def run_skeleton_detector(frame):
    init_pose_model()
    if not pose_model or pose_model is False:
        return None
    results = pose_model(frame, conf=0.25, verbose=False)
    if not results or len(results[0].keypoints) == 0:
        return None
    kpts = results[0].keypoints.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    return [{
        "keypoints": kpts[i][:, :2],
        "keypoint_scores": kpts[i][:, 2],
        "box": boxes[i],
        "score": float(scores[i])
    } for i in range(len(kpts))]

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)

@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")

@app.route("/coach")
def coach_page():
    return render_template("chat.html")

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

# ============================================================
# STAGE 1: Detect ALL players + draw skeletons on ALL
# ============================================================

@app.route("/detect-players", methods=["POST"])
def detect_players():
    global current_video_path, detected_players, last_detected_frame, court_homography

    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    save_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(str(save_path))
    current_video_path = str(save_path)
    detected_players = {}
    court_homography = None
    annotated_video_url = None

    init_yolo()
    if not yolo_model:
        return jsonify({"error": "YOLO failed to load"}), 500

    try:
        cap = cv2.VideoCapture(str(save_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        if duration_sec > 120:
            cap.release()
            return jsonify({"error": f"Video too long: {duration_sec:.1f}s (max 120s)"}), 400

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"error": "Failed to read video"}), 400

        h, w = frame.shape[:2]
        orig_w, orig_h = w, h
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        # Detect people
        results = yolo_model(frame, conf=0.25, verbose=False)
        detections = []
        min_box_width = max(30, int(w * 0.02))
        min_box_height = max(40, int(h * 0.04))

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            if cls <= 9 and conf >= 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = x2 - x1, y2 - y1
                aspect = bh / (bw + 1e-6)
                if bw < min_box_width or bh < min_box_height:
                    continue
                if aspect < 0.4 or aspect > 6.0:
                    continue
                detections.append((x1, y1, x2, y2))

        detections = apply_nms(detections, overlap_threshold=0.35)
        detections = filter_close_detections(detections, min_distance_px=25)
        if len(detections) > 15:
            detections = sorted(detections, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:15]

        detected_players = {}
        for idx, (x1, y1, x2, y2) in enumerate(detections, start=1):
            detected_players[f"Player{idx}"] = {
                "initial_box": (x1, y1, x2, y2),
                "frame_size": (w, h),
                "original_frame_size": (orig_w, orig_h),
                "boxes": [], "poses": [], "metrics": {},
            }

        # Draw preview WITH SKELETONS ON ALL DETECTED PEOPLE
        frame_with_boxes = frame.copy()
        pose_results = run_skeleton_detector(frame)

        # Match each detection to nearest pose by center distance
        used_poses = set()
        for idx, (x1, y1, x2, y2) in enumerate(detections, start=1):
            label = f"Player{idx}"

            # Find closest pose by center distance (not just IoU)
            best_idx = None
            best_dist = float('inf')
            det_cx = (x1 + x2) / 2
            det_cy = (y1 + y2) / 2

            for p_idx, pose in enumerate(pose_results or []):
                if p_idx in used_poses:
                    continue
                px1, py1, px2, py2 = pose["box"]
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                dist = np.sqrt((det_cx - pcx)**2 + (det_cy - pcy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = p_idx

            # Draw skeleton if found (even if far, as long as it's the closest)
            if best_idx is not None and best_dist < 200:  # 200px threshold
                used_poses.add(best_idx)
                draw_skeleton(frame_with_boxes, pose_results[best_idx]["keypoints"],
                             pose_results[best_idx]["keypoint_scores"], score_thresh=0.25, color=(0, 220, 255))

            # Draw box and label
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, label, (x1, max(y1 - 10, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame_with_boxes)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        last_detected_frame = frame_b64

        return jsonify({
            "frame": frame_b64,
            "labels": list(detected_players.keys()),
            "count": len(detected_players),
            "frame_width": w,
            "frame_height": h,
            "duration_sec": round(duration_sec, 1),
            "message": f"Detected {len(detected_players)} players. Select 1 to track."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

# ============================================================
# COURT CORNERS: Click to mark
# ============================================================

@app.route("/set-court-corners", methods=["POST"])
def set_court_corners():
    global court_homography
    data = request.json or {}
    corners = data.get("corners")
    if not corners or len(corners) < 3 or len(corners) > 4:
        return jsonify({"error": "Provide 3 or 4 corner points [x, y]"}), 400

    court_homography = CourtHomography()
    court_homography.set_manual_corners(np.array(corners, dtype=np.float32))

    # Camera angle info from the homography class
    angle_info = ""
    if court_homography.camera_angle == "high_angle":
        angle_info = "Camera at elevated angle. Height measurements may be compressed."
    elif court_homography.camera_angle == "low_angle":
        angle_info = "Camera at low angle. Depth perception exaggerated."
    elif court_homography.camera_angle == "side_angle":
        angle_info = "Camera from side angle. Lateral measurements adjusted."
    else:
        angle_info = "Camera angle is neutral. Good for analysis."

    return jsonify({
        "success": True,
        "message": "Court corners set. " + angle_info,
        "camera_angle_info": angle_info,
        "computed_corner": court_homography.computed_corner,
    })

# ============================================================
# STAGE 2: Track 1 selected player
# ============================================================

@app.route("/track-player", methods=["POST"])
def track_player():
    global current_video_path, detected_players
    data = request.json or {}
    player_label = data.get("player", "")

    if not player_label or player_label not in detected_players:
        return jsonify({"error": "Select a player first"}), 400
    if not current_video_path:
        return jsonify({"error": "No video loaded"}), 400

    initial_boxes = {label: detected_players[label]["initial_box"] for label in detected_players}
    frame_size = detected_players[player_label].get("frame_size")

    task = process_video_analysis.delay(current_video_path, [player_label], initial_boxes, frame_size)
    return jsonify({"success": True, "task_id": task.id, "message": f"Tracking {player_label}..."})

@app.route("/task-status/<task_id>")
def task_status(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.state == "PENDING":
        return jsonify({"state": "PENDING", "progress": 0})
    elif task.state == "PROGRESS":
        return jsonify({
            "state": task.state,
            "progress": task.info.get("progress", 0) if task.info else 0,
            "step": task.info.get("step", "") if task.info else "",
        })
    elif task.state == "SUCCESS":
        result = task.result
        if isinstance(result, dict) and "error" in result:
            return jsonify({"state": "FAILURE", "error": result["error"]})

        for player, data in result.get("player_data", {}).items():
            if player in detected_players:
                detected_players[player]["boxes"] = data.get("boxes", [])
                detected_players[player]["poses"] = data.get("poses", [])
                detected_players[player]["metrics"] = data.get("box_metrics", {})
                detected_players[player]["pose_metrics"] = data.get("pose_metrics", {})
                detected_players[player]["court_positions"] = data.get("court_positions", [])
                detected_players[player]["real_movement"] = data.get("real_movement")

        return jsonify({
            "state": "SUCCESS",
            "frames_tracked": result.get("total_frames_processed", 0),
            "court_calibrated": result.get("court_calibrated", False),
        })
    else:
        error_msg = "Task failed"
        try:
            if task.info is not None:
                error_msg = str(task.info)
        except Exception:
            error_msg = "Task failed with unserializable error"
        return jsonify({"state": task.state, "error": error_msg})

# ============================================================
# STAGE 3: Generate skeleton video overlay
# ============================================================

@app.route("/generate-skeleton-video", methods=["POST"])
def generate_skeleton_video():
    global current_video_path, detected_players, annotated_video_url

    data = request.json or {}
    player_label = data.get("player", "")

    if not player_label or player_label not in detected_players:
        return jsonify({"error": "Select a player first"}), 400
    if not current_video_path:
        return jsonify({"error": "No video"}), 400

    output_name = f"skeleton_{player_label}_{int(time.time())}.mp4"
    output_path = UPLOAD_FOLDER / output_name

    try:
        frames = create_skeleton_video(current_video_path, player_label, output_path)
        annotated_video_url = f"/uploads/{output_name}"
        return jsonify({
            "success": True,
            "video_url": annotated_video_url,
            "frames_processed": frames,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def create_skeleton_video(video_path, player_label, output_path, max_seconds=120):
    """Generate video with skeleton overlay on the selected player."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, int(max_seconds * fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output at reasonable size
    out_w = min(w, 960)
    out_h = int(h * (out_w / w))

    # Try multiple codecs
    for codec in ["mp4v", "avc1", "XVID", "MJPG"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        if writer.isOpened():
            break
    else:
        cap.release()
        raise RuntimeError("Could not open video writer with any codec")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pose model
    init_pose_model()
    if not pose_model or pose_model is False:
        cap.release()
        writer.release()
        raise RuntimeError("Pose model not available")

    # Get player info
    init_box = detected_players[player_label]["initial_box"]
    frame_size = detected_players[player_label]["frame_size"]
    sx = out_w / frame_size[0]
    sy = out_h / frame_size[1]

    # Scale initial box to output size
    prev_box = (int(init_box[0]*sx), int(init_box[1]*sy), int(init_box[2]*sx), int(init_box[3]*sy))

    tracker = MultiPoseTracker(max_age=8)
    processed = 0

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (out_w, out_h))
        annotated = resized.copy()

        # Detect poses
        pose_dets = run_skeleton_detector(resized) or []

        # Track
        tracker_dets = [{"keypoints": p["keypoints"], "box": p["box"], "score": p["score"]} for p in pose_dets]
        confirmed = tracker.update(tracker_dets)

        # Find track closest to previous box
        target_track = None
        best_dist = float('inf')
        for track in confirmed:
            dist = center_distance(prev_box, track.box)
            if dist < best_dist:
                best_dist = dist
                target_track = track

        if target_track:
            prev_box = tuple(map(int, target_track.box))
            x1, y1, x2, y2 = prev_box

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 150, 255), 3)
            cv2.putText(annotated, player_label, (x1, max(y1-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

            # Draw smoothed skeleton
            smoothed = target_track.get_smoothed_keypoints()
            draw_skeleton(annotated, smoothed, target_track.keypoint_scores,
                         score_thresh=0.3, color=(0, 150, 255), thickness=2)

            # Draw action label if available
            # Detect action on this frame for the player
            action_results = yolo_model(resized, conf=0.25, verbose=False)
            for box in action_results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                if center_distance(prev_box, (bx1, by1, bx2, by2)) < 100 and cls <= 9:
                    action_name = BADMINTON_CLASSES.get(cls, "player").replace("player_", "").upper()
                    label = f"{action_name}"
                    cv2.putText(annotated, label, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(annotated)
        processed += 1

    cap.release()
    writer.release()
    return processed

# ============================================================
# STAGE 4: Analyze player
# ============================================================

@app.route("/analyze-player", methods=["POST"])
def analyze_player():
    global player_analysis_context
    data = request.json or {}
    player = data.get("player", "")

    if not player or player not in detected_players:
        return jsonify({"error": "Player not found"}), 400

    try:
        poses = detected_players[player].get("poses", [])
        boxes = detected_players[player].get("boxes", [])
        real_movement = detected_players[player].get("real_movement")

        if not poses:
            return jsonify({"error": "No pose data"}), 400

        postures = [p.get("posture") for p in poses if p.get("posture")]
        balances = [p.get("balance") for p in poses if p.get("balance")]
        stances = [p.get("stance") for p in poses if p.get("stance")]
        heights = [p.get("height_ratio") for p in poses if p.get("height_ratio")]

        dominant_posture = max(set(postures), key=postures.count) if postures else "unknown"
        dominant_balance = max(set(balances), key=balances.count) if balances else "unknown"
        dominant_stance = max(set(stances), key=stances.count) if stances else "unknown"
        avg_height = float(np.mean(heights)) if heights else 0.0

        pose_metrics = summarize_pose_metrics(poses)
        box_metrics = compute_box_metrics(boxes)

        if real_movement:
            box_metrics["total_movement_meters"] = real_movement.get("total_meters", 0)
            box_metrics["avg_step_meters"] = real_movement.get("avg_step_meters", 0)

        total_movement = box_metrics.get("total_movement_px", 0)

        if total_movement < 50:
            feedback_data = {
                "strengths": ["Player detected and tracked successfully."],
                "weaknesses": ["Limited movement detected in video sample."],
                "drills": ["Record a longer rally with more court coverage."],
                "improvements": ["More active footage needed for deeper analysis."],
            }
            ai_commentary = "Limited motion in sample. Record an active rally for full analysis."
        else:
            feedback_data = build_rule_based_player_feedback(
                player, dominant_posture, dominant_balance,
                dominant_stance, avg_height, pose_metrics, box_metrics
            )

            # Camera angle context
            angle_note = ""
            if court_homography and court_homography.is_calibrated:
                angle_note = "Court perspective calibrated. Metrics adjusted for camera angle. "

            if has_ai_analyzer():
                focus_issues = []
                if pose_metrics.get("centred_ratio", 0) < 0.5:
                    focus_issues.append("center balance")
                if pose_metrics.get("upright_ratio", 0) < 0.5:
                    focus_issues.append("posture")
                if box_metrics.get("movement_consistency", 0) < 0.6:
                    focus_issues.append("footwork consistency")

                focus_str = ", ".join(focus_issues[:3]) if focus_issues else "overall technique"

                prompt = f"""Analyze {player}'s badminton technique based on these metrics:
Posture: {dominant_posture}, Balance: {dominant_balance}, Stance: {dominant_stance}
Upright ratio: {pose_metrics.get('upright_ratio', 0):.1%}
Centered ratio: {pose_metrics.get('centred_ratio', 0):.1%}
Athletic ready: {pose_metrics.get('athletic_ready_ratio', 0):.1%}
Movement consistency: {box_metrics.get('movement_consistency', 0):.2f}
Avg step: {box_metrics.get('avg_step_px', 0):.1f}px
Total movement: {box_metrics.get('total_movement_px', 0):.0f}px
{angle_note}
Focus areas: {focus_str}

Provide structured coaching feedback with specific drills. Use plain text, no markdown."""

                ai_commentary = generate_structured_feedback(prompt, max_new_tokens=500)
                if not ai_commentary:
                    ai_commentary = "AI analysis unavailable. Using rule-based feedback."
            else:
                ai_commentary = "DeepSeek API key not configured. Add DEEPSEEK_API_KEY to .env for AI coaching."

        player_analysis_context[player] = {
            "profile": {"posture": dominant_posture, "balance": dominant_balance, "stance": dominant_stance, "height_ratio": avg_height},
            "pose_metrics": pose_metrics,
            "box_metrics": box_metrics,
            "feedback": feedback_data,
            "ai_commentary": ai_commentary,
            "ai_model": get_ai_model_name(),
        }

        result = f"""COACHING ANALYSIS: {player}
{'=' * 50}

TECHNICAL PROFILE:
  Posture: {dominant_posture}
  Balance: {dominant_balance}
  Stance: {dominant_stance}
  Height Ratio: {round(avg_height, 2)}

POSE METRICS:
  Upright Ratio: {pose_metrics.get('upright_ratio', 0):.1%}
  Centered Ratio: {pose_metrics.get('centred_ratio', 0):.1%}
  Athletic Ready: {pose_metrics.get('athletic_ready_ratio', 0):.1%}

MOVEMENT METRICS:
  Total Movement: {box_metrics.get('total_movement_px', 0):.0f}px
  Avg Step: {box_metrics.get('avg_step_px', 0):.1f}px
  Consistency: {box_metrics.get('movement_consistency', 0):.2f}

FEEDBACK:
{chr(10).join('- ' + s for s in feedback_data.get('strengths', []))}

WEAKNESSES:
{chr(10).join('- ' + w for w in feedback_data.get('weaknesses', []))}

DRILLS:
{chr(10).join('- ' + d for d in feedback_data.get('drills', []))}

{'=' * 50}
AI COACH COMMENTARY:
{ai_commentary}
{'=' * 50}"""

        return jsonify({
            "analysis": result,
            "analysis_data": player_analysis_context[player],
            "ai_commentary": ai_commentary,
            "real_world_metrics": real_movement,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# CHAT
# ============================================================

@app.route("/player-chat", methods=["POST"])
def player_chat():
    global player_analysis_context
    try:
        data = request.get_json(force=True)
        player = data.get("player", "").strip()
        question = data.get("question", "").strip()

        if not player or not question:
            return jsonify({"success": False, "error": "Player and question required"}), 400
        if player not in player_analysis_context:
            return jsonify({"success": False, "error": "Run analysis first"}), 400

        ctx = player_analysis_context[player]

        reply = None
        if has_ai_analyzer():
            reply = generate_chat_reply(player, question, ctx)

        if not reply:
            reply = f"[Fallback] Based on {player}'s data: Posture={ctx['profile']['posture']}, Balance={ctx['profile']['balance']}. Key drill: {ctx['feedback']['drills'][0] if ctx['feedback']['drills'] else 'General footwork practice'}."

        return jsonify({
            "success": True, "player": player, "question": question,
            "reply": reply, "ai_model": get_ai_model_name(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================
# UTILS
# ============================================================

@app.route("/cleanup", methods=["POST"])
def trigger_cleanup():
    task = cleanup_old_files.delay(str(UPLOAD_FOLDER), max_age_hours=24)
    return jsonify({"success": True, "task_id": task.id})

@app.route("/debug-context", methods=["GET"])
def debug_context():
    return jsonify({
        "detected_players": list(detected_players.keys()),
        "analyzed_players": list(player_analysis_context.keys()),
        "court_calibrated": court_homography.is_calibrated if court_homography else False,
        "ai_available": has_ai_analyzer(),
        "ai_model": get_ai_model_name(),
        "video_path": current_video_path,
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
