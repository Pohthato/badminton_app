"""
OmniCourt Professional - Advanced Badminton Analysis Platform
Integrates professional court detection, live analysis, and deep player analytics
"""
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import base64
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional

# Import new professional modules
from professional_court_detector import ProfessionalCourtHomography
from advanced_player_analyzer import AdvancedPlayerAnalyzer
from live_video_analyzer import LiveVideoAnalyzer
from shuttlecock_detector import detect_shuttlecocks, detect_shuttle_keypoints, init_shuttle_detector
from pose_tracker import MultiPoseTracker
from visualization_utils import draw_skeleton
from data_processing import apply_nms, center_distance, filter_close_detections
from world_class_analyzer import WorldClassAnalyzer

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Error: {error}", exc_info=True)
    return jsonify({"success": False, "error": str(error)}), 500

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB

BADMINTON_CLASSES = {
    0: "player", 1: "player_smash", 2: "player_clear", 3: "player_drop",
    4: "player_net_shot", 5: "player_lift", 6: "player_push",
    7: "player_drive", 8: "player_cross", 9: "player_net_kill"
}

# Global state
yolo_model = None
pose_model = None
pose_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_video_path = None
court_homography = ProfessionalCourtHomography()
detected_players = {}
player_analyzers = {}  # Per-player analyzers
live_analyzer = None
world_class_analyzer = None
world_class_results = {}


def init_yolo():
    """Initialize YOLO action detection model."""
    global yolo_model
    if yolo_model is None:
        try:
            model_path = BASE_DIR / "models" / "badminton_actions_v1.pt"
            if model_path.exists():
                yolo_model = YOLO(str(model_path))
            else:
                logger.warning(f"Model not found at {model_path}, using default")
                yolo_model = YOLO("yolov8n.pt")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            yolo_model.to(device)
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            yolo_model = False
    
    return yolo_model


def init_shuttle():
    """Initialize world-class shuttlecock detector with best.pt."""
    return init_shuttle_detector()


def init_pose_model():
    """Initialize YOLOv8 pose detection model."""
    global pose_model
    if pose_model is None:
        try:
            pose_model = YOLO("yolov8s-pose.pt")
            pose_model.to(pose_model_device)
        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
            pose_model = False
    
    return pose_model


def detect_skeleton(frame: np.ndarray) -> Optional[List[Dict]]:
    """Run pose detection on frame."""
    init_pose_model()
    if not pose_model or pose_model is False:
        return None
    
    try:
        results = pose_model(frame, conf=0.25, verbose=False)
        if not results or len(results[0].keypoints) == 0:
            return None
        
        kpts = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        detections = []
        for i in range(len(kpts)):
            detections.append({
                "keypoints": kpts[i][:, :2],
                "keypoint_scores": kpts[i][:, 2],
                "box": boxes[i],
                "score": float(scores[i])
            })
        
        return detections
    except Exception as e:
        logger.error(f"Pose detection failed: {e}")
        return None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    """Serve modern frontend."""
    return render_template("index_modern.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/analysis")
def analysis_page():
    """Analysis dashboard."""
    return render_template("analysis.html")


@app.route("/coach")
def coach_page():
    """AI coaching feedback."""
    return render_template("chat.html")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ============================================================
# STAGE 1: Upload and Detect Players
# ============================================================

@app.route("/detect-players", methods=["POST"])
def detect_players():
    """
    Upload video and detect all players.
    Returns: frame with detected skeleton overlays, player count
    """
    global current_video_path, detected_players, court_homography, live_analyzer
    
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files["video"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Save video
    filename = Path(file.filename).stem + "_" + str(int(time.time())) + Path(file.filename).suffix
    save_path = UPLOAD_FOLDER / filename
    file.save(str(save_path))
    current_video_path = str(save_path)
    
    # Reset state
    detected_players = {}
    court_homography = ProfessionalCourtHomography()
    player_analyzers = {}
    
    init_yolo()
    init_pose_model()
    
    try:
        # Read first frame
        cap = cv2.VideoCapture(str(save_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        # Limit duration
        if duration_sec > 300:  # 5 minute limit
            cap.release()
            return jsonify({"error": f"Video too long: {duration_sec:.1f}s (max 300s)"}), 400
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"error": "Cannot read video"}), 400
        
        h, w = frame.shape[:2]
        
        # Resize if needed
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
        
        # Detect court automatically
        court_detected = court_homography.detect_court_automatic(frame)
        if court_detected:
            logger.info(f"Court detected using: {court_homography.detection_method}")
        else:
            logger.info("Court auto-detection failed, manual calibration needed")
        
        # Detect players
        pose_results = detect_skeleton(frame) or []
        
        if not pose_results:
            return jsonify({"error": "No players detected"}), 400
        
        # Create player entries
        frame_annotated = frame.copy()
        for idx, pose in enumerate(pose_results):
            player_id = f"player_{idx}"
            detected_players[player_id] = {
                "index": idx,
                "box": pose["box"].tolist(),
                "keypoints": pose["keypoints"].tolist(),
                "keypoint_scores": pose["keypoint_scores"].tolist(),
                "confidence": pose["score"]
            }
            
            # Draw skeleton
            draw_skeleton(frame_annotated, pose["keypoints"], pose["keypoint_scores"],
                         score_thresh=0.3, thickness=2)
            
            # Draw player label
            x1, y1, x2, y2 = pose["box"]
            cv2.rectangle(frame_annotated, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            cv2.putText(frame_annotated, f"Player {idx+1}", (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame_annotated)
        frame_b64 = base64.b64encode(buffer).tobytes().decode('utf-8')
        
        # Create live analyzer
        live_analyzer = LiveVideoAnalyzer(
            pose_model=pose_model,
            yolo_model=yolo_model,
            shuttle_detector=None,  # TODO: Initialize if model available
            court_homography=court_homography
        )
        
        return jsonify({
            "success": True,
            "video_path": str(save_path),
            "frame": frame_b64,
            "players": list(detected_players.keys()),
            "player_count": len(detected_players),
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "court_calibrated": court_homography.is_calibrated,
            "court_method": court_homography.detection_method,
            "message": f"Detected {len(detected_players)} players"
        })
    
    except Exception as e:
        logger.error(f"Error in detect_players: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# STAGE 2: Set Court Corners (Manual)
# ============================================================

@app.route("/calibrate-court", methods=["POST"])
def calibrate_court():
    """
    Manually set court corners via user clicks.
    Accepts 2, 3, or 4 corner points.
    """
    global court_homography
    
    data = request.json or {}
    corners = data.get("corners", [])
    
    if not corners or not (2 <= len(corners) <= 4):
        return jsonify({"error": "Provide 2-4 corner points [x, y]"}), 400
    
    try:
        success = court_homography.set_manual_corners(corners)
        
        if not success:
            return jsonify({"error": "Invalid court geometry"}), 400
        
        diagnostics = court_homography.get_court_diagnostics()
        
        return jsonify({
            "success": True,
            "message": "Court calibrated successfully",
            "diagnostics": diagnostics
        })
    
    except Exception as e:
        logger.error(f"Error calibrating court: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# STAGE 3: Live Analysis Stream
# ============================================================

@app.route("/analyze-live", methods=["POST"])
def analyze_live():
    """
    Stream live analysis for selected player.
    Returns frame-by-frame analysis as JSON stream.
    """
    global current_video_path, live_analyzer, player_analyzers
    
    data = request.json or {}
    player_id = data.get("player_id", 0)
    skip_frames = data.get("skip_frames", 0)  # Skip N frames for performance
    
    if not current_video_path or not Path(current_video_path).exists():
        return jsonify({"error": "No video loaded"}), 400
    
    if not live_analyzer:
        return jsonify({"error": "Live analyzer not initialized"}), 400
    
    # Create analyzer for this player if needed
    if player_id not in player_analyzers:
        player_analyzers[player_id] = AdvancedPlayerAnalyzer(
            court_homography=court_homography
        )
    
    player_analyzer = player_analyzers[player_id]
    
    def generate():
        """Generator for streaming analysis frames."""
        try:
            frame_gen = live_analyzer.process_video_stream(
                current_video_path,
                player_analyzer=player_analyzer,
                skip_frames=skip_frames
            )
            
            for frame_data in frame_gen:
                # Send as SSE (Server-Sent Events)
                yield f"data: {json.dumps(frame_data)}\n\n"
        
        except Exception as e:
            logger.error(f"Error in analyze_live: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return app.response_class(generate(), mimetype='text/event-stream')


# ============================================================
# STAGE 4: Analysis Report
# ============================================================

@app.route("/player-report/<player_id>", methods=["GET"])
def player_report(player_id: str):
    """
    Generate comprehensive analysis report for a player.
    """
    if int(player_id) not in player_analyzers:
        return jsonify({"error": "Player not analyzed"}), 404
    
    analyzer = player_analyzers[int(player_id)]
    report = analyzer.generate_session_report()
    
    if not report:
        return jsonify({"error": "No analysis data"}), 400
    
    return jsonify({
        "success": True,
        "player_id": player_id,
        "report": report
    })


# ============================================================
# COURT DIAGNOSTICS
# ============================================================

@app.route("/court-diagnostics", methods=["GET"])
def court_diagnostics():
    """Get current court calibration diagnostics."""
    if not court_homography:
        return jsonify({"error": "No court initialized"}), 400
    
    diagnostics = court_homography.get_court_diagnostics()
    return jsonify({
        "success": True,
        "diagnostics": diagnostics
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models": {
            "yolo": yolo_model is not False,
            "pose": pose_model is not False,
            "cuda_available": torch.cuda.is_available()
        }
    })


# ============================================================
# STAGE 2: First Frame with Skeleton Detection
# ============================================================

@app.route("/api/first-frame", methods=["POST"])
def get_first_frame():
    """Extract first frame with skeleton detection for player selection."""
    global current_video_path
    
    if current_video_path is None:
        return jsonify({"error": "No video loaded"}), 400
    
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Cannot read frame"}), 400
    
    init_pose_model()
    if not pose_model or pose_model is False:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jsonify({"success": True, "frame": base64.b64encode(buffer).decode('utf-8'),
                       "frame_width": frame.shape[1], "frame_height": frame.shape[0],
                       "players": [], "player_count": 0})
    
    try:
        results = pose_model(frame, conf=0.25, verbose=False)
        players = []
        if results and len(results) > 0:
            result = results[0]
            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None
                scores = result.boxes.conf.cpu().numpy() if result.boxes else None
                for i in range(len(kpts)):
                    keypoints = kpts[i][:, :2].tolist()
                    kpt_scores = kpts[i][:, 2].tolist()
                    box = boxes[i].tolist() if boxes is not None and i < len(boxes) else [0,0,0,0]
                    score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                    cx = float(np.mean([kp[0] for kp in keypoints if kp[0] > 0]) if any(kp[0] > 0 for kp in keypoints) else 0)
                    cy = float(np.mean([kp[1] for kp in keypoints if kp[1] > 0]) if any(kp[1] > 0 for kp in keypoints) else 0)
                    players.append({"id": i, "keypoints": keypoints, "keypoint_scores": kpt_scores,
                                   "box": box, "score": score, "center_x": cx, "center_y": cy})
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"success": True, "frame": frame_base64, "frame_width": frame.shape[1],
                       "frame_height": frame.shape[0], "players": players, "player_count": len(players)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# STAGE 3: Court Calibration
# ============================================================

@app.route("/api/set-court-corners", methods=["POST"])
def set_court_corners():
    """Set court corners and compute homography."""
    global court_homography
    
    data = request.get_json()
    if not data or "corners" not in data:
        return jsonify({"error": "No corners provided"}), 400
    
    corners = data["corners"]
    if len(corners) < 3:
        return jsonify({"error": "At least 3 corners required"}), 400
    
    try:
        src_points = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float32)
        
        court_length = 13.4
        court_width = 6.1
        
        if len(src_points) == 3:
            angles = []
            for i in range(3):
                others = [j for j in range(3) if j != i]
                v1 = src_points[others[0]] - src_points[i]
                v2 = src_points[others[1]] - src_points[i]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angles.append(np.arccos(np.clip(cos_a, -1, 1)))
            shared_idx = int(np.argmax(angles))
            shared = src_points[shared_idx]
            others = np.array([src_points[i] for i in range(3) if i != shared_idx])
            fourth = others[0] + others[1] - shared
            src_full = np.zeros((4, 2), dtype=np.float32)
            src_full[:3] = src_points
            src_full[3] = fourth
            src_points = src_full
        
        # Sort corners
        src_points = sort_corners(src_points)
        
        success = court_homography.set_manual_corners(src_points.tolist())
        
        if success:
            return jsonify({
                "success": True,
                "homography": True,
                "corners": court_homography.court_corners_px.tolist() if court_homography.court_corners_px is not None else None,
                "court_dimensions": {"width_m": court_width, "length_m": court_length},
                "diagnostics": court_homography.get_court_diagnostics()
            })
        return jsonify({"error": "Failed to compute homography"}), 500
    except Exception as e:
        logger.error(f"Court calibration failed: {e}")
        return jsonify({"error": str(e)}), 500


def sort_corners(corners):
    """Sort corners to: top-left, top-right, bottom-right, bottom-left."""
    if len(corners) < 4:
        return corners
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]], dtype=np.float32)


# ============================================================
# STAGE 4: World-Class Analysis
# ============================================================

@app.route("/api/analyze-player", methods=["POST"])
def analyze_player():
    """Run world-class analysis on selected player."""
    global current_video_path, court_homography, world_class_analyzer, world_class_results
    
    if current_video_path is None:
        return jsonify({"error": "No video loaded"}), 400
    
    data = request.get_json()
    if not data or "player_id" not in data:
        return jsonify({"error": "No player_id provided"}), 400
    
    player_id = data["player_id"]
    
    try:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        world_class_analyzer = WorldClassAnalyzer(court_homography)
        world_class_analyzer.set_fps(fps)
        init_shuttle()
        init_pose_model()
        
        if not pose_model or pose_model is False:
            cap.release()
            return jsonify({"error": "Pose model not available"}), 500
        
        frame_count = 0
        skip = data.get("skip_frames", 2)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % (skip + 1) != 0:
                frame_count += 1
                continue
            
            timestamp = frame_count / fps
            pose_results = pose_model(frame, conf=0.25, verbose=False)
            shuttle_detections = detect_shuttle_keypoints(frame, conf_threshold=0.15)
            
            if pose_results and len(pose_results) > 0:
                result = pose_results[0]
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kpts = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None
                    
                    if len(kpts) > player_id:
                        player_kps = kpts[player_id][:, :2].tolist()
                        player_scores = kpts[player_id][:, 2].tolist()
                        box = boxes[player_id].tolist() if boxes is not None and player_id < len(boxes) else [0,0,0,0]
                        
                        world_class_analyzer.analyze_frame(
                            player_kps, player_scores,
                            shuttle_detections, box, frame_count, timestamp)
            
            frame_count += 1
        
        cap.release()
        report = world_class_analyzer.generate_comprehensive_report()
        world_class_results = report
        
        return jsonify({
            "success": True, "report": report,
            "video_info": {"fps": fps, "total_frames": total_frames,
                          "width": width, "height": height,
                          "duration": total_frames / fps if fps > 0 else 0}
        })
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["GET"])
def get_report():
    if not world_class_results:
        return jsonify({"error": "No analysis results"}), 404
    return jsonify({"success": True, "report": world_class_results})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message"}), 400
    msg = data["message"]
    try:
        from llm_feedback import generate_chat_reply
        ctx = {"profile": {}, "feedback": {}, "pose_metrics": {}, "box_metrics": {}}
        if world_class_results:
            ctx["profile"] = {"posture": "good" if world_class_results.get("technical_scores", {}).get("posture", 0) > 0.6 else "needs work",
                          "balance": "good" if world_class_results.get("technical_scores", {}).get("balance", 0) > 0.6 else "needs work",
                          "stance": "athletic"}
            ctx["feedback"] = {"strengths": world_class_results.get("strengths", []),
                              "weaknesses": world_class_results.get("weaknesses", []),
                              "drills": world_class_results.get("drills", [])}
        reply = generate_chat_reply("player", msg, ctx)
        if reply:
            return jsonify({"success": True, "reply": reply})
    except Exception as e:
        logger.error(f"Chat error: {e}")
    return jsonify({"success": True, "reply": _fallback_chat(msg, world_class_results)})


def _fallback_chat(message, results):
    if not results:
        return "Please analyze a video first."
    msg = message.lower()
    if "strength" in msg:
        s = results.get("strengths", [])
        return f"Your strengths: {"; ".join(s[:3])}" if s else "Your fundamentals are solid."
    if "weak" in msg or "improve" in msg:
        w = results.get("weaknesses", [])
        return f"Areas to improve: {"; ".join(w[:3])}" if w else "Focus on consistency."
    if "shot" in msg or "stroke" in msg:
        st = results.get("stroke_analysis", {})
        d = st.get("distribution", {})
        if d:
            mc = max(d, key=d.get)
            return f"Most common shot: {mc} ({d[mc]} times)"
        return "Work on developing a wider range of shots."
    if "drill" in msg or "practice" in msg:
        dr = results.get("drills", [])
        return f"Drills: {"; ".join(dr[:3])}" if dr else "Practice shadow badminton daily."
    if "score" in msg:
        sc = results.get("technical_scores", {})
        return f"Overall score: {sc.get("overall", 0):.0%}"
    return f"Session: {results.get("session_summary", {}).get("total_strokes", 0)} strokes. Ask about strengths, weaknesses, shots, or drills."


if __name__ == "__main__":
    logger.info("Starting OmniCourt Professional Server...")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    init_yolo()
    init_pose_model()
    init_shuttle()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
