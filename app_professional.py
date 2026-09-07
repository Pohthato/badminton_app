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
from shuttlecock_detector import detect_shuttlecocks
from pose_tracker import MultiPoseTracker
from visualization_utils import draw_skeleton
from data_processing import apply_nms, center_distance, filter_close_detections

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
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info("Starting OmniCourt Professional Server...")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Initialize models
    init_yolo()
    init_pose_model()
    
    # Run server
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
