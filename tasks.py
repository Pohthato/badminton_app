"""
Celery tasks for background video processing.
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from celery import Celery
from ultralytics import YOLO

from court_homography import CourtHomography
from pose_tracker import MultiPoseTracker
from shuttlecock_detector import detect_shuttlecocks
from data_processing import apply_nms, center_distance, compute_box_metrics, filter_close_detections
from player_analysis import compute_pose_features_from_keypoints, find_best_pose_match, summarize_pose_metrics
from visualization_utils import draw_skeleton

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("badminton_tasks", broker=redis_url, backend=redis_url)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"], result_serializer="json",
    timezone="UTC", enable_utc=True, task_track_started=True,
    task_time_limit=600, task_soft_time_limit=540,
    worker_prefetch_multiplier=1,
)

MAX_PROCESSING_FRAMES = 3000
MAX_VIDEO_DURATION_SEC = 120
POSE_CONFIDENCE = 0.25

BADMINTON_CLASSES = {
    0: "player", 1: "player_smash", 2: "player_clear", 3: "player_drop",
    4: "player_net_shot", 5: "player_lift", 6: "player_push",
    7: "player_drive", 8: "player_cross", 9: "player_net_kill"
}

@celery_app.task(bind=True)
def process_video_analysis(self, video_path, player_labels, initial_boxes, frame_size):
    import time
    start_time = time.time()

    self.update_state(state="PROGRESS", meta={"progress": 5, "step": "Loading models..."})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "badminton_actions_v1.pt"
    try:
        action_model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
        action_model.to(device)
    except Exception as e:
        return {"error": f"Failed to load action model: {str(e)}"}

    self.update_state(state="PROGRESS", meta={"progress": 10, "step": "Loading pose model..."})
    try:
        pose_model = YOLO("yolov8s-pose.pt")
        pose_model.to(device)
    except Exception as e:
        return {"error": f"Failed to load pose model: {str(e)}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if duration_sec > MAX_VIDEO_DURATION_SEC:
        cap.release()
        return {"error": f"Video too long: {duration_sec:.1f}s (max {MAX_VIDEO_DURATION_SEC}s)"}

    max_frames = min(total_frames, MAX_PROCESSING_FRAMES)

    trackers = {label: MultiPoseTracker(max_age=8) for label in player_labels}
    court = CourtHomography()
    court_calibrated = False

    player_data = {label: {"boxes": [], "poses": [], "box_metrics": {}, "pose_metrics": {}, "court_positions": []} for label in player_labels}

    frame_idx = 0
    processed = 0

    try:
        while frame_idx < max_frames:
            elapsed = time.time() - start_time
            if elapsed > 480:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 30 == 0:
                progress = int(10 + (frame_idx / max_frames) * 85)
                self.update_state(state="PROGRESS", meta={"progress": progress, "step": f"Processing frame {frame_idx}/{max_frames}..."})

            h, w = frame.shape[:2]

            if frame_idx == 0:
                court_calibrated = court.calibrate(frame)

            # Detect players
            results = action_model(frame, conf=0.25, verbose=False)
            detections = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if cls <= 9 and conf >= 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({"box": (x1, y1, x2, y2), "action": BADMINTON_CLASSES.get(cls, "player").replace("player_", "").upper(), "conf": conf})

            # Detect poses
            pose_results = pose_model(frame, conf=POSE_CONFIDENCE, verbose=False)
            pose_detections = []
            if pose_results and len(pose_results[0].keypoints) > 0:
                kpts = pose_results[0].keypoints.data.cpu().numpy()
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                scores = pose_results[0].boxes.conf.cpu().numpy()
                for i in range(len(kpts)):
                    pose_detections.append({
                        "keypoints": kpts[i][:, :2],
                        "keypoint_scores": kpts[i][:, 2],
                        "box": tuple(map(int, boxes[i])),
                        "score": float(scores[i])
                    })

            # Track each selected player
            for label in player_labels:
                init_box = initial_boxes.get(label)
                if not init_box:
                    continue

                sx = w / frame_size[0] if frame_size else 1.0
                sy = h / frame_size[1] if frame_size else 1.0
                target_box = (int(init_box[0]*sx), int(init_box[1]*sy), int(init_box[2]*sx), int(init_box[3]*sy))

                best_det = None
                best_score = -999
                for det in detections:
                    dist = center_distance(target_box, det["box"])
                    score = 100 - dist * 0.1 + det["conf"] * 10
                    if score > best_score:
                        best_score = score
                        best_det = det

                if best_det and best_score > 0.15:
                    tracker_dets = [{"keypoints": p["keypoints"], "box": p["box"], "score": p["score"]} for p in pose_detections]
                    confirmed = trackers[label].update(tracker_dets)
                    target_track = min(confirmed, key=lambda t: center_distance(best_det["box"], t.box), default=None)

                    if target_track:
                        box = tuple(map(int, target_track.box))
                        player_data[label]["boxes"].append(box)

                        smoothed_kpts = target_track.get_smoothed_keypoints()
                        kpt_scores = target_track.keypoint_scores

                        if smoothed_kpts is not None and smoothed_kpts.shape == (17, 2):
                            pose_features = compute_pose_features_from_keypoints(smoothed_kpts, kpt_scores)
                            player_data[label]["poses"].append(pose_features)

                        if court_calibrated:
                            pos = court.get_player_court_position(box)
                            if pos:
                                player_data[label]["court_positions"].append(pos)

            frame_idx += 1
            processed += 1

        cap.release()

        self.update_state(state="PROGRESS", meta={"progress": 95, "step": "Computing metrics..."})

        for label in player_labels:
            boxes = player_data[label]["boxes"]
            poses = player_data[label]["poses"]

            if boxes:
                player_data[label]["box_metrics"] = compute_box_metrics(boxes)

            if poses and len(poses) > 0:
                try:
                    player_data[label]["pose_metrics"] = summarize_pose_metrics(poses)
                except Exception as e:
                    player_data[label]["pose_metrics"] = {"upright_ratio": 0, "centred_ratio": 0, "athletic_ready_ratio": 0, "error": str(e)}

            court_positions = player_data[label]["court_positions"]
            if court_positions and len(court_positions) > 1:
                total_m = 0
                steps = []
                for i in range(1, len(court_positions)):
                    dx = court_positions[i][0] - court_positions[i-1][0]
                    dy = court_positions[i][1] - court_positions[i-1][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    total_m += dist
                    steps.append(dist)
                player_data[label]["real_movement"] = {
                    "total_meters": round(total_m, 2),
                    "avg_step_meters": round(np.mean(steps), 3) if steps else 0,
                }

        self.update_state(state="PROGRESS", meta={"progress": 100, "step": "Complete!"})

        return {
            "player_data": player_data,
            "total_frames_processed": processed,
            "court_calibrated": court_calibrated,
            "duration_sec": round(duration_sec, 1),
        }

    except Exception as e:
        cap.release()
        import traceback
        return {"error": f"Processing failed: {str(e)}\n{traceback.format_exc()}"}

@celery_app.task
def cleanup_old_files(upload_folder, max_age_hours=24):
    import time
    from pathlib import Path
    folder = Path(upload_folder)
    if not folder.exists():
        return {"cleaned": 0}
    now = time.time()
    max_age_sec = max_age_hours * 3600
    cleaned = 0
    for f in folder.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > max_age_sec:
            f.unlink()
            cleaned += 1
    return {"cleaned": cleaned}
