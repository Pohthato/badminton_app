"""
Live Video Analysis Pipeline - Professional Grade
Handles real-time frame processing, analysis, and streaming
"""
import cv2
import numpy as np
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import logging
from collections import deque
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class LiveVideoAnalyzer:
    """
    Processes video frames in real-time with multi-threaded analysis.
    Handles skeleton detection, shuttlecock tracking, racket detection.
    """
    
    def __init__(self, pose_model, yolo_model, shuttle_detector=None,
                 court_homography=None, max_queue_size=30):
        self.pose_model = pose_model
        self.yolo_model = yolo_model
        self.shuttle_detector = shuttle_detector
        self.court_homography = court_homography
        
        # Threading
        self.analysis_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        
        # Frame tracking
        self.current_frame_number = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        
        # Visualization settings
        self.draw_skeleton = True
        self.draw_shuttlecock = True
        self.draw_racket = True
        self.draw_court_points = True
        self.overlay_opacity = 0.7
    
    def process_video_stream(self, video_path: str, 
                           player_analyzer=None,
                           skip_frames: int = 0) -> Generator[Dict, None, None]:
        """
        Process video file and yield analysis results with visualization frames.
        
        Yields dicts with:
        - 'frame_number': int
        - 'timestamp': float
        - 'frame_base64': str (annotated frame as base64 PNG)
        - 'analysis': dict (detailed analysis data)
        - 'metadata': dict (video info)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            metadata = {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration_seconds': total_frames / fps if fps > 0 else 0
            }
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    self.frames_skipped += 1
                    continue
                
                timestamp = frame_count / fps if fps > 0 else 0.0
                
                # Resize if too large
                if width > 1280:
                    scale = 1280 / width
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
                # Run analysis
                analysis = self._analyze_frame(
                    frame, frame_count, timestamp, player_analyzer
                )
                
                # Create annotated frame
                annotated = self._create_annotated_frame(frame, analysis)
                
                # Convert to base64 for transmission
                frame_base64 = self._frame_to_base64(annotated)
                
                # Yield result
                yield {
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'frame_base64': frame_base64,
                    'analysis': analysis,
                    'metadata': metadata
                }
                
                frame_count += 1
                self.frames_processed += 1
        
        finally:
            cap.release()
    
    def _analyze_frame(self, frame: np.ndarray, frame_number: int,
                      timestamp: float, player_analyzer=None) -> Dict:
        """Run all analysis on a single frame."""
        h, w = frame.shape[:2]
        analysis = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'frame_shape': [h, w],
            'detections': {
                'players': [],
                'shuttlecock': None,
                'racket': None
            },
            'player_analysis': [],
            'court_points': []
        }
        
        # Detect players with pose
        if self.pose_model and self.pose_model is not False:
            try:
                results = self.pose_model(frame, conf=0.25, verbose=False)
                
                if results and len(results[0].keypoints) > 0:
                    kpts = results[0].keypoints.data.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    
                    for i in range(len(kpts)):
                        player_data = {
                            'player_id': i,
                            'box': boxes[i].tolist(),
                            'confidence': float(scores[i]),
                            'keypoints': kpts[i][:, :2].tolist(),
                            'keypoint_scores': kpts[i][:, 2].tolist()
                        }
                        
                        # Run player-specific analysis
                        if player_analyzer:
                            player_metrics = player_analyzer.analyze_frame(
                                kpts[i][:, :2],
                                kpts[i][:, 2],
                                boxes[i],
                                frame_number,
                                timestamp
                            )
                            player_data['analysis'] = player_metrics
                            
                            # Map to court if calibrated
                            if self.court_homography and self.court_homography.is_calibrated:
                                hip_center = (kpts[i][11][:2] + kpts[i][12][:2]) / 2
                                court_pos = self.court_homography.map_pixel_to_court(hip_center)
                                if court_pos is not None:
                                    player_data['court_position'] = court_pos.tolist()
                        
                        analysis['detections']['players'].append(player_data)
            
            except Exception as e:
                logger.warning(f"Pose detection failed: {e}")
        
        # Detect shuttlecock
        if self.shuttle_detector:
            try:
                shuttle_boxes = self.shuttle_detector(frame, conf_threshold=0.25)
                if shuttle_boxes:
                    # Get the highest confidence detection
                    shuttle = shuttle_boxes[0]
                    x1, y1, x2, y2, conf = shuttle
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    
                    shuttle_data = {
                        'box': [x1, y1, x2, y2],
                        'center': center,
                        'confidence': float(conf)
                    }
                    
                    # Map to court if calibrated
                    if self.court_homography and self.court_homography.is_calibrated:
                        court_pos = self.court_homography.map_pixel_to_court(np.array(center))
                        if court_pos is not None:
                            shuttle_data['court_position'] = court_pos.tolist()
                    
                    analysis['detections']['shuttlecock'] = shuttle_data
            
            except Exception as e:
                logger.warning(f"Shuttlecock detection failed: {e}")
        
        return analysis
    
    def _create_annotated_frame(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """Create annotated frame with overlays."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw skeleton overlays
        if self.draw_skeleton and analysis['detections']['players']:
            for player in analysis['detections']['players']:
                keypoints = player.get('keypoints', [])
                keypoint_scores = player.get('keypoint_scores', [])
                
                if keypoints:
                    annotated = self._draw_skeleton(
                        annotated, keypoints, keypoint_scores
                    )
        
        # Draw shuttlecock
        if self.draw_shuttlecock and analysis['detections']['shuttlecock']:
            shuttle = analysis['detections']['shuttlecock']
            x1, y1, x2, y2 = shuttle['box']
            center = shuttle['center']
            conf = shuttle['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 255), 2)
            
            # Draw center point
            cv2.circle(annotated, (int(center[0]), int(center[1])), 4,
                      (0, 255, 255), -1)
            
            # Draw confidence
            cv2.putText(annotated, f"Shuttle: {conf:.2f}",
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw court points if calibrated
        if self.draw_court_points and self.court_homography:
            if self.court_homography.is_calibrated:
                annotated = self._draw_court_overlay(annotated)
        
        # Draw frame info
        cv2.putText(annotated, f"Frame: {analysis['frame_number']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Time: {analysis['timestamp']:.2f}s",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if analysis['detections']['players']:
            cv2.putText(annotated, f"Players: {len(analysis['detections']['players'])}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: List[List[float]],
                      keypoint_scores: List[float], 
                      min_confidence: float = 0.3) -> np.ndarray:
        """Draw skeleton on frame."""
        keypoints = np.array(keypoints, dtype=np.float32)
        keypoint_scores = np.array(keypoint_scores, dtype=np.float32)
        
        # COCO skeleton connections
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in skeleton_connections:
            if (keypoint_scores[start_idx] > min_confidence and
                keypoint_scores[end_idx] > min_confidence):
                
                start_pos = tuple(keypoints[start_idx].astype(int))
                end_pos = tuple(keypoints[end_idx].astype(int))
                
                # Color based on body part
                if start_idx < 5:  # Head
                    color = (0, 255, 0)
                elif start_idx < 11:  # Arms
                    color = (0, 165, 255)
                else:  # Legs
                    color = (255, 0, 0)
                
                cv2.line(frame, start_pos, end_pos, color, 2)
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if keypoint_scores[i] > min_confidence:
                pos = (int(x), int(y))
                color = (0, 255, 0)
                cv2.circle(frame, pos, 3, color, -1)
        
        return frame
    
    def _draw_court_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw court boundary and reference lines."""
        if not self.court_homography or not self.court_homography.court_corners_px:
            return frame
        
        corners = self.court_homography.court_corners_px.astype(int)
        
        # Draw court boundary
        cv2.polylines(frame, [corners], True, (0, 255, 255), 2)
        
        # Draw corner points
        for i, corner in enumerate(corners):
            cv2.circle(frame, tuple(corner), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"C{i}", tuple(corner + np.array([10, 10])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 PNG string."""
        _, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()
        return base64.b64encode(frame_bytes).decode('utf-8')
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'total_processed': self.frames_processed + self.frames_skipped,
            'skip_ratio': (self.frames_skipped / (self.frames_processed + self.frames_skipped + 1e-6))
        }
