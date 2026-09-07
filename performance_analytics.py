"""
Performance Analytics Engine - Professional Grade
Generates comprehensive statistics and performance insights
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrokeMetrics:
    """Metrics for a single stroke."""
    frame_start: int
    frame_end: int
    stroke_type: str
    velocity_peak: float
    position_start: Tuple[float, float]
    position_end: Tuple[float, float]
    duration_frames: int
    distance_traveled: float
    confidence: float


@dataclass
class MovementPattern:
    """Movement pattern across multiple frames."""
    pattern_type: str  # "rally", "rally_exchange", "movement", etc.
    frames: List[int]
    positions: List[Tuple[float, float]]
    velocities: List[float]
    avg_velocity: float
    max_velocity: float
    duration_sec: float


class PerformanceAnalytics:
    """
    Comprehensive performance analysis engine.
    Generates actionable insights from motion tracking data.
    """
    
    def __init__(self, court_homography=None, fps: float = 30.0):
        self.court_homography = court_homography
        self.fps = fps
        self.motion_history = []
        self.strokes = []
        self.movement_patterns = []
        
        # Thresholds
        self.velocity_stroke_threshold = 3.0  # pixels/frame
        self.velocity_movement_threshold = 0.5  # pixels/frame
        self.min_stroke_duration = 3  # frames
        self.max_stroke_duration = 120  # frames
    
    def add_frame_data(self, position: np.ndarray, velocity: float, 
                      keypoints: np.ndarray, keypoint_scores: np.ndarray,
                      frame_number: int, timestamp: float):
        """Add frame data to analysis."""
        self.motion_history.append({
            'frame': frame_number,
            'timestamp': timestamp,
            'position': position.copy() if isinstance(position, np.ndarray) else position,
            'velocity': velocity,
            'keypoints': keypoints.copy() if isinstance(keypoints, np.ndarray) else keypoints,
            'keypoint_scores': keypoint_scores.copy() if isinstance(keypoint_scores, np.ndarray) else keypoint_scores
        })
    
    def detect_strokes(self) -> List[StrokeMetrics]:
        """
        Detect strokes from motion history.
        Returns list of detected strokes with metrics.
        """
        if len(self.motion_history) < self.min_stroke_duration:
            return []
        
        strokes = []
        velocities = [m['velocity'] for m in self.motion_history]
        
        # Find peaks (strokes)
        in_stroke = False
        stroke_start = None
        max_velocity = 0.0
        
        for i, vel in enumerate(velocities):
            if vel > self.velocity_stroke_threshold:
                if not in_stroke:
                    stroke_start = i
                    in_stroke = True
                    max_velocity = vel
                else:
                    max_velocity = max(max_velocity, vel)
            else:
                if in_stroke:
                    stroke_end = i
                    duration = stroke_end - stroke_start
                    
                    if self.min_stroke_duration <= duration <= self.max_stroke_duration:
                        # Calculate metrics
                        start_pos = np.array(self.motion_history[stroke_start]['position'])
                        end_pos = np.array(self.motion_history[stroke_end]['position'])
                        distance = np.linalg.norm(end_pos - start_pos)
                        
                        stroke = StrokeMetrics(
                            frame_start=self.motion_history[stroke_start]['frame'],
                            frame_end=self.motion_history[stroke_end]['frame'],
                            stroke_type='detected',
                            velocity_peak=max_velocity,
                            position_start=tuple(start_pos),
                            position_end=tuple(end_pos),
                            duration_frames=duration,
                            distance_traveled=distance,
                            confidence=min(0.8, max_velocity / 30.0)  # Normalize confidence
                        )
                        strokes.append(stroke)
                    
                    in_stroke = False
                    max_velocity = 0.0
        
        self.strokes = strokes
        return strokes
    
    def analyze_court_coverage(self) -> Dict:
        """
        Analyze how much of the court was covered by the player.
        """
        if not self.motion_history:
            return {}
        
        if self.court_homography and self.court_homography.is_calibrated:
            # Map to court coordinates
            court_positions = []
            for m in self.motion_history:
                pos = m['position']
                court_pos = self.court_homography.map_pixel_to_court(np.array(pos))
                if court_pos is not None:
                    court_positions.append(court_pos)
            
            if court_positions:
                positions = np.array(court_positions)
                
                coverage = {
                    'min_x': float(positions[:, 0].min()),
                    'max_x': float(positions[:, 0].max()),
                    'min_y': float(positions[:, 1].min()),
                    'max_y': float(positions[:, 1].max()),
                    'coverage_width': float(positions[:, 0].max() - positions[:, 0].min()),
                    'coverage_length': float(positions[:, 1].max() - positions[:, 1].min()),
                    'court_area_covered': float(
                        (positions[:, 0].max() - positions[:, 0].min()) *
                        (positions[:, 1].max() - positions[:, 1].min())
                    ),
                    'total_distance_m': float(
                        np.sum([np.linalg.norm(positions[i+1] - positions[i]) 
                               for i in range(len(positions)-1)])
                    ) if len(positions) > 1 else 0.0,
                    'is_court_calibrated': True
                }
                return coverage
        
        # Fallback: pixel-based analysis
        positions = np.array([m['position'] for m in self.motion_history])
        
        return {
            'min_x': float(positions[:, 0].min()),
            'max_x': float(positions[:, 0].max()),
            'min_y': float(positions[:, 1].min()),
            'max_y': float(positions[:, 1].max()),
            'width': float(positions[:, 0].max() - positions[:, 0].min()),
            'height': float(positions[:, 1].max() - positions[:, 1].min()),
            'area': float(
                (positions[:, 0].max() - positions[:, 0].min()) *
                (positions[:, 1].max() - positions[:, 1].min())
            ),
            'is_court_calibrated': False
        }
    
    def analyze_movement_efficiency(self) -> Dict:
        """
        Analyze movement efficiency and footwork quality.
        """
        if len(self.motion_history) < 2:
            return {}
        
        velocities = np.array([m['velocity'] for m in self.motion_history])
        
        # Classify movement phases
        static_frames = np.sum(velocities < self.velocity_movement_threshold)
        active_frames = np.sum(velocities >= self.velocity_movement_threshold)
        stroke_frames = np.sum(velocities > self.velocity_stroke_threshold)
        
        total_frames = len(velocities)
        
        efficiency = {
            'static_time_pct': float(100 * static_frames / total_frames),
            'active_time_pct': float(100 * active_frames / total_frames),
            'stroke_time_pct': float(100 * stroke_frames / total_frames),
            'avg_velocity': float(np.mean(velocities)),
            'max_velocity': float(np.max(velocities)),
            'min_velocity': float(np.min(velocities)),
            'velocity_std': float(np.std(velocities)),
            'velocity_variance': float(np.var(velocities)),
            'movement_smoothness': self._calculate_smoothness(velocities)
        }
        
        return efficiency
    
    def _calculate_smoothness(self, velocities: np.ndarray) -> float:
        """
        Calculate movement smoothness (lower is smoother).
        Based on jerk (3rd derivative of position).
        """
        if len(velocities) < 3:
            return 0.0
        
        # Calculate acceleration (2nd derivative)
        accelerations = np.diff(velocities)
        
        # Calculate jerk (3rd derivative)
        jerks = np.diff(accelerations)
        
        # Smoothness score (normalized jerk)
        smoothness = float(np.mean(np.abs(jerks)) / (np.std(velocities) + 1e-6))
        
        return min(1.0, smoothness)  # Normalize to 0-1
    
    def analyze_posture_quality(self) -> Dict:
        """
        Analyze posture quality across session.
        """
        if not self.motion_history:
            return {}
        
        upright_count = 0
        bent_count = 0
        ready_count = 0
        
        for m in self.motion_history:
            keypoints = m.get('keypoints')
            keypoint_scores = m.get('keypoint_scores')
            
            if keypoints is not None and keypoint_scores is not None:
                # Simple posture heuristics
                if keypoint_scores[5] > 0.3 and keypoint_scores[12] > 0.3:
                    shoulder = keypoints[5]
                    hip = keypoints[12]
                    
                    # Check if upright (shoulders above hips)
                    if shoulder[1] < hip[1]:
                        upright_count += 1
                
                # Check if ready (level shoulders)
                if keypoint_scores[5] > 0.3 and keypoint_scores[6] > 0.3:
                    if abs(keypoints[5][1] - keypoints[6][1]) < 15:
                        ready_count += 1
                
                bent_count += 1
        
        total = len(self.motion_history)
        
        return {
            'upright_pct': float(100 * upright_count / max(total, 1)),
            'ready_stance_pct': float(100 * ready_count / max(total, 1)),
            'posture_score': float((upright_count + ready_count) / max(total * 2, 1))
        }
    
    def analyze_stroke_patterns(self) -> Dict:
        """
        Analyze stroke patterns and consistency.
        """
        if not self.strokes:
            return {'stroke_count': 0}
        
        strokes = self.strokes
        
        velocities = np.array([s.velocity_peak for s in strokes])
        durations = np.array([s.duration_frames for s in strokes])
        distances = np.array([s.distance_traveled for s in strokes])
        
        return {
            'stroke_count': len(strokes),
            'avg_velocity': float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
            'velocity_std': float(np.std(velocities)) if len(velocities) > 0 else 0.0,
            'velocity_consistency': float(1.0 - (np.std(velocities) / (np.mean(velocities) + 1e-6))) 
                                   if len(velocities) > 0 else 0.0,
            'avg_stroke_duration_frames': float(np.mean(durations)) if len(durations) > 0 else 0.0,
            'avg_distance_per_stroke': float(np.mean(distances)) if len(distances) > 0 else 0.0,
            'stroke_frequency_per_min': float(len(strokes) * 60 / (len(self.motion_history) / self.fps + 1e-6))
        }
    
    def calculate_fatigue_index(self) -> Dict:
        """
        Estimate fatigue based on movement patterns.
        Lower velocity, increased pauses, etc.
        """
        if len(self.motion_history) < 100:
            return {'fatigue_index': 0.0, 'data_insufficient': True}
        
        # Split into quarters
        quarter_len = len(self.motion_history) // 4
        quarters_data = []
        
        for q in range(4):
            start = q * quarter_len
            end = (q + 1) * quarter_len
            quarter_velocities = np.array([m['velocity'] for m in self.motion_history[start:end]])
            quarters_data.append({
                'avg_velocity': np.mean(quarter_velocities),
                'active_time': np.sum(quarter_velocities > self.velocity_movement_threshold) / len(quarter_velocities)
            })
        
        # Calculate trend (should decrease with fatigue)
        velocity_trend = quarters_data[-1]['avg_velocity'] / (quarters_data[0]['avg_velocity'] + 1e-6)
        activity_trend = quarters_data[-1]['active_time'] / (quarters_data[0]['active_time'] + 1e-6)
        
        # Fatigue index (0 = fresh, 1 = fatigued)
        fatigue_index = max(0.0, min(1.0, 1.0 - (velocity_trend + activity_trend) / 2))
        
        return {
            'fatigue_index': float(fatigue_index),
            'velocity_decline_pct': float(100 * (1 - velocity_trend)),
            'activity_decline_pct': float(100 * (1 - activity_trend)),
            'trend_data': quarters_data
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate complete performance report.
        """
        self.detect_strokes()
        
        return {
            'summary': {
                'total_frames': len(self.motion_history),
                'duration_sec': len(self.motion_history) / self.fps if self.fps > 0 else 0
            },
            'court_coverage': self.analyze_court_coverage(),
            'movement_efficiency': self.analyze_movement_efficiency(),
            'posture_quality': self.analyze_posture_quality(),
            'stroke_analysis': self.analyze_stroke_patterns(),
            'fatigue_analysis': self.calculate_fatigue_index(),
            'stroke_details': [
                {
                    'frame_start': s.frame_start,
                    'frame_end': s.frame_end,
                    'type': s.stroke_type,
                    'velocity_peak': s.velocity_peak,
                    'duration_frames': s.duration_frames,
                    'distance': s.distance_traveled,
                    'confidence': s.confidence
                }
                for s in self.strokes
            ]
        }
