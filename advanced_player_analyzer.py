"""
Advanced Player Analysis Engine - Professional Grade
Provides comprehensive biomechanical and performance analysis
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrokeType(Enum):
    """Badminton stroke classifications."""
    SMASH = "smash"
    CLEAR = "clear"
    DROP = "drop"
    NET = "net_shot"
    LIFT = "lift"
    PUSH = "push"
    DRIVE = "drive"
    BLOCK = "block"
    SERVICE = "service"
    UNKNOWN = "unknown"


class MovementPhase(Enum):
    """Phases of a badminton stroke."""
    PREPARATION = "preparation"
    BACKSWING = "backswing"
    ACCELERATION = "acceleration"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class JointAngle:
    """Joint angle measurement."""
    joint_name: str
    angle_degrees: float
    confidence: float
    
    def __repr__(self):
        return f"{self.joint_name}: {self.angle_degrees:.1f}°"


@dataclass
class MotionMetrics:
    """Motion metrics for a player."""
    position_2d: np.ndarray  # [x, y] in court meters
    velocity: float  # pixels per frame or m/s
    acceleration: float
    direction_angle: float  # 0-360 degrees
    joint_angles: List[JointAngle]
    center_of_mass_height: float
    balance_center: float  # X coordinate of center of mass
    frame_number: int
    timestamp: float


class AdvancedPlayerAnalyzer:
    """
    Professional badminton player analysis engine.
    Analyzes biomechanics, movement patterns, stroke efficiency, and performance.
    """
    
    # COCO skeleton keypoints
    KEYPOINT_NAMES = {
        0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
        5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
        9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
        13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
    }
    
    def __init__(self, court_homography=None):
        self.court_homography = court_homography
        self.motion_history = []
        self.stroke_sequence = []
        self.current_stroke = None
        self.frame_count = 0
        
        # Thresholds for analysis
        self.min_confidence = 0.3
        self.velocity_threshold = 5.0  # pixels per frame
        self.acceleration_threshold = 2.0  # pixels per frame²
    
    def analyze_frame(self, keypoints: np.ndarray, keypoint_scores: np.ndarray,
                     box: np.ndarray, frame_number: int, timestamp: float = 0.0) -> Dict:
        """
        Comprehensive analysis of a single frame.
        Returns detailed metrics about player posture, position, and motion.
        """
        keypoints = np.array(keypoints, dtype=np.float64)
        keypoint_scores = np.array(keypoint_scores, dtype=np.float64)
        
        # Ensure proper shape
        if keypoints.shape != (17, 2):
            keypoints = np.zeros((17, 2), dtype=np.float64)
        if keypoint_scores.shape[0] != 17:
            keypoint_scores = np.ones(17) * 0.0
        
        # Extract basic metrics
        position_2d = self._estimate_player_position(keypoints, keypoint_scores)
        center_of_mass = self._calculate_center_of_mass(keypoints, keypoint_scores)
        joint_angles = self._calculate_joint_angles(keypoints, keypoint_scores)
        
        # Calculate motion metrics
        if self.motion_history:
            last_motion = self.motion_history[-1]
            velocity, acceleration = self._calculate_velocity_acceleration(
                position_2d, last_motion
            )
        else:
            velocity = 0.0
            acceleration = 0.0
        
        # Posture analysis
        posture_analysis = self._analyze_posture(keypoints, keypoint_scores)
        
        # Balance analysis
        balance_analysis = self._analyze_balance(keypoints, keypoint_scores)
        
        # Footwork analysis
        footwork_analysis = self._analyze_footwork(keypoints, keypoint_scores)
        
        # Determine movement phase
        movement_phase = self._determine_movement_phase(
            keypoints, keypoint_scores, velocity, joint_angles
        )
        
        # Detect potential stroke
        stroke_info = self._detect_stroke(
            keypoints, keypoint_scores, joint_angles, velocity
        )
        
        metrics = MotionMetrics(
            position_2d=position_2d,
            velocity=velocity,
            acceleration=acceleration,
            direction_angle=self._calculate_direction(position_2d),
            joint_angles=joint_angles,
            center_of_mass_height=center_of_mass[1],
            balance_center=center_of_mass[0],
            frame_number=frame_number,
            timestamp=timestamp
        )
        
        self.motion_history.append(metrics)
        self.frame_count += 1
        
        return {
            'metrics': metrics,
            'posture': posture_analysis,
            'balance': balance_analysis,
            'footwork': footwork_analysis,
            'movement_phase': movement_phase.value,
            'stroke': stroke_info,
            'joint_angles': [(j.joint_name, j.angle_degrees, j.confidence) 
                           for j in joint_angles],
            'keypoint_confidence': keypoint_scores.tolist()
        }
    
    def _estimate_player_position(self, keypoints: np.ndarray,
                                  keypoint_scores: np.ndarray) -> np.ndarray:
        """Estimate player's court position (2D meters or pixel coordinates)."""
        # Use hip center as primary position indicator
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_score = keypoint_scores[11]
        right_score = keypoint_scores[12]
        
        if left_score > self.min_confidence and right_score > self.min_confidence:
            position = (left_hip + right_hip) / 2
        elif left_score > self.min_confidence:
            position = left_hip
        elif right_score > self.min_confidence:
            position = right_hip
        else:
            # Fallback to shoulder center
            position = (keypoints[5] + keypoints[6]) / 2
        
        # Map to court coordinates if homography available
        if self.court_homography and self.court_homography.is_calibrated:
            court_pos = self.court_homography.map_pixel_to_court(position)
            if court_pos is not None:
                return court_pos
        
        return position
    
    def _calculate_center_of_mass(self, keypoints: np.ndarray,
                                  keypoint_scores: np.ndarray) -> np.ndarray:
        """Calculate center of mass using joint positions and weights."""
        # Approximate body segment masses (relative)
        joint_masses = {
            0: 1.0,   # nose (head)
            5: 3.0, 6: 3.0,    # shoulders
            7: 2.0, 8: 2.0,    # elbows
            9: 1.0, 10: 1.0,   # wrists
            11: 2.5, 12: 2.5,  # hips
            13: 2.0, 14: 2.0,  # knees
            15: 1.5, 16: 1.5   # ankles
        }
        
        total_mass = 0.0
        weighted_position = np.zeros(2, dtype=np.float64)
        
        for joint_id, mass in joint_masses.items():
            if keypoint_scores[joint_id] > self.min_confidence:
                weighted_position += keypoints[joint_id] * mass
                total_mass += mass
        
        if total_mass > 0:
            return weighted_position / total_mass
        else:
            return np.array([0.0, 0.0], dtype=np.float64)
    
    def _calculate_joint_angles(self, keypoints: np.ndarray,
                                keypoint_scores: np.ndarray) -> List[JointAngle]:
        """Calculate angles at major joints."""
        angles = []
        
        # Define joints as (start, center, end) triplets
        joint_triplets = [
            ('right_arm', (6, 8, 10)),      # right shoulder -> elbow -> wrist
            ('left_arm', (5, 7, 9)),        # left shoulder -> elbow -> wrist
            ('right_elbow', (6, 8, 10)),
            ('left_elbow', (5, 7, 9)),
            ('right_leg', (12, 14, 16)),    # right hip -> knee -> ankle
            ('left_leg', (11, 13, 15)),     # left hip -> knee -> ankle
            ('torso', (6, 12, 16)),         # right shoulder -> right hip -> right ankle
            ('stance', (11, 12, None)),     # hip width
        ]
        
        for joint_name, (idx1, idx2, idx3) in joint_triplets:
            if idx3 is not None:
                if (keypoint_scores[idx1] > self.min_confidence and
                    keypoint_scores[idx2] > self.min_confidence and
                    keypoint_scores[idx3] > self.min_confidence):
                    
                    angle = self._angle_between_points(
                        keypoints[idx1], keypoints[idx2], keypoints[idx3]
                    )
                    confidence = min(keypoint_scores[idx1], keypoint_scores[idx2], keypoint_scores[idx3])
                    angles.append(JointAngle(joint_name, angle, float(confidence)))
            else:
                # Special case: hip width
                if (keypoint_scores[idx1] > self.min_confidence and
                    keypoint_scores[idx2] > self.min_confidence):
                    dist = np.linalg.norm(keypoints[idx2] - keypoints[idx1])
                    confidence = min(keypoint_scores[idx1], keypoint_scores[idx2])
                    angles.append(JointAngle(joint_name, dist, float(confidence)))
        
        return angles
    
    def _angle_between_points(self, p1: np.ndarray, center: np.ndarray,
                             p2: np.ndarray) -> float:
        """Calculate angle at center between three points."""
        v1 = p1 - center
        v2 = p2 - center
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _calculate_velocity_acceleration(self, current_pos: np.ndarray,
                                         last_motion: MotionMetrics) -> Tuple[float, float]:
        """Calculate velocity and acceleration from motion history."""
        delta_pos = current_pos - last_motion.position_2d
        velocity = np.linalg.norm(delta_pos)
        
        # Approximate acceleration
        acceleration = velocity - last_motion.velocity
        
        return float(velocity), float(acceleration)
    
    def _calculate_direction(self, position: np.ndarray) -> float:
        """Calculate movement direction angle."""
        if len(self.motion_history) < 2:
            return 0.0
        
        prev_pos = self.motion_history[-1].position_2d
        delta = position - prev_pos
        
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        return angle % 360
    
    def _analyze_posture(self, keypoints: np.ndarray,
                        keypoint_scores: np.ndarray) -> Dict:
        """Analyze player's posture quality and readiness."""
        analysis = {
            'posture_type': 'unknown',
            'is_ready': False,
            'torso_angle': 0.0,
            'head_position': 'unknown',
            'shoulder_height_diff': 0.0
        }
        
        # Torso angle
        if (keypoint_scores[5] > self.min_confidence and keypoint_scores[11] > self.min_confidence and
            keypoint_scores[6] > self.min_confidence and keypoint_scores[12] > self.min_confidence):
            
            left_shoulder = keypoints[5]
            left_hip = keypoints[11]
            torso_vec = left_hip - left_shoulder
            torso_angle = np.degrees(np.arctan2(torso_vec[1], torso_vec[0]))
            analysis['torso_angle'] = float(torso_angle)
            
            # Determine posture type
            if -30 < torso_angle < 30:
                analysis['posture_type'] = 'upright'
            elif 30 <= torso_angle < 90:
                analysis['posture_type'] = 'slight_bend'
            elif torso_angle >= 90:
                analysis['posture_type'] = 'bent_forward'
        
        # Shoulder level (readiness indicator)
        if keypoint_scores[5] > self.min_confidence and keypoint_scores[6] > self.min_confidence:
            left_shoulder_y = keypoints[5, 1]
            right_shoulder_y = keypoints[6, 1]
            diff = abs(left_shoulder_y - right_shoulder_y)
            analysis['shoulder_height_diff'] = float(diff)
            
            # If shoulders are level, player is ready
            analysis['is_ready'] = diff < 20  # pixels threshold
        
        # Head position
        if keypoint_scores[0] > self.min_confidence:
            nose = keypoints[0]
            if keypoint_scores[5] > self.min_confidence and keypoint_scores[6] > self.min_confidence:
                shoulder_center = (keypoints[5] + keypoints[6]) / 2
                head_offset = nose[0] - shoulder_center[0]
                
                if head_offset < -20:
                    analysis['head_position'] = 'left'
                elif head_offset > 20:
                    analysis['head_position'] = 'right'
                else:
                    analysis['head_position'] = 'centered'
        
        return analysis
    
    def _analyze_balance(self, keypoints: np.ndarray,
                        keypoint_scores: np.ndarray) -> Dict:
        """Analyze player's balance and weight distribution."""
        analysis = {
            'balance_status': 'unknown',
            'weight_distribution': 'centered',
            'stance_width': 0.0,
            'stability_score': 0.0
        }
        
        # Calculate stance width
        if keypoint_scores[15] > self.min_confidence and keypoint_scores[16] > self.min_confidence:
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            stance_width = np.linalg.norm(right_ankle - left_ankle)
            analysis['stance_width'] = float(stance_width)
            
            # Determine balance status
            if stance_width < 30:
                analysis['balance_status'] = 'narrow_stance'
                analysis['stability_score'] = 0.5
            elif stance_width < 60:
                analysis['balance_status'] = 'neutral'
                analysis['stability_score'] = 0.8
            else:
                analysis['balance_status'] = 'wide_stance'
                analysis['stability_score'] = 0.9
        
        # Calculate center of mass relative to feet
        if (keypoint_scores[11] > self.min_confidence and keypoint_scores[12] > self.min_confidence and
            keypoint_scores[15] > self.min_confidence and keypoint_scores[16] > self.min_confidence):
            
            hip_center = (keypoints[11] + keypoints[12]) / 2
            ankle_center = (keypoints[15] + keypoints[16]) / 2
            
            offset = hip_center[0] - ankle_center[0]
            if abs(offset) < 10:
                analysis['weight_distribution'] = 'centered'
            elif offset < 0:
                analysis['weight_distribution'] = 'left'
            else:
                analysis['weight_distribution'] = 'right'
        
        return analysis
    
    def _analyze_footwork(self, keypoints: np.ndarray,
                         keypoint_scores: np.ndarray) -> Dict:
        """Analyze footwork quality and positioning."""
        analysis = {
            'foot_position': 'unknown',
            'is_athletic_stance': False,
            'left_ankle_height': 0.0,
            'right_ankle_height': 0.0
        }
        
        # Ankle positions
        if keypoint_scores[15] > self.min_confidence:
            analysis['left_ankle_height'] = float(keypoints[15, 1])
        if keypoint_scores[16] > self.min_confidence:
            analysis['right_ankle_height'] = float(keypoints[16, 1])
        
        # Check if in athletic stance (bent knees)
        if (keypoint_scores[13] > self.min_confidence and keypoint_scores[14] > self.min_confidence and
            keypoint_scores[11] > self.min_confidence and keypoint_scores[12] > self.min_confidence):
            
            left_knee_angle = self._angle_between_points(
                keypoints[11], keypoints[13], keypoints[15]
            )
            right_knee_angle = self._angle_between_points(
                keypoints[12], keypoints[14], keypoints[16]
            )
            
            # Athletic stance has bent knees (120-160 degrees)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            analysis['is_athletic_stance'] = 120 < avg_knee_angle < 160
        
        return analysis
    
    def _determine_movement_phase(self, keypoints: np.ndarray,
                                 keypoint_scores: np.ndarray,
                                 velocity: float,
                                 joint_angles: List[JointAngle]) -> MovementPhase:
        """Determine current phase of stroke execution."""
        if velocity > self.velocity_threshold:
            # High velocity indicates acceleration or execution
            # Check arm position to distinguish
            if (keypoint_scores[9] > self.min_confidence and
                keypoint_scores[10] > self.min_confidence):
                
                wrist_y_avg = (keypoints[9, 1] + keypoints[10, 1]) / 2
                shoulder_y_avg = (keypoints[5, 1] + keypoints[6, 1]) / 2
                
                if wrist_y_avg > shoulder_y_avg:
                    return MovementPhase.ACCELERATION
        
        return MovementPhase.RECOVERY
    
    def _detect_stroke(self, keypoints: np.ndarray, keypoint_scores: np.ndarray,
                      joint_angles: List[JointAngle], velocity: float) -> Dict:
        """Attempt to classify the current stroke type."""
        stroke_info = {
            'type': StrokeType.UNKNOWN.value,
            'confidence': 0.0,
            'characteristics': []
        }
        
        # This would normally use a ML model trained on stroke videos
        # For now, use heuristics based on joint angles and velocities
        
        if velocity < self.velocity_threshold:
            stroke_info['type'] = StrokeType.UNKNOWN.value
            return stroke_info
        
        # Extract right arm angles
        right_arm_angle = next((j.angle_degrees for j in joint_angles if j.joint_name == 'right_arm'), None)
        
        if right_arm_angle:
            # Smash: very extended arm, high velocity
            if right_arm_angle > 160 and velocity > 15:
                stroke_info['type'] = StrokeType.SMASH.value
                stroke_info['confidence'] = 0.6
                stroke_info['characteristics'] = ['extended_arm', 'high_velocity']
            # Drop/Net: low arm position
            elif keypoints[9, 1] > keypoints[6, 1]:
                stroke_info['type'] = StrokeType.DROP.value
                stroke_info['confidence'] = 0.5
                stroke_info['characteristics'] = ['low_arm']
        
        return stroke_info
    
    def generate_session_report(self) -> Dict:
        """Generate comprehensive analysis report for entire session."""
        if not self.motion_history:
            return {}
        
        motion_array = np.array([
            m.velocity for m in self.motion_history
        ])
        
        position_array = np.array([
            m.position_2d for m in self.motion_history
        ])
        
        report = {
            'total_frames': self.frame_count,
            'motion_statistics': {
                'avg_velocity': float(np.mean(motion_array)),
                'max_velocity': float(np.max(motion_array)),
                'min_velocity': float(np.min(motion_array)),
                'velocity_std': float(np.std(motion_array))
            },
            'court_coverage': {
                'min_x': float(position_array[:, 0].min()),
                'max_x': float(position_array[:, 0].max()),
                'min_y': float(position_array[:, 1].min()),
                'max_y': float(position_array[:, 1].max()),
                'coverage_area': float((position_array[:, 0].max() - position_array[:, 0].min()) *
                                     (position_array[:, 1].max() - position_array[:, 1].min()))
            },
            'detected_strokes': len(self.stroke_sequence)
        }
        
        return report
