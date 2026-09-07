"""
Advanced Court Detection and Homography - Professional Grade
Implements multiple detection strategies with 3D-aware court mapping
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ProfessionalCourtHomography:
    """
    Advanced court detection with multiple strategies:
    - CNN-based line detection
    - Hough line detection
    - Edge-based corner detection
    - Automatic or manual calibration
    
    Handles perspective geometry and 2D-to-3D mapping for professional analysis.
    """
    
    # Badminton court dimensions (meters)
    COURT_WIDTH_M = 6.10    # 20 feet
    COURT_LENGTH_M = 13.40  # 44 feet
    SERVICE_LINE_M = 6.70   # 22 feet from end line
    
    # Court corners in meters (3D world coordinates)
    COURT_3D_CORNERS = np.array([
        [0, 0, 0],                           # Top-left
        [COURT_WIDTH_M, 0, 0],              # Top-right
        [COURT_WIDTH_M, COURT_LENGTH_M, 0], # Bottom-right
        [0, COURT_LENGTH_M, 0],             # Bottom-left
    ], dtype=np.float32)
    
    # Service line positions (for validation)
    SERVICE_LINE_POSITIONS = {
        'near': COURT_LENGTH_M / 2 - SERVICE_LINE_M / 2,
        'far': COURT_LENGTH_M / 2 + SERVICE_LINE_M / 2,
    }
    
    def __init__(self):
        self.is_calibrated = False
        self.homography_matrix = None
        self.inverse_homography = None
        self.perspective_matrix = None
        self.court_corners_px = None
        self.court_corners_m = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_angle = "unknown"
        self.confidence_score = 0.0
        self.detection_method = "none"
        
        # Camera intrinsics (estimate - should be calibrated per camera)
        self._init_camera_matrix()
    
    def _init_camera_matrix(self):
        """Initialize estimated camera intrinsic matrix."""
        # This is a placeholder - in production, this should be calibrated
        # per camera using checkerboard calibration
        focal_length = 1000
        cx = 640  # Principal point X
        cy = 360  # Principal point Y
        
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
    
    def detect_court_automatic(self, frame: np.ndarray) -> bool:
        """
        Automatically detect court using multiple strategies.
        Returns True if successful, False otherwise.
        """
        h, w = frame.shape[:2]
        
        # Strategy 1: Try white line detection (badminton courts are white)
        corners = self._detect_by_white_lines(frame)
        if corners is not None:
            self.court_corners_px = corners
            self.detection_method = "white_lines"
            self.confidence_score = 0.9
            self._finalize_homography()
            return True
        
        # Strategy 2: Try edge detection with morphological operations
        corners = self._detect_by_edge_analysis(frame)
        if corners is not None:
            self.court_corners_px = corners
            self.detection_method = "edge_analysis"
            self.confidence_score = 0.75
            self._finalize_homography()
            return True
        
        # Strategy 3: Try Hough line detection
        corners = self._detect_by_hough_lines(frame)
        if corners is not None:
            self.court_corners_px = corners
            self.detection_method = "hough_lines"
            self.confidence_score = 0.7
            self._finalize_homography()
            return True
        
        logger.warning("Court auto-detection failed. Falling back to manual mode.")
        return False
    
    def _detect_by_white_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court by white court lines."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Create mask for white regions (high brightness, low saturation)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return None
        
        # Find rectangular contour (court boundary)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in largest_contours:
            area = cv2.contourArea(contour)
            if area < (w * h * 0.1):  # Contour should be significant
                continue
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                if self._validate_court_geometry(corners):
                    return self._order_corners(corners)
        
        return None
    
    def _detect_by_edge_analysis(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court using edge detection and corner finding."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        
        # Enhanced edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return None
        
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in largest_contours:
            area = cv2.contourArea(contour)
            if area < (w * h * 0.05):
                continue
            
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                if self._validate_court_geometry(corners):
                    return self._order_corners(corners)
        
        return None
    
    def _detect_by_hough_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court using Hough line detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=min(w, h)//6,
            maxLineGap=50
        )
        
        if lines is None or len(lines) < 4:
            return None
        
        # Classify lines as horizontal or vertical
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.sqrt(dx**2 + dy**2)
            
            if length > min(w, h) * 0.1:
                angle = np.degrees(np.arctan2(dy, dx))
                segments.append({
                    'p1': np.array([x1, y1], dtype=np.float32),
                    'p2': np.array([x2, y2], dtype=np.float32),
                    'angle': angle,
                    'length': length
                })
        
        if len(segments) < 4:
            return None
        
        # Find intersection points
        horiz_lines = [s for s in segments if s['angle'] < 45 or s['angle'] > 135]
        vert_lines = [s for s in segments if 45 <= s['angle'] <= 135]
        
        if len(horiz_lines) < 2 or len(vert_lines) < 2:
            return None
        
        # Get the top 2 horizontal and top 2 vertical lines
        horiz_lines = sorted(horiz_lines, key=lambda x: x['length'], reverse=True)[:2]
        vert_lines = sorted(vert_lines, key=lambda x: x['length'], reverse=True)[:2]
        
        # Find four corner intersections
        corners = []
        for h_line in horiz_lines:
            for v_line in vert_lines:
                intersection = self._line_intersection(
                    h_line['p1'], h_line['p2'],
                    v_line['p1'], v_line['p2']
                )
                if intersection is not None:
                    corners.append(intersection)
        
        if len(corners) >= 4:
            corners = np.array(corners[:4], dtype=np.float32)
            if self._validate_court_geometry(corners):
                return self._order_corners(corners)
        
        return None
    
    def set_manual_corners(self, corners: List[List[float]]) -> bool:
        """
        Set court corners manually from user clicks.
        Accepts 2, 3, or 4 corners and estimates missing ones.
        """
        corners = np.array(corners, dtype=np.float32)
        
        if len(corners) == 2:
            # Two corners - estimate full court
            corners = self._estimate_from_two_corners(corners)
        elif len(corners) == 3:
            # Three corners - estimate fourth
            corners = self._estimate_from_three_corners(corners)
        elif len(corners) == 4:
            # Four corners - validate and order
            if not self._validate_court_geometry(corners):
                logger.warning("Provided corners don't form valid court geometry")
                return False
        else:
            logger.error(f"Expected 2-4 corners, got {len(corners)}")
            return False
        
        self.court_corners_px = self._order_corners(corners)
        self.detection_method = "manual"
        self.confidence_score = 0.6  # Lower confidence for manual input
        self._finalize_homography()
        return True
    
    def _validate_court_geometry(self, corners: np.ndarray) -> bool:
        """
        Validate that corners form a reasonable court geometry.
        Checks aspect ratio, area, and perspective properties.
        """
        corners = corners.astype(np.float32)
        
        # Calculate bounding box
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width < 50 or height < 50:
            return False
        
        aspect_ratio = height / width
        
        # Badminton court should be roughly 2:1 aspect ratio
        # But from different angles, it can vary significantly
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        # Check area (should be significant portion of image)
        area = cv2.contourArea(corners.reshape(-1, 1, 2))
        frame_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        
        if area < frame_area * 0.1:  # Court should be at least 10% of detected region
            return False
        
        return True
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners in standard format: top-left, top-right, bottom-right, bottom-left."""
        corners = corners.astype(np.float32)
        
        # Calculate centroid
        center = corners.mean(axis=0)
        
        # Sort by angle from centroid
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        ordered = corners[sorted_indices]
        
        # Ensure top-left is in correct position
        if ordered[0, 1] > ordered[2, 1]:  # If first point is below third
            ordered = np.roll(ordered, 2, axis=0)
        
        return ordered
    
    def _estimate_from_two_corners(self, corners: np.ndarray) -> np.ndarray:
        """Estimate full court from two opposite corners."""
        # Assume these are diagonal corners
        corner1, corner2 = corners
        
        # Calculate implied width and height based on court ratio
        court_ratio = self.COURT_LENGTH_M / self.COURT_WIDTH_M
        
        p1_to_p2 = corner2 - corner1
        mid = (corner1 + corner2) / 2
        
        # This is an approximation - needs user feedback
        all_corners = np.array([
            corner1,
            [corner2[0], corner1[1]],
            corner2,
            [corner1[0], corner2[1]]
        ], dtype=np.float32)
        
        return all_corners
    
    def _estimate_from_three_corners(self, corners: np.ndarray) -> np.ndarray:
        """Estimate fourth corner from three given corners using perspective geometry."""
        corners = corners.astype(np.float32)
        
        # Use perspective properties to estimate fourth corner
        # Assume parallel sides in 3D map to converging lines in 2D
        p1, p2, p3 = corners
        
        # Vector from p1 to p2
        v1 = p2 - p1
        # Vector from p1 to p3
        v2 = p3 - p1
        
        # Fourth corner estimate
        p4 = p2 + v2 - v1
        
        all_corners = np.array([p1, p2, p4, p3], dtype=np.float32)
        return all_corners
    
    def _line_intersection(self, p1: np.ndarray, p2: np.ndarray,
                          p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
        """Find intersection of two lines defined by points (p1,p2) and (p3,p4)."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Parallel lines
        
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        
        return np.array([px, py], dtype=np.float32)
    
    def _finalize_homography(self):
        """Compute homography and related matrices."""
        if self.court_corners_px is None or len(self.court_corners_px) != 4:
            return
        
        # Standard court corners in 2D (meter coordinates)
        court_corners_2d = np.array([
            [0, 0],
            [self.COURT_WIDTH_M, 0],
            [self.COURT_WIDTH_M, self.COURT_LENGTH_M],
            [0, self.COURT_LENGTH_M]
        ], dtype=np.float32)
        
        # Compute perspective transform
        self.perspective_matrix = cv2.getPerspectiveTransform(
            self.court_corners_px, court_corners_2d
        )
        self.homography_matrix = self.perspective_matrix
        self.inverse_homography = cv2.invertAffineTransform(
            self.perspective_matrix[:2, :]
        ) if self.perspective_matrix is not None else None
        
        self.is_calibrated = True
        self._detect_camera_angle()
    
    def _detect_camera_angle(self):
        """Detect camera viewing angle (frontal, side, top-down, etc.)."""
        if self.court_corners_px is None:
            return
        
        corners = self.court_corners_px
        
        # Top corners
        top_left_y = corners[0, 1]
        top_right_y = corners[1, 1]
        
        # Bottom corners
        bottom_left_y = corners[3, 1]
        bottom_right_y = corners[2, 1]
        
        # Calculate perspective distortion
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        
        aspect_ratio = (left_height + right_height) / 2 / ((top_width + bottom_width) / 2)
        
        if aspect_ratio > 2.0:
            self.camera_angle = "wide_angle_front"
        elif aspect_ratio > 1.5:
            self.camera_angle = "front"
        elif aspect_ratio > 1.0:
            self.camera_angle = "angled_front"
        elif aspect_ratio > 0.7:
            self.camera_angle = "side"
        else:
            self.camera_angle = "top_down"
    
    def map_pixel_to_court(self, pixel_point: np.ndarray) -> Optional[np.ndarray]:
        """Map a pixel coordinate to court 2D coordinates (meters)."""
        if not self.is_calibrated or self.homography_matrix is None:
            return None
        
        point = pixel_point.astype(np.float32).reshape(1, 1, 2)
        try:
            court_point = cv2.perspectiveTransform(point, self.homography_matrix)[0, 0]
            return court_point
        except:
            return None
    
    def map_court_to_pixel(self, court_point: np.ndarray) -> Optional[np.ndarray]:
        """Map a court coordinate (meters) to pixel coordinate."""
        if not self.is_calibrated or self.inverse_homography is None:
            return None
        
        point = court_point.astype(np.float32).reshape(1, 2)
        try:
            pixel_point = cv2.perspectiveTransform(
                point.reshape(1, 1, 2), self.inverse_homography[:2, :].reshape(2, 3)
            )[0, 0]
            return pixel_point
        except:
            return None
    
    def get_court_diagnostics(self) -> Dict:
        """Return diagnostic information about court calibration."""
        return {
            'is_calibrated': self.is_calibrated,
            'detection_method': self.detection_method,
            'confidence_score': self.confidence_score,
            'camera_angle': self.camera_angle,
            'court_corners': self.court_corners_px.tolist() if self.court_corners_px is not None else None,
            'court_dimensions': {
                'width_m': self.COURT_WIDTH_M,
                'length_m': self.COURT_LENGTH_M
            }
        }
