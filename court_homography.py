"""
Court homography - auto-detect + manual click support.
3-corner completion using projective geometry (vanishing points).
"""
import cv2
import numpy as np

class CourtHomography:
    COURT_WIDTH_M = 6.10
    COURT_LENGTH_M = 13.40

    def __init__(self):
        self.is_calibrated = False
        self.homography_matrix = None
        self.inverse_homography = None
        self.court_corners_px = None
        self.court_corners_m = None
        self.camera_angle = "unknown"
        self.computed_corner = None  # The 4th corner we calculated

    def calibrate(self, frame):
        """Auto-detect court lines. Returns True/False."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=min(w,h)//4, maxLineGap=20)

        if lines is None or len(lines) < 4:
            return False

        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            segments.append((x1, y1, x2, y2, length, angle))

        horiz = [s for s in segments if abs(s[5]) < 30 or abs(s[5]) > 150]
        vert = [s for s in segments if 60 < abs(s[5]) < 120]
        if len(horiz) < 2 or len(vert) < 2:
            return False

        all_x = [p[0] for p in segments] + [p[2] for p in segments]
        all_y = [p[1] for p in segments] + [p[3] for p in segments]
        x_min = int(np.percentile(all_x, 10))
        x_max = int(np.percentile(all_x, 90))
        y_min = int(np.percentile(all_y, 10))
        y_max = int(np.percentile(all_y, 90))

        aspect = (y_max - y_min) / (x_max - x_min + 1e-6)
        if aspect < 0.3 or aspect > 2.0:
            return False

        self.court_corners_px = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float32)
        self.court_corners_m = np.array([[0,0],[self.COURT_WIDTH_M,0],[self.COURT_WIDTH_M,self.COURT_LENGTH_M],[0,self.COURT_LENGTH_M]], dtype=np.float32)
        self._compute_homography()
        self.is_calibrated = True
        self._detect_camera_angle()
        return True

    def set_manual_corners(self, corners_px):
        """Set court corners from user clicks. Accepts 3 or 4."""
        corners = np.array(corners_px, dtype=np.float32)

        if len(corners) == 3:
            corners, method = self._compute_fourth_corner_projective(corners)
            self.computed_corner = {
                "index": 3,
                "position": corners[3].tolist(),
                "method": method,
                "note": "4th corner estimated from perspective geometry. Verify it looks correct."
            }
        else:
            self.computed_corner = None

        self.court_corners_px = corners
        self.court_corners_m = np.array([[0,0],[self.COURT_WIDTH_M,0],[self.COURT_WIDTH_M,self.COURT_LENGTH_M],[0,self.COURT_LENGTH_M]], dtype=np.float32)
        self._compute_homography()
        self.is_calibrated = True
        self._detect_camera_angle()
        return True

    def _line_intersection(self, p1, p2, p3, p4):
        """Find intersection of line(p1,p2) and line(p3,p4). Returns [x,y] or None."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None  # Parallel lines

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py], dtype=np.float32)

    def _compute_fourth_corner_projective(self, three_corners):
        """
        Compute 4th corner of a rectangle from 3 corners using projective geometry.

        For a rectangle ABCD in perspective:
        - AB || DC (meet at vanishing point V1)
        - BC || AD (meet at vanishing point V2)
        - D = intersection of line(A, V2) and line(C, V1)

        With only 3 corners, we estimate vanishing points by assuming the 3 points
        are consecutive and using the frame geometry for the second parallel line.
        """
        p0, p1, p2 = three_corners

        # Edge vectors
        e0 = p1 - p0  # e.g., top edge TL->TR
        e1 = p2 - p1  # e.g., right edge TR->BR

        # Method 1: Pure parallelogram (best for near-orthographic / high camera angles)
        p3_para = p0 + p2 - p1

        # Method 2: Vanishing point estimation
        # Extend edge p0->p1 to find where a parallel line through p2 would go
        # Extend edge p1->p2 to find where a parallel line through p0 would go
        # In true perspective, these meet at vanishing points

        # Estimate V1: where line(p0,p1) meets line(p2, p2 + e0)
        # i.e., extend p0->p1 and extend p2 in same direction
        v1 = self._line_intersection(p0, p1, p2, p2 + e0)

        # Estimate V2: where line(p1,p2) meets line(p0, p0 + e1)
        v2 = self._line_intersection(p1, p2, p0, p0 + e1)

        if v1 is not None and v2 is not None:
            # D lies on line(p0, v2) [parallel to BC] 
            # AND on line(p2, v1) [parallel to AB]
            p3_vp = self._line_intersection(p0, v2, p2, v1)

            if p3_vp is not None:
                # Check if vanishing points are reasonable (not too far)
                v1_dist = np.linalg.norm(v1 - p0)
                v2_dist = np.linalg.norm(v2 - p0)

                # If vanishing points are extremely far, perspective is weak
                # Use a weighted blend of parallelogram and vanishing point
                if v1_dist > 5000 or v2_dist > 5000:
                    # Weak perspective - parallelogram is more reliable
                    weight = 0.8
                    p3_final = weight * p3_para + (1 - weight) * p3_vp
                    method = "blended (weak perspective)"
                else:
                    # Strong perspective - vanishing point method is better
                    p3_final = p3_vp
                    method = "vanishing_point"
            else:
                p3_final = p3_para
                method = "parallelogram_fallback"
        else:
            p3_final = p3_para
            method = "parallelogram"

        return np.array([p0, p1, p2, p3_final], dtype=np.float32), method

    def _compute_homography(self):
        self.homography_matrix, _ = cv2.findHomography(self.court_corners_px, self.court_corners_m)
        self.inverse_homography, _ = cv2.findHomography(self.court_corners_m, self.court_corners_px)

    def _detect_camera_angle(self):
        if self.court_corners_px is None:
            return
        tl, tr, br, bl = self.court_corners_px
        top_width = np.linalg.norm(tr - tl)
        bottom_width = np.linalg.norm(br - bl)
        left_height = np.linalg.norm(bl - tl)
        right_height = np.linalg.norm(br - tr)

        if top_width < bottom_width * 0.7:
            self.camera_angle = "high_angle"
        elif bottom_width < top_width * 0.7:
            self.camera_angle = "low_angle"
        elif left_height < right_height * 0.7 or right_height < left_height * 0.7:
            self.camera_angle = "side_angle"
        else:
            self.camera_angle = "neutral"

    def pixel_to_world(self, point_px):
        if not self.is_calibrated or self.homography_matrix is None:
            return None
        pt = np.array([[point_px]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
        return world_pt[0][0]

    def get_player_court_position(self, box):
        if not self.is_calibrated:
            return None
        x1, y1, x2, y2 = box
        return self.pixel_to_world(((x1+x2)/2, y2))

    def draw_court_overlay(self, frame, alpha=0.3):
        if not self.is_calibrated or self.court_corners_px is None:
            return frame
        overlay = frame.copy()
        pts = self.court_corners_px.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0,255,128), 2)
        cv2.fillPoly(overlay, [pts], (0,255,128))
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        return frame
