"""
Multi-pose tracker with temporal smoothing for badminton players.
NumPy 2.x compatible with explicit float64 dtypes.
"""
import numpy as np


class PoseTrack:
    """Single pose track with Kalman-like smoothing."""

    def __init__(self, initial_keypoints, initial_box, track_id):
        self.track_id = track_id
        self.keypoints = np.array(initial_keypoints, dtype=np.float64)  # (17, 2)
        self.box = np.array(initial_box, dtype=np.float64)  # (4,)
        self.keypoint_scores = np.ones(17, dtype=np.float64) * 0.5

        self.keypoint_velocity = np.zeros_like(self.keypoints, dtype=np.float64)
        self.box_velocity = np.zeros(4, dtype=np.float64)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        self.alpha = 0.65  # Smoothing factor
        self.keypoint_history = [self.keypoints.copy()]
        self.box_history = [self.box.copy()]

    def predict(self):
        """Predict next state using velocity."""
        self.keypoints += self.keypoint_velocity
        self.box += self.box_velocity
        self.age += 1
        self.time_since_update += 1

    def update(self, keypoints, box, keypoint_scores=None):
        """Update track with new detection."""
        keypoints = np.array(keypoints, dtype=np.float64)
        box = np.array(box, dtype=np.float64)

        if keypoints.shape != (17, 2):
            padded = np.zeros((17, 2), dtype=np.float64)
            n = min(keypoints.shape[0], 17)
            padded[:n] = keypoints[:n]
            keypoints = padded

        # Compute velocity
        new_kpt_vel = keypoints - self.keypoints
        new_box_vel = box - self.box

        # Exponential smoothing
        self.keypoint_velocity = self.alpha * self.keypoint_velocity + (1 - self.alpha) * new_kpt_vel
        self.box_velocity = self.alpha * self.box_velocity + (1 - self.alpha) * new_box_vel

        # Update state
        self.keypoints = keypoints
        self.box = box

        if keypoint_scores is not None:
            keypoint_scores = np.array(keypoint_scores, dtype=np.float64)
            if keypoint_scores.shape[0] != 17:
                padded = np.zeros(17, dtype=np.float64)
                n = min(keypoint_scores.shape[0], 17)
                padded[:n] = keypoint_scores[:n]
                keypoint_scores = padded
            self.keypoint_scores = keypoint_scores

        self.hits += 1
        self.time_since_update = 0
        self.keypoint_history.append(self.keypoints.copy())
        self.box_history.append(self.box.copy())

        # Keep history bounded
        if len(self.keypoint_history) > 10:
            self.keypoint_history.pop(0)
            self.box_history.pop(0)

    def get_smoothed_keypoints(self, window=5):
        """Get temporally smoothed keypoints."""
        if len(self.keypoint_history) == 0:
            return self.keypoints.copy()

        window = min(window, len(self.keypoint_history))
        recent = self.keypoint_history[-window:]

        # Weighted average (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()

        smoothed = np.zeros_like(self.keypoints)
        for i, kpts in enumerate(recent):
            smoothed += weights[i] * kpts

        return smoothed

    @property
    def is_confirmed(self):
        return self.hits >= 2

    @property
    def is_deleted(self):
        return self.time_since_update > 8


class MultiPoseTracker:
    """Track multiple poses across frames."""

    def __init__(self, max_age=8):
        self.tracks = []
        self.max_age = max_age
        self.next_id = 0

    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of dicts with 'keypoints', 'box', 'score'
        """
        # Predict existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = []

        # Simple greedy matching by box IoU
        track_indices = list(range(len(self.tracks)))
        det_indices = list(range(len(detections)))

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                cost_matrix[t_idx, d_idx] = self._compute_cost(track, det)

        # Greedy assignment
        matched_pairs = []
        while cost_matrix.size > 0 and cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            t_idx, d_idx = min_idx

            if cost_matrix[t_idx, d_idx] > 200:  # Threshold
                break

            matched_pairs.append((t_idx, d_idx))
            cost_matrix = np.delete(cost_matrix, t_idx, axis=0)
            cost_matrix = np.delete(cost_matrix, d_idx, axis=1)

            # Adjust indices
            track_indices.pop(t_idx)
            det_indices.pop(d_idx)

        # Update matched tracks
        for t_idx, d_idx in matched_pairs:
            track = self.tracks[t_idx]
            det = detections[d_idx]
            track.update(
                det["keypoints"],
                det["box"],
                det.get("keypoint_scores")
            )

        # Unmatched tracks
        for t_idx in sorted([i for i in range(len(self.tracks)) if i not in [p[0] for p in matched_pairs]], reverse=True):
            unmatched_tracks.append(self.tracks[t_idx])

        # Unmatched detections -> new tracks
        matched_dets = set(d_idx for _, d_idx in matched_pairs)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = PoseTrack(
                    det["keypoints"],
                    det["box"],
                    self.next_id
                )
                if det.get("keypoint_scores") is not None:
                    new_track.keypoint_scores = np.array(det["keypoint_scores"], dtype=np.float64)
                self.tracks.append(new_track)
                self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        return [t for t in self.tracks if t.is_confirmed]

    def _compute_cost(self, track, detection):
        """Compute matching cost between track and detection."""
        # Box IoU cost
        t_box = track.box
        d_box = np.array(detection["box"], dtype=np.float64)

        ix1 = max(t_box[0], d_box[0])
        iy1 = max(t_box[1], d_box[1])
        ix2 = min(t_box[2], d_box[2])
        iy2 = min(t_box[3], d_box[3])

        if ix2 <= ix1 or iy2 <= iy1:
            iou = 0.0
        else:
            inter = (ix2 - ix1) * (iy2 - iy1)
            t_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
            d_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
            union = t_area + d_area - inter
            iou = inter / (union + 1e-6)

        # Keypoint distance cost
        d_kpts = np.array(detection["keypoints"], dtype=np.float64)
        if d_kpts.shape != (17, 2):
            padded = np.zeros((17, 2), dtype=np.float64)
            n = min(d_kpts.shape[0], 17)
            padded[:n] = d_kpts[:n]
            d_kpts = padded

        kpt_dist = np.mean(np.linalg.norm(track.keypoints - d_kpts, axis=1))

        # Combined cost (lower = better match)
        cost = (1 - iou) * 100 + kpt_dist * 0.5
        return cost
