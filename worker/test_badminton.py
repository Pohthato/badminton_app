import unittest

import numpy as np

from badminton import (
    SINGLES_WIDTH_M,
    classify_shot,
    court_width_m,
    detect_contacts,
    detect_racket_swings,
    racket_swing_speed_metrics,
    segment_rallies,
    shot_distribution,
    split_step_amplitude_metrics,
)


class CourtModelTests(unittest.TestCase):
    def test_singles_is_not_doubles_width(self):
        self.assertEqual(court_width_m("singles"), SINGLES_WIDTH_M)
        self.assertEqual(court_width_m("doubles"), 6.10)
        self.assertLess(court_width_m("singles"), court_width_m("doubles"))


class ContactAndShotTests(unittest.TestCase):
    def test_contact_requires_speed_collapse_and_racket_proximity(self):
        shuttle = []
        x = 200.0
        speed = 40.0
        for index in range(12):
            if index >= 8:
                speed = 4.0
            x += speed
            shuttle.append({"frame": index, "timeMs": index * 33, "xy": np.array([x, 120.0]), "confidence": 0.9})
        racket = [{"timeMs": 8 * 33, "xy": np.array([shuttle[8]["xy"][0], 118.0]), "confidence": 0.88}]
        contacts = detect_contacts(shuttle, [], racket, 800.0)
        self.assertGreaterEqual(len(contacts), 1)
        self.assertEqual(contacts[0]["type"], "contact")

    def test_unverified_shots_are_excluded_from_distribution(self):
        shots = [
            {"label": "clear", "verified": True},
            {"label": "drop", "verified": False},
            {"label": "clear", "verified": True},
        ]
        self.assertEqual(shot_distribution(shots), {"clear": 2})

    def test_low_evidence_shot_stays_unverified(self):
        classified = classify_shot(
            {"confidence": 0.9},
            {"outboundSpeed": 10, "depth": "unknown", "imageDy": 0, "afterCount": 0, "nearNet": False, "fromBack": False, "direction": "unknown"},
            0,
        )
        self.assertFalse(classified["verified"])
        self.assertEqual(classified["label"], "unverified")

    def test_rallies_split_on_long_gaps(self):
        contacts = [
            {"timeMs": 100, "frame": 3, "confidence": 0.9, "type": "contact", "source": "shuttle"},
            {"timeMs": 400, "frame": 12, "confidence": 0.9, "type": "contact", "source": "shuttle"},
            {"timeMs": 4000, "frame": 120, "confidence": 0.9, "type": "contact", "source": "shuttle"},
        ]
        rallies = segment_rallies(contacts, [])
        self.assertEqual(len(rallies), 2)
        self.assertEqual(rallies[0]["contactCount"], 2)
        self.assertEqual(rallies[1]["contactCount"], 1)


class RacketSwingTests(unittest.TestCase):
    def _racket_with_impact(self, contact_ms=1000, fps=30):
        """Racket head that accelerates into a 1000 ms contact and slows after."""
        racket = []
        x = 100.0
        step = 6.0
        dt = 1000 // fps
        for frame in range(40):
            t = frame * dt
            if 400 <= t <= 1000:
                step = 38.0
            elif t > 1000:
                step = 5.0
            x += step
            racket.append({"frame": frame, "timeMs": t, "xy": np.array([x, 150.0]), "confidence": 0.9})
        contact = {"frame": 30, "timeMs": contact_ms, "confidence": 0.9, "type": "contact", "source": "shuttle"}
        return racket, contact

    def test_swing_required_racket_acceleration_near_contact(self):
        racket, contact = self._racket_with_impact()
        swings = detect_racket_swings(racket, [contact], 800.0)
        self.assertGreaterEqual(len(swings), 1)
        self.assertEqual(swings[0]["type"], "racket_swing")
        self.assertEqual(swings[0]["source"], "racket")
        self.assertLessEqual(abs(swings[0]["timeMs"] - contact["timeMs"]), 400)

    def test_no_swing_without_confident_peak_motion(self):
        racket = [{"frame": i, "timeMs": i * 33, "xy": np.array([100.0, 150.0]), "confidence": 0.9} for i in range(40)]
        contact = {"frame": 30, "timeMs": 990, "confidence": 0.9, "type": "contact", "source": "shuttle"}
        self.assertEqual(detect_racket_swings(racket, [contact], 800.0), [])

    def test_swing_speed_metric_normalises_by_diagonal(self):
        racket, contact = self._racket_with_impact()
        swings = detect_racket_swings(racket, [contact], 800.0)
        metrics = racket_swing_speed_metrics(swings, racket, 800.0)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["metric"], "racket head swing speed")
        self.assertGreater(metrics[0]["value"], 0)
        self.assertEqual(metrics[0]["evidenceFrames"][0]["source"], "racket")


class SplitStepAmplitudeTests(unittest.TestCase):
    def _pose_with_dip(self, dip_frame=25):
        """Pose samples where the ankles dip before a 1000 ms contact rebound."""
        pose = []
        for frame in range(36):
            t = frame * 33
            dip = -6.0 if dip_frame - 3 <= frame <= dip_frame else 0.0
            ankle_y = 200.0 + dip
            keypoints = np.zeros((17, 2), dtype=np.float32)
            keypoints[:, 0] = np.arange(17, dtype=np.float32)
            keypoints[15, 1] = ankle_y
            keypoints[16, 1] = ankle_y
            scores = np.full(17, 0.8, dtype=np.float32)
            pose.append({"frame": frame, "timeMs": t, "confidence": 0.85, "keypoints": keypoints, "scores": scores})
        contact = {"timeMs": 1000, "frame": 30, "confidence": 0.9, "type": "contact", "source": "shuttle"}
        return pose, contact

    def test_split_step_amplitude_metric_from_ankle_dip(self):
        pose, contact = self._pose_with_dip()
        metrics = split_step_amplitude_metrics(pose, [contact], 800.0)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["metric"], "split-step amplitude")
        self.assertEqual(metrics[0]["unit"], "normalized frame diagonal")
        self.assertGreater(metrics[0]["value"], 0)
        self.assertLessEqual(metrics[0]["value"], 0.05)
        self.assertEqual(metrics[0]["evidenceFrames"][0]["source"], "pose")

    def test_no_amplitude_metric_without_a_contact(self):
        pose, contact = self._pose_with_dip()
        self.assertEqual(split_step_amplitude_metrics(pose, [], 800.0), [])


class ShotLandingSideTests(unittest.TestCase):
    def test_classified_shot_carries_landing_side(self):
        classified = classify_shot(
            {"confidence": 0.9},
            {"outboundSpeed": 900, "depth": "mid", "imageDy": 0, "afterCount": 3, "nearNet": False, "fromBack": False, "direction": "straight", "landingSide": "opponent_half", "beforeCount": 2},
            0,
        )
        self.assertEqual(classified["landingSide"], "opponent_half")


if __name__ == "__main__":
    unittest.main()
