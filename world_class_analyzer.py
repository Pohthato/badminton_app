"""
World-Class Badminton Analysis Engine
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ShotType(Enum):
    SMASH = "smash"
    CLEAR = "clear"
    DROP = "drop"
    NET_SHOT = "net_shot"
    LIFT = "lift"
    PUSH = "push"
    DRIVE = "drive"
    BLOCK = "block"
    SERVE = "serve"
    HALF_SMASH = "half_smash"
    JUMP_SMASH = "jump_smash"
    UNKNOWN = "unknown"


class StanceType(Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    OVERHEAD = "overhead"
    NET = "net"


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class SkeletonFrame:
    keypoints: List[Keypoint]
    frame_number: int
    timestamp: float
    confidence: float = 0.0


@dataclass
class StrokeEvent:
    frame_number: int
    timestamp: float
    shot_type: ShotType
    confidence: float
    contact_point: Optional[Tuple[float, float]] = None
    racket_speed: float = 0.0
    arm_angle: float = 0.0
    body_rotation: float = 0.0


@dataclass
class MovementSegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    movement_type: str
    avg_speed: float = 0.0
    max_speed: float = 0.0


class ShuttleTracker:
    def __init__(self, max_history=90):
        self.positions = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.confidences = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history)

    def add_detection(self, x, y, timestamp, confidence):
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
        self.confidences.append(confidence)
        if len(self.positions) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            if dt > 0:
                vx = (self.positions[-1][0] - self.positions[-2][0]) / dt
                vy = (self.positions[-1][1] - self.positions[-2][1]) / dt
                self.velocities.append((vx, vy))

    def get_speed(self):
        if not self.velocities: return 0.0
        vx, vy = self.velocities[-1]
        return np.sqrt(vx**2 + vy**2)

    def get_direction(self):
        if not self.velocities: return 0.0
        vx, vy = self.velocities[-1]
        return np.degrees(np.arctan2(vy, vx))

    def predict_landing(self):
        if len(self.positions) < 3 or not self.velocities: return None
        avx = np.mean([v[0] for v in list(self.velocities)[-5:]])
        avy = np.mean([v[1] for v in list(self.velocities)[-5:]])
        lx, ly = self.positions[-1]
        return (lx + avx * 10, ly + avy * 10)


class WorldClassAnalyzer:
    COURT_LENGTH = 13.4
    COURT_WIDTH = 6.1
    NET_HEIGHT = 1.55
    KP_NOSE = 0
    KP_LS = 5
    KP_RS = 6
    KP_LE = 7
    KP_RE = 8
    KP_LW = 9
    KP_RW = 10
    KP_LH = 11
    KP_RH = 12
    KP_LK = 13
    KP_RK = 14
    KP_LA = 15
    KP_RA = 16

    def __init__(self, court_homography=None):
        self.court_homography = court_homography
        self.skeleton_history = []
        self.shuttle_tracker = ShuttleTracker()
        self.stroke_events = []
        self.movement_segments = []
        self.frame_count = 0
        self.fps = 30.0
        self.player_pos_hist = []
        self.court_pos_hist = []
        self.vel_hist = []
        self.acc_hist = []
        self.balance_hist = []
        self.posture_scores = []
        self.stance_hist = []
        self.swing_hist = []
        self.strengths = []
        self.weaknesses = []
        self.mistakes = []
        self.recommendations = []
        self.drills = []

    def set_fps(self, fps): self.fps = fps
    def set_court_homography(self, h): self.court_homography = h

    def analyze_frame(self, kps, scores, shuttle_dets, bbox, fn, ts):
        self.frame_count = fn
        sk = self._parse_skeleton(kps, scores, fn, ts)
        self.skeleton_history.append(sk)
        bs = None
        if shuttle_dets:
            bs = max(shuttle_dets, key=lambda d: d.get("confidence", 0))
            self.shuttle_tracker.add_detection(bs["center_x"], bs["center_y"], ts, bs["confidence"])
        m = self._compute_metrics(sk, bs, bbox, fn, ts)
        st = self._detect_stroke(sk, m, fn, ts)
        if st: self.stroke_events.append(st)
        self._update_movement(m, fn, ts)
        self.posture_scores.append(self._assess_posture(sk))
        self.stance_hist.append(self._detect_stance(sk))
        self.balance_hist.append(self._assess_balance(sk))
        return m

    def _parse_skeleton(self, kps, scores, fn, ts):
        pts = []
        for i, (kp, sc) in enumerate(zip(kps, scores)):
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                pts.append(Keypoint(float(kp[0]), float(kp[1]), float(sc)))
            else:
                pts.append(Keypoint(0.0, 0.0, 0.0))
        ac = np.mean([k.confidence for k in pts]) if pts else 0.0
        return SkeletonFrame(pts, fn, ts, ac)

    def _compute_metrics(self, sk, shuttle, bbox, fn, ts):
        kps = sk.keypoints
        if len(kps) > 12:
            px = (kps[self.KP_LH].x + kps[self.KP_RH].x) / 2
            py = (kps[self.KP_LH].y + kps[self.KP_RH].y) / 2
            pos = (px, py)
        else:
            pos = (0, 0)
        self.player_pos_hist.append(pos)
        cp = None
        if self.court_homography and self.court_homography.is_calibrated:
            cpt = self.court_homography.map_pixel_to_court(np.array(pos))
            if cpt is not None:
                cp = (float(cpt[0]), float(cpt[1]))
                self.court_pos_hist.append(cp)
        vel = 0.0
        dire = 0.0
        if len(self.player_pos_hist) >= 2:
            dx = self.player_pos_hist[-1][0] - self.player_pos_hist[-2][0]
            dy = self.player_pos_hist[-1][1] - self.player_pos_hist[-2][1]
            dt = 1.0 / self.fps
            vel = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
            dire = np.degrees(np.arctan2(dy, dx))
        self.vel_hist.append(vel)
        acc = 0.0
        if len(self.vel_hist) >= 2:
            dt = 1.0 / self.fps
            acc = (self.vel_hist[-1] - self.vel_hist[-2]) / dt if dt > 0 else 0
        self.acc_hist.append(acc)
        ja = self._joint_angles(kps)
        rs = self._racket_swing(kps, fn)
        sd = None
        if shuttle:
            sd = np.sqrt((pos[0] - shuttle["center_x"])**2 + (pos[1] - shuttle["center_y"])**2)
        return {"frame": fn, "timestamp": ts, "position": pos, "court_position": cp,
                "velocity": vel, "acceleration": acc, "direction": dire,
                "joint_angles": ja, "racket_speed": rs, "shuttle_detected": shuttle is not None,
                "shuttle_distance": sd, "shuttle_speed": self.shuttle_tracker.get_speed() if shuttle else 0,
                "shuttle_direction": self.shuttle_tracker.get_direction() if shuttle else 0,
                "posture_score": self._assess_posture(sk), "balance_score": self._assess_balance(sk),
                "stance": self._detect_stance(sk).value}

    def _joint_angles(self, kps):
        a = {}
        if len(kps) > 10:
            a["right_elbow"] = self._angle3(kps[6], kps[8], kps[10])
            a["left_elbow"] = self._angle3(kps[5], kps[7], kps[9])
            a["right_knee"] = self._angle3(kps[12], kps[14], kps[16])
            a["left_knee"] = self._angle3(kps[11], kps[13], kps[15])
            a["trunk_lean"] = self._trunk_lean(kps)
            a["right_arm_ext"] = self._arm_ext(kps, "right")
            a["left_arm_ext"] = self._arm_ext(kps, "left")
        return a

    def _angle3(self, p1, p2, p3):
        a = np.array([p1.x - p2.x, p1.y - p2.y])
        b = np.array([p3.x - p2.x, p3.y - p2.y])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6: return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))

    def _trunk_lean(self, kps):
        if len(kps) < 13: return 0.0
        sx = (kps[5].x + kps[6].x) / 2
        sy = (kps[5].y + kps[6].y) / 2
        hx = (kps[11].x + kps[12].x) / 2
        hy = (kps[11].y + kps[12].y) / 2
        return float(np.degrees(np.arctan2(sx - hx, -(sy - hy))))

    def _arm_ext(self, kps, side):
        if len(kps) < 11: return 0.0
        s, e, w = (kps[6], kps[8], kps[10]) if side == "right" else (kps[5], kps[7], kps[9])
        up = np.sqrt((e.x - s.x)**2 + (e.y - s.y)**2)
        fa = np.sqrt((w.x - e.x)**2 + (w.y - e.y)**2)
        mr = up + fa
        act = np.sqrt((w.x - s.x)**2 + (w.y - s.y)**2)
        return float(act / mr) if mr > 0 else 0.0

    def _racket_swing(self, kps, fn):
        if len(kps) < 11 or len(self.skeleton_history) < 3: return 0.0
        cur = kps[10]
        pk = self.skeleton_history[-2].keypoints
        if len(pk) < 11: return 0.0
        prev = pk[10]
        dt = 1.0 / self.fps
        sp = np.sqrt((cur.x - prev.x)**2 + (cur.y - prev.y)**2) / dt
        self.swing_hist.append((fn, sp))
        return float(sp)

    def _detect_stroke(self, sk, m, fn, ts):
        if len(self.vel_hist) < 5 or len(self.swing_hist) < 5: return None
        rs = m.get("racket_speed", 0)
        ja = m.get("joint_angles", {})
        rae = ja.get("right_arm_ext", 0)
        re = ja.get("right_elbow", 90)
        tl = ja.get("trunk_lean", 0)
        spds = [s for _, s in self.swing_hist[-30:]]
        thr = np.percentile(spds, 85) if len(spds) > 10 else 50
        if rs < thr: return None
        st = ShotType.UNKNOWN
        cf = 0.5
        if rs > thr * 1.5 and rae > 0.85:
            st = ShotType.JUMP_SMASH if tl < -10 else ShotType.SMASH
            cf = 0.75 if tl < -10 else 0.7
        elif rs > thr * 1.2 and rae > 0.7:
            st = ShotType.CLEAR; cf = 0.6
        elif rs > thr and re > 120:
            st = ShotType.DROP; cf = 0.55
        elif rs > thr * 0.7 and rae > 0.6:
            st = ShotType.NET_SHOT; cf = 0.5
        elif rs > thr and abs(tl) < 10:
            st = ShotType.DRIVE; cf = 0.5
        elif rs > thr * 1.3 and rae > 0.75:
            st = ShotType.HALF_SMASH; cf = 0.55
        if st != ShotType.UNKNOWN:
            cpt = None
            if len(sk.keypoints) > 10:
                cpt = (sk.keypoints[10].x, sk.keypoints[10].y)
            return StrokeEvent(fn, ts, st, cf, cpt, rs, re, tl)
        return None

    def _update_movement(self, m, fn, ts):
        v = m.get("velocity", 0)
        if not self.movement_segments:
            self.movement_segments.append(MovementSegment(fn, fn, ts, ts, "idle"))
        cur = self.movement_segments[-1]
        cur.end_frame = fn
        cur.end_time = ts
        if v < 2: nt = "idle"
        elif v < 8: nt = "walking"
        elif v < 20: nt = "jogging"
        elif v < 40: nt = "running"
        else: nt = "sprinting"
        if nt != cur.movement_type:
            if fn > cur.start_frame:
                cur.avg_speed = float(np.mean(self.vel_hist[cur.start_frame:fn]))
                cur.max_speed = float(np.max(self.vel_hist[cur.start_frame:fn]))
            self.movement_segments.append(MovementSegment(fn, fn, ts, ts, nt))

    def _assess_posture(self, sk):
        k = sk.keypoints
        if len(k) < 13: return 0.0
        s = 0.0
        if k[0].y < k[5].y: s += 0.2
        if abs(k[5].y - k[6].y) < 20: s += 0.2
        if abs(k[11].y - k[12].y) < 15: s += 0.2
        if (k[5].y + k[6].y) / 2 < (k[11].y + k[12].y) / 2: s += 0.2
        if k[13].y > k[11].y or k[14].y > k[12].y: s += 0.2
        return min(s, 1.0)

    def _assess_balance(self, sk):
        k = sk.keypoints
        if len(k) < 17: return 0.5
        cx = (k[11].x + k[12].x) / 2
        bx = (k[15].x + k[16].x) / 2
        return float(max(0, 1 - abs(cx - bx) / 100))

    def _detect_stance(self, sk):
        k = sk.keypoints
        if len(k) < 13: return StanceType.NEUTRAL
        hah = k[9].y < k[0].y or k[10].y < k[0].y
        fw = abs(k[15].x - k[16].x)
        sw = abs(k[5].x - k[6].x)
        if hah: return StanceType.OVERHEAD
        elif fw > sw * 2.0: return StanceType.DEFENSIVE
        elif fw > sw * 1.5: return StanceType.NEUTRAL
        else: return StanceType.OFFENSIVE

    def generate_comprehensive_report(self):
        r = {"session_summary": self._sum_sess(), "stroke_analysis": self._sum_strokes(),
             "movement_analysis": self._sum_move(), "biomechanical_analysis": self._sum_bio(),
             "tactical_analysis": self._sum_tact(), "physical_analysis": self._sum_phys(),
             "technical_scores": self._tech_scores()}
        self._coach_insights(r)
        r["strengths"] = self.strengths or ["Solid fundamentals demonstrated"]
        r["weaknesses"] = self.weaknesses or ["Minor refinements possible"]
        r["mistakes"] = self.mistakes or ["No major technical mistakes"]
        r["recommendations"] = self.recommendations
        r["drills"] = self.drills
        return r

    def _sum_sess(self):
        dur = self.frame_count / self.fps if self.fps > 0 else 0
        av = float(np.mean(self.vel_hist)) if self.vel_hist else 0
        return {"total_frames": self.frame_count, "duration_seconds": dur,
                "duration_formatted": f"{int(dur // 60)}:{int(dur % 60):02d}",
                "total_strokes": len(self.stroke_events), "avg_speed": av,
                "max_speed": float(np.max(self.vel_hist)) if self.vel_hist else 0,
                "total_distance": float(sum(self.vel_hist) / self.fps) if self.vel_hist else 0,
                "avg_posture": float(np.mean(self.posture_scores)) if self.posture_scores else 0,
                "avg_balance": float(np.mean(self.balance_hist)) if self.balance_hist else 0,
                "court_coverage": self._court_cov(), "movement_breakdown": self._move_break()}

    def _sum_strokes(self):
        if not self.stroke_events: return {"total": 0, "distribution": {}, "avg_confidence": 0}
        dist = {}
        for s in self.stroke_events: dist[s.shot_type.value] = dist.get(s.shot_type.value, 0) + 1
        ac = float(np.mean([s.confidence for s in self.stroke_events]))
        qtys = []
        for s in self.stroke_events:
            q = 0.5
            if s.racket_speed > 50: q += 0.2
            if s.arm_angle > 140: q += 0.15
            if abs(s.body_rotation) < 20: q += 0.15
            qtys.append(min(q, 1.0))
        dm = (self.frame_count / self.fps / 60) if self.fps > 0 else 1
        return {"total": len(self.stroke_events), "distribution": dist,
                "avg_confidence": ac, "avg_quality": float(np.mean(qtys)),
                "shots_per_minute": len(self.stroke_events) / max(dm, 0.1),
                "most_common_shot": max(dist, key=dist.get) if dist else "none",
                "detailed_strokes": [{"frame": s.frame_number, "time": s.timestamp,
                    "type": s.shot_type.value, "confidence": s.confidence,
                    "racket_speed": s.racket_speed} for s in self.stroke_events]}

    def _sum_move(self):
        st = {}
        for seg in self.movement_segments: st[seg.movement_type] = st.get(seg.movement_type, 0) + 1
        active = sum(1 for v in self.vel_hist if v > 5)
        total = len(self.vel_hist)
        rec = self._recovery()
        return {"total_segments": len(self.movement_segments), "segment_breakdown": st,
                "court_coverage": self._court_cov(), "recovery_analysis": rec,
                "activity_ratio": float(active / total) if total > 0 else 0,
                "avg_acceleration": float(np.mean(np.abs(self.acc_hist))) if self.acc_hist else 0,
                "direction_changes": self._dir_changes(), "base_position": self._base_pos()}

    def _sum_bio(self):
        aa = {}
        for sk in self.skeleton_history[::30]:
            a = self._joint_angles(sk.keypoints)
            for k, v in a.items(): aa.setdefault(k, []).append(v)
        avg = {k: float(np.mean(v)) for k, v in aa.items()}
        rom = {k: {"min": float(min(v)), "max": float(max(v)), "range": float(max(v) - min(v))} for k, v in aa.items()}
        return {"average_joint_angles": avg, "range_of_motion": rom,
                "avg_posture_score": float(np.mean(self.posture_scores)) if self.posture_scores else 0,
                "avg_balance_score": float(np.mean(self.balance_hist)) if self.balance_hist else 0,
                "posture_consistency": float(1.0 - np.std(self.posture_scores)) if self.posture_scores else 0}

    def _sum_tact(self):
        if len(self.stroke_events) < 3: return {"pattern_detected": False}
        seq = [s.shot_type.value for s in self.stroke_events]
        from collections import Counter
        pats = [tuple(seq[i:i+3]) for i in range(len(seq) - 2)]
        pc = Counter(pats)
        mc = pc.most_common(3)
        off = sum(1 for s in self.stroke_events if s.shot_type in [ShotType.SMASH, ShotType.JUMP_SMASH, ShotType.NET_SHOT])
        deff = sum(1 for s in self.stroke_events if s.shot_type in [ShotType.CLEAR, ShotType.LIFT])
        return {"shot_sequence": seq, "patterns": [{"pattern": list(p), "count": c} for p, c in mc],
                "rallies": self._rallies(), "shot_variety": len(set(seq)) / len(ShotType),
                "offensive_ratio": off / max(len(self.stroke_events), 1),
                "defensive_ratio": deff / max(len(self.stroke_events), 1)}

    def _court_cov(self):
        if not self.court_pos_hist: return {"coverage_percentage": 0}
        x = [p[0] for p in self.court_pos_hist]
        y = [p[1] for p in self.court_pos_hist]
        if not x: return {"coverage_percentage": 0}
        xr = max(x) - min(x)
        yr = max(y) - min(y)
        ay = float(np.mean(y))
        pref = "front" if ay < self.COURT_LENGTH * 0.33 else ("mid" if ay < self.COURT_LENGTH * 0.66 else "back")
        return {"x_range": float(xr), "y_range": float(yr),
                "coverage_percentage": float(min(xr / self.COURT_WIDTH, 1) * min(yr / self.COURT_LENGTH, 1) * 100),
                "preferred_area": pref}

    def _move_break(self):
        if not self.vel_hist: return {}
        t = len(self.vel_hist)
        return {"stationary_pct": sum(1 for v in self.vel_hist if v < 2) / t * 100,
                "walking_pct": sum(1 for v in self.vel_hist if 2 <= v < 8) / t * 100,
                "running_pct": sum(1 for v in self.vel_hist if 8 <= v < 30) / t * 100,
                "sprinting_pct": sum(1 for v in self.vel_hist if v >= 30) / t * 100}

    def _recovery(self):
        if len(self.stroke_events) < 2: return {"avg_recovery_quality": 0.5}
        recs = []
        for i in range(len(self.stroke_events) - 1):
            c = self.stroke_events[i]
            n = self.stroke_events[i + 1]
            rt = n.timestamp - c.timestamp
            rq = 0.5
            if c.frame_number < len(self.player_pos_hist) and n.frame_number < len(self.player_pos_hist):
                bx = float(np.mean([p[0] for p in self.player_pos_hist[max(0, c.frame_number-30):c.frame_number]]))
                rps = [self.player_pos_hist[ff] for ff in range(c.frame_number, min(n.frame_number, len(self.player_pos_hist)))]
                if rps:
                    rq = float(max(0, 1 - np.mean([abs(p[0] - bx) for p in rps]) / 100))
            recs.append({"after_stroke": c.shot_type.value, "recovery_time": rt, "recovery_quality": rq})
        return {"avg_recovery_time": float(np.mean([r["recovery_time"] for r in recs])) if recs else 0,
                "avg_recovery_quality": float(np.mean([r["recovery_quality"] for r in recs])) if recs else 0.5,
                "recovery_patterns": recs}

    def _dir_changes(self):
        if len(self.player_pos_hist) < 3: return 0
        c = 0
        for i in range(2, len(self.player_pos_hist)):
            pdx = self.player_pos_hist[i-1][0] - self.player_pos_hist[i-2][0]
            cdx = self.player_pos_hist[i][0] - self.player_pos_hist[i-1][0]
            if pdx * cdx < 0: c += 1
        return c

    def _base_pos(self):
        if not self.court_pos_hist or not self.court_homography: return 0.5
        d = [np.sqrt((p[0] - self.COURT_WIDTH/2)**2 + (p[1] - self.COURT_LENGTH/2)**2) for p in self.court_pos_hist]
        if not d: return 0.5
        return float(max(0, 1 - np.mean(d) / np.sqrt((self.COURT_WIDTH/2)**2 + (self.COURT_LENGTH/2)**2)))

    def _rallies(self):
        if len(self.stroke_events) < 2: return []
        rallies = []
        cr = [self.stroke_events[0]]
        for i in range(1, len(self.stroke_events)):
            if self.stroke_events[i].timestamp - self.stroke_events[i-1].timestamp > 5.0:
                if len(cr) >= 2:
                    rallies.append({"stroke_count": len(cr), "duration": cr[-1].timestamp - cr[0].timestamp,
                                   "shot_types": [s.shot_type.value for s in cr]})
                cr = [self.stroke_events[i]]
            else:
                cr.append(self.stroke_events[i])
        if len(cr) >= 2:
            rallies.append({"stroke_count": len(cr), "duration": cr[-1].timestamp - cr[0].timestamp,
                           "shot_types": [s.shot_type.value for s in cr]})
        return rallies

    def _coach_insights(self, r):
        self.strengths = []
        self.weaknesses = []
        self.mistakes = []
        self.recommendations = []
        self.drills = []
        sc = r.get("technical_scores", {})
        st = r.get("stroke_analysis", {})
        mv = r.get("movement_analysis", {})
        ph = r.get("physical_analysis", {})
        ta = r.get("tactical_analysis", {})
        if sc.get("posture", 0) > 0.7: self.strengths.append("Excellent posture maintenance")
        if sc.get("balance", 0) > 0.7: self.strengths.append("Strong balance during strokes")
        if sc.get("footwork", 0) > 0.6: self.strengths.append("Good court coverage")
        if ta.get("shot_variety", 0) > 0.4: self.strengths.append("Good shot variety")
        if ph.get("explosive_movements", 0) > 5: self.strengths.append("Explosive movement capability")
        if st.get("avg_quality", 0) > 0.6: self.strengths.append("Solid stroke mechanics")
        if sc.get("posture", 0) < 0.5: self.weaknesses.append("Posture deteriorates during play")
        if sc.get("balance", 0) < 0.5: self.weaknesses.append("Balance issues during directional changes")
        if sc.get("recovery", 0) < 0.5: self.weaknesses.append("Slow recovery to base position")
        if ta.get("shot_variety", 0) < 0.2: self.weaknesses.append("Limited shot variety")
        if ta.get("offensive_ratio", 0) < 0.2: self.weaknesses.append("Not enough offensive pressure")
        if ph.get("fatigue_index", 0) > 0.4: self.weaknesses.append("Fatigue detected")
        if mv.get("activity_ratio", 0) < 0.3: self.weaknesses.append("Low movement activity")
        if self.stroke_events:
            lc = [s for s in self.stroke_events if s.confidence < 0.4]
            if len(lc) > len(self.stroke_events) * 0.3:
                self.mistakes.append("Many strokes show poor preparation")
        rec = mv.get("recovery_analysis", {})
        if rec.get("avg_recovery_quality", 1) < 0.4:
            self.mistakes.append("Caught out of position after attacking")
        if sc.get("tactical_awareness", 0) < 0.3:
            self.mistakes.append("Tactical pattern too predictable")
        if self.weaknesses:
            wl = " ".join(self.weaknesses).lower()
            if "recovery" in wl: self.recommendations.append("Practice split-step timing")
            if "posture" in wl: self.recommendations.append("Strengthen core exercises")
            if "variety" in wl or "predictable" in wl: self.recommendations.append("Develop 3-shot combinations")
            if "balance" in wl: self.recommendations.append("Add single-leg balance drills")
            if "fatigue" in wl or "conditioning" in wl: self.recommendations.append("Implement interval training")
        else:
            self.recommendations.append("Maintain training intensity")
            self.recommendations.append("Focus on match simulation")
        self.drills.append("Shadow badminton: 3 minutes continuous movement")
        self.drills.append("Multi-shuttle feeding: 20 shuttles")
        if any("offensive" in w.lower() for w in self.weaknesses):
            self.drills.append("Attack combination drills")
        if any("footwork" in w.lower() or "coverage" in w.lower() for w in self.weaknesses):
            self.drills.append("Six-corner footwork drills")
