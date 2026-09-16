"""Evidence-gated badminton event layer.

Contact, rally, shot, split-step, and recovery outputs are produced only from
observed tracks. Labels below the confidence gate stay unverified and are
excluded from shot counts used for coaching.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

COURT_LENGTH_M = 13.40
SINGLES_WIDTH_M = 5.18
DOUBLES_WIDTH_M = 6.10
NET_Y_M = COURT_LENGTH_M / 2
MIN_CONTACT_GAP_MS = 280
SHOT_LABELS = ("smash", "clear", "drop", "net", "lift", "drive", "push", "serve")
VERIFIED_SHOT_CONFIDENCE = 0.62


def court_width_m(court_type: str) -> float:
    return SINGLES_WIDTH_M if court_type == "singles" else DOUBLES_WIDTH_M


def court_center(court_type: str, side: str) -> np.ndarray:
    width = court_width_m(court_type)
    base_y = 3.20 if side == "near" else COURT_LENGTH_M - 3.20
    return np.array([width / 2, base_y], dtype=np.float32)


def _velocity(points: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    output = []
    for index in range(1, len(points)):
        dt = (points[index][0] - points[index - 1][0]) / 1000
        if dt <= 0:
            continue
        output.append((points[index][0], (points[index][1] - points[index - 1][1]) / dt))
    return output


def detect_contacts(
    shuttle: list[dict[str, Any]],
    pose: list[dict[str, Any]],
    racket: list[dict[str, Any]],
    frame_diagonal: float,
) -> list[dict[str, Any]]:
    if len(shuttle) < 3:
        return []
    timed = [(item["timeMs"], np.asarray(item["xy"], dtype=np.float32), item) for item in shuttle]
    velocities = _velocity([(time_ms, xy) for time_ms, xy, _ in timed])
    if len(velocities) < 2:
        return []
    pose_by_time = {item["timeMs"]: item for item in pose}
    racket_by_time = {item["timeMs"]: item for item in racket}
    contacts: list[dict[str, Any]] = []
    last_ms = -10_000

    for index in range(1, len(velocities)):
        time_ms, velocity = velocities[index]
        previous = velocities[index - 1][1]
        speed = float(np.linalg.norm(velocity))
        previous_speed = float(np.linalg.norm(previous))
        # Thresholds are expressed relative to the frame diagonal so they stay
        # meaningful across phone resolutions and camera distances. A contact
        # is a direction reversal of a fast shuttle, or a sharp speed collapse
        # (impact) from a shuttle that was moving at real rally speed.
        reversal = float(np.dot(previous, velocity)) < 0 and previous_speed > frame_diagonal * 1.2
        collapse = previous_speed > frame_diagonal * 0.9 and speed < previous_speed * 0.45
        if not (reversal or collapse):
            continue
        if time_ms - last_ms < MIN_CONTACT_GAP_MS:
            continue
        sample = next((item for item_time, _, item in timed if item_time == time_ms), timed[min(index + 1, len(timed) - 1)][2])
        proximity = 0.0
        racket_sample = racket_by_time.get(time_ms)
        if racket_sample is not None:
            distance = float(np.linalg.norm(np.asarray(sample["xy"]) - np.asarray(racket_sample["xy"])))
            if distance < frame_diagonal * 0.12:
                proximity = max(proximity, 0.85 * racket_sample["confidence"])
        pose_sample = pose_by_time.get(time_ms)
        if pose_sample is not None:
            wrists = []
            keypoints, scores = pose_sample["keypoints"], pose_sample["scores"]
            for joint in (9, 10):
                if scores[joint] >= 0.4:
                    wrists.append(keypoints[joint])
            if wrists:
                distance = min(float(np.linalg.norm(np.asarray(sample["xy"]) - wrist)) for wrist in wrists)
                if distance < frame_diagonal * 0.14:
                    proximity = max(proximity, 0.7 * pose_sample["confidence"])
        if proximity < 0.35 and not collapse:
            continue
        confidence = float(np.clip(0.35 + 0.4 * sample["confidence"] + 0.35 * proximity, 0, 1))
        if confidence < 0.55:
            continue
        contacts.append({
            "type": "contact",
            "frame": sample["frame"],
            "timeMs": time_ms,
            "confidence": round(confidence, 3),
            "source": "shuttle",
        })
        last_ms = time_ms
    return contacts


def segment_rallies(contacts: list[dict[str, Any]], shuttle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not contacts:
        return []
    gap_ms = 1600
    rallies = []
    current = [contacts[0]]
    for contact in contacts[1:]:
        if contact["timeMs"] - current[-1]["timeMs"] > gap_ms:
            rallies.append(current)
            current = [contact]
        else:
            current.append(contact)
    rallies.append(current)
    shuttle_times = [item["timeMs"] for item in shuttle]
    output = []
    for index, group in enumerate(rallies):
        start = max(0, group[0]["timeMs"] - 400)
        end = group[-1]["timeMs"] + 500
        if shuttle_times:
            nearby = [time for time in shuttle_times if start - 200 <= time <= end + 800]
            if nearby:
                end = max(end, nearby[-1])
        output.append({
            "index": index,
            "startMs": start,
            "endMs": end,
            "contactCount": len(group),
            "contacts": group,
        })
    return output


def _shot_geometry(contact: dict[str, Any], shuttle: list[dict[str, Any]], court_type: str, has_court: bool) -> dict[str, Any]:
    after = [item for item in shuttle if contact["timeMs"] < item["timeMs"] <= contact["timeMs"] + 900]
    before = [item for item in shuttle if contact["timeMs"] - 240 <= item["timeMs"] <= contact["timeMs"]]
    direction, depth, landing_side = "unknown", "unknown", "unclear"
    outbound_speed = 0.0
    image_dy = 0.0
    if len(after) >= 2:
        delta = np.asarray(after[-1]["xy"], dtype=np.float32) - np.asarray(after[0]["xy"], dtype=np.float32)
        dt = (after[-1]["timeMs"] - after[0]["timeMs"]) / 1000
        if dt > 0:
            outbound_speed = float(np.linalg.norm(delta) / dt)
            image_dy = float(delta[1] / dt)
    court_points = [item.get("court") for item in after if item.get("court") is not None]
    if has_court and court_points:
        # A mid-air shuttle is not on the floor plane, so ground-plane
        # projection is biased. We only use the observed cloud's final
        # position and treat "side of the net" conservatively: positions
        # deep beyond the centre line mean the shuttle crossed the net.
        mean_y = float(np.mean([float(point[1]) for point in court_points]))
        half_gap = 0.45
        if mean_y > NET_Y_M + half_gap:
            landing_side = "opponent_half"
        elif mean_y < NET_Y_M - half_gap:
            landing_side = "own_half"
        else:
            landing_side = "unclear"
    if has_court and len(court_points) >= 2:
        start, end = np.asarray(court_points[0]), np.asarray(court_points[-1])
        dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
        direction = "cross" if abs(dx) >= 1.15 else "straight"
        landing_y = float(end[1])
        opponent_front = NET_Y_M + 0.4
        if dy >= 0:
            if landing_y < opponent_front + 1.8:
                depth = "short"
            elif landing_y < COURT_LENGTH_M - 2.3:
                depth = "mid"
            else:
                depth = "back"
        else:
            if landing_y > NET_Y_M - 1.8:
                depth = "short"
            elif landing_y > 2.3:
                depth = "mid"
            else:
                depth = "back"
    return {
        "direction": direction,
        "depth": depth,
        "landingSide": landing_side,
        "outboundSpeed": outbound_speed,
        "imageDy": image_dy,
        "beforeCount": len(before),
        "afterCount": len(after),
        "nearNet": bool(has_court and court_points and abs(float(court_points[0][1]) - NET_Y_M) < 1.6),
        "fromBack": bool(has_court and court_points and (float(court_points[0][1]) < 3.4 or float(court_points[0][1]) > COURT_LENGTH_M - 3.4)),
    }


def classify_shot(contact: dict[str, Any], geometry: dict[str, Any], rally_index_in_rally: int) -> dict[str, Any]:
    speed, depth, image_dy = geometry["outboundSpeed"], geometry["depth"], geometry["imageDy"]
    label = "unverified"
    confidence = 0.35
    if geometry["afterCount"] < 2:
        return {"label": label, "confidence": confidence, "verified": False, **geometry}
    if rally_index_in_rally == 0 and geometry["fromBack"] and speed < 2200:
        label, confidence = "serve", 0.66
    elif geometry["nearNet"] and speed < 1400 and depth in ("short", "unknown"):
        label, confidence = "net", 0.68
    elif depth == "short" and speed < 1600:
        label, confidence = "drop", 0.64
    elif depth == "back" and speed >= 1400 and image_dy <= 400:
        label, confidence = "clear", 0.67
    elif speed >= 2400 and image_dy > 350:
        label, confidence = "smash", 0.7
    elif speed >= 1800 and depth in ("mid", "unknown"):
        label, confidence = "drive", 0.63
    elif geometry["nearNet"] and depth == "back":
        label, confidence = "lift", 0.64
    elif speed < 1200:
        label, confidence = "push", 0.6
    verified = confidence >= VERIFIED_SHOT_CONFIDENCE and label in SHOT_LABELS
    if not verified:
        label = "unverified"
    return {
        "label": label,
        "confidence": round(min(confidence, contact["confidence"]), 3),
        "verified": verified,
        "direction": geometry["direction"],
        "depth": geometry["depth"],
        "landingSide": geometry["landingSide"],
    }


def classify_shots(
    rallies: list[dict[str, Any]],
    shuttle: list[dict[str, Any]],
    court_type: str,
    has_court: bool,
) -> list[dict[str, Any]]:
    shots = []
    for rally in rallies:
        for index, contact in enumerate(rally["contacts"]):
            geometry = _shot_geometry(contact, shuttle, court_type, has_court)
            classified = classify_shot(contact, geometry, index)
            shots.append({
                "frame": contact["frame"],
                "timeMs": contact["timeMs"],
                "confidence": classified["confidence"],
                "label": classified["label"],
                "verified": classified["verified"],
                "direction": classified["direction"],
                "depth": classified["depth"],
                "landingSide": classified["landingSide"],
                "rallyIndex": rally["index"],
            })
    return shots


def detect_split_steps(pose: list[dict[str, Any]], contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(pose) < 5 or not contacts:
        return []
    events = []
    for contact in contacts:
        window = [item for item in pose if contact["timeMs"] - 480 <= item["timeMs"] <= contact["timeMs"] - 80]
        if len(window) < 4:
            continue
        ankle_y = []
        for item in window:
            scores = item["scores"]
            if min(scores[15], scores[16]) < 0.45:
                continue
            ankle_y.append((item["timeMs"], item["frame"], float((item["keypoints"][15][1] + item["keypoints"][16][1]) / 2), item["confidence"]))
        if len(ankle_y) < 4:
            continue
        values = np.array([item[2] for item in ankle_y])
        dip_index = int(np.argmin(values))
        if dip_index == 0 or dip_index == len(values) - 1:
            continue
        drop = float(values[0] - values[dip_index])
        rebound = float(values[-1] - values[dip_index])
        if drop < 1.2 or rebound < 0.8:
            continue
        sample = ankle_y[dip_index]
        events.append({
            "type": "split_step",
            "frame": sample[1],
            "timeMs": sample[0],
            "confidence": round(float(np.clip(0.55 + 0.25 * sample[3], 0, 0.92)), 3),
            "source": "pose",
        })
    return events


SWING_TIME_WINDOW_MS = 700
SWING_PEAK_TOLERANCE_MS = 260


def detect_racket_swings(
    racket: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    frame_diagonal: float,
) -> list[dict[str, Any]]:
    """Racket-head speed profile around a contact.

    A hit produces an acceleration into impact and a deceleration after it
    (through-swing). We require a confident peak close to the contact time and
    either a hard post-impact deceleration or a clear acceleration before it.
    Detection only returns evidence-gated events; no swing claim is made when
    the racket layer is unavailable or occluded.
    """
    if len(racket) < 4 or not contacts or frame_diagonal <= 0:
        return []
    timed = sorted(
        ((item["timeMs"], np.asarray(item["xy"], dtype=np.float32), item) for item in racket),
        key=lambda entry: entry[0],
    )
    if len(timed) < 4:
        return []
    speeds: list[tuple[float, float, int]] = []
    for sensor_idx in range(1, len(timed)):
        dt = (timed[sensor_idx][0] - timed[sensor_idx - 1][0]) / 1000
        if dt <= 0:
            continue
        distance = float(np.linalg.norm(timed[sensor_idx][1] - timed[sensor_idx - 1][1]))
        speeds.append((timed[sensor_idx][0], distance / dt, sensor_idx))
    if len(speeds) < 3:
        return []
    median_speed = float(np.median([speed for _, speed, _ in speeds])) or 0.0
    min_peak = max(frame_diagonal * 1.2, median_speed * 2.0)
    events: list[dict[str, Any]] = []
    for contact in contacts:
        window = [
            (time_ms, speed, sensor_idx)
            for time_ms, speed, sensor_idx in speeds
            if contact["timeMs"] - SWING_TIME_WINDOW_MS <= time_ms <= contact["timeMs"] + SWING_TIME_WINDOW_MS
        ]
        if len(window) < 3:
            continue
        # A fast smash produces a speed plateau (backswing + impact) followed by
        # a through-swing deceleration. The peak is the LAST sample still at
        # plateau speed, which sits closest to the actual contact.
        peak_time: float | None = None
        peak_speed = 0.0
        for sensor_index, (time_ms, speed, _unused_sensor) in enumerate(window):
            if speed < min_peak:
                continue
            ahead = window[sensor_index + 1: sensor_index + 4]
            plateau_or_dropping = bool(ahead and max(entry[1] for entry in ahead) <= speed)
            if plateau_or_dropping and speed >= peak_speed:
                peak_speed = speed
                peak_time = time_ms
        if peak_time is None:
            continue
        if abs(peak_time - contact["timeMs"]) > SWING_PEAK_TOLERANCE_MS:
            continue
        before = [entry for entry in window if entry[0] < peak_time]
        after = [entry for entry in window if entry[0] > peak_time][:3]
        if not before:
            continue
        rise = peak_speed - float(np.mean([entry[1] for entry in before[-3:]]))
        decayed = bool(after and float(np.mean([entry[1] for entry in after])) < peak_speed * 0.55)
        if rise < frame_diagonal * 0.9 and not decayed:
            continue
        sample = next((item for time_ms, _, item in timed if time_ms == peak_time), None)
        confidence = float(np.clip(0.5 + 0.3 * peak_speed / min_peak, 0, 0.92))
        if confidence < 0.6:
            continue
        events.append({
            "type": "racket_swing",
            "frame": sample["frame"] if sample else 0,
            "timeMs": peak_time,
            "confidence": round(confidence, 3),
            "source": "racket",
        })
    return events


def split_step_amplitude_metrics(
    pose: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    frame_diagonal: float,
) -> list[dict[str, Any]]:
    """Mean ankle drop of a split step, normalised by frame diagonal.

    Split step is explosive footwork: the leading ankle dips before contact.
    Amplitude here is image-space only (no metre claim) and is gated on the
    same rebound signature used for split-step events.
    """
    if len(pose) < 5 or not contacts or frame_diagonal <= 0:
        return []
    dips: list[tuple[float, dict[str, Any]]] = []
    for contact in contacts:
        window = [item for item in pose if contact["timeMs"] - 480 <= item["timeMs"] <= contact["timeMs"] - 80]
        if len(window) < 4:
            continue
        ankle_y: list[tuple[float, int, float, float]] = []
        for item in window:
            scores = item["scores"]
            if min(scores[15], scores[16]) < 0.45:
                continue
            ankle_y.append((item["timeMs"], item["frame"], float((item["keypoints"][15][1] + item["keypoints"][16][1]) / 2), item["confidence"]))
        if len(ankle_y) < 4:
            continue
        values = np.array([entry[2] for entry in ankle_y])
        dip_index = int(np.argmin(values))
        if dip_index == 0 or dip_index == len(values) - 1:
            continue
        amplitude = float(values[0] - values[dip_index])
        rebound = float(values[-1] - values[dip_index])
        if amplitude < 1.2 or rebound < 0.8:
            continue
        sample = ankle_y[dip_index]
        dips.append((amplitude, {"frame": sample[1], "timeMs": sample[0], "confidence": sample[3]}))
    if not dips:
        return []
    evidence = [{"frame": item[1]["frame"], "timeMs": item[1]["timeMs"], "confidence": round(float(item[1]["confidence"]), 3), "source": "pose"} for item in dips[:12]]
    confidence = float(np.clip(np.mean([item[1]["confidence"] for item in dips]), 0, 1))
    return [{
        "metric": "split-step amplitude",
        "value": round(float(np.mean([amplitude / frame_diagonal for amplitude, _ in dips])), 3),
        "unit": "normalized frame diagonal",
        "direction": "higher_is_better",
        "confidence": round(confidence, 3),
        "evidenceFrames": evidence,
    }]


def racket_swing_speed_metrics(
    swings: list[dict[str, Any]],
    racket: list[dict[str, Any]],
    frame_diagonal: float,
) -> list[dict[str, Any]]:
    """Peak racket-head speed at each detected swing, normalised per second."""
    if not swings or len(racket) < 3 or frame_diagonal <= 0:
        return []
    timed: dict[float, np.ndarray] = {}
    for item in racket:
        timed[item["timeMs"]] = np.asarray(item["xy"], dtype=np.float32)
    ordered = sorted(timed.items())
    speed_at: dict[float, float] = {}
    for index in range(1, len(ordered)):
        t0, point0 = ordered[index - 1]
        t1, point1 = ordered[index]
        dt = (t1 - t0) / 1000
        if dt > 0:
            speed_at[t1] = float(np.linalg.norm(point1 - point0)) / dt
    if not speed_at:
        return []
    values: list[float] = []
    evidence: list[dict[str, Any]] = []
    for swing in swings:
        peak = max(
            (speed_at[key] for key in speed_at if abs(key - swing["timeMs"]) <= 240),
            default=None,
        )
        if peak is None:
            continue
        values.append(peak / frame_diagonal)
        evidence.append({"frame": swing["frame"], "timeMs": swing["timeMs"], "confidence": swing["confidence"], "source": "racket"})
    if not values:
        return []
    return [{
        "metric": "racket head swing speed",
        "value": round(float(np.mean(values)), 3),
        "unit": "frame diagonals per second",
        "direction": "higher_is_better",
        "confidence": round(float(np.clip(np.mean([entry["confidence"] for entry in evidence]), 0, 1)), 3),
        "evidenceFrames": evidence[:12],
    }]


def detect_recoveries(
    pose: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    court_type: str,
    side: str,
    has_court: bool,
) -> list[dict[str, Any]]:
    if not has_court or not contacts:
        return []
    base = court_center(court_type, side)
    events = []
    for contact in contacts:
        window = [item for item in pose if contact["timeMs"] < item["timeMs"] <= contact["timeMs"] + 1600 and item.get("court") is not None]
        recovered = next((item for item in window if float(np.linalg.norm(np.asarray(item["court"]) - base)) <= 1.45), None)
        if recovered is None:
            continue
        events.append({
            "type": "recovery_complete",
            "frame": recovered["frame"],
            "timeMs": recovered["timeMs"],
            "confidence": round(float(np.clip(recovered["confidence"], 0, 1)), 3),
            "source": "court",
        })
    return events


def shot_distribution(shots: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shot in shots:
        if not shot.get("verified"):
            continue
        counts[shot["label"]] = counts.get(shot["label"], 0) + 1
    return counts


def rally_summaries(rallies: list[dict[str, Any]], shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rally: dict[int, list[dict[str, Any]]] = {}
    for shot in shots:
        by_rally.setdefault(shot["rallyIndex"], []).append(shot)
    return [
        {
            "index": rally["index"],
            "startMs": rally["startMs"],
            "endMs": rally["endMs"],
            "contactCount": rally["contactCount"],
            "verifiedShots": sum(1 for shot in by_rally.get(rally["index"], []) if shot.get("verified")),
        }
        for rally in rallies
    ]


def capture_warnings(source_fps: float, duration_s: float, shuttle_ratio: float, pose_ratio: float) -> list[str]:
    notes = []
    if source_fps < 60:
        notes.append(f"Source is {source_fps:.0f} fps. Overhead smashes need 60 fps or higher; 120 fps slow-motion capture gives reliable contact timing and swing speed evidence.")
    if source_fps < 45:
        notes.append("Below 45 fps the shuttle and racket paths are under-sampled. Racket swing speed and landing-side labels are withheld or low-confidence.")
    if duration_s > 90:
        notes.append("Clip is longer than 90 seconds. Rally-length uploads produce better contact timing and lower GPU cost.")
    if shuttle_ratio < 0.18:
        notes.append("Shuttle evidence is sparse. Use a rear or side court view with the shuttle visible against a stable background.")
    if pose_ratio < 0.45:
        notes.append("Selected-player pose was often missing. Keep the athlete fully in frame and avoid digital zoom.")
    return notes


def build_event_layer(
    shuttle: list[dict[str, Any]],
    pose: list[dict[str, Any]],
    racket: list[dict[str, Any]],
    frame_diagonal: float,
    court_type: str,
    side: str,
    has_court: bool,
) -> dict[str, Any]:
    contacts = detect_contacts(shuttle, pose, racket, frame_diagonal)
    rallies = segment_rallies(contacts, shuttle)
    shots = classify_shots(rallies, shuttle, court_type, has_court)
    split_steps = detect_split_steps(pose, contacts)
    recoveries = detect_recoveries(pose, contacts, court_type, side, has_court)
    swings = detect_racket_swings(racket, contacts, frame_diagonal)
    events = [*contacts, *split_steps, *recoveries, *swings]
    events.sort(key=lambda item: item["timeMs"])
    metrics = [
        *split_step_amplitude_metrics(pose, contacts, frame_diagonal),
        *racket_swing_speed_metrics(swings, racket, frame_diagonal),
    ]
    return {
        "events": events,
        "rallies": rally_summaries(rallies, shots),
        "shots": shots,
        "shotDistribution": shot_distribution(shots),
        "metrics": metrics,
    }
