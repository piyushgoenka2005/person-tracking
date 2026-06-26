"""COCO-17 keypoint geometry helpers for behaviour rules."""

from __future__ import annotations

import math

# COCO-17 indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def _valid(kps: list[tuple[float, float]], scores: list[float], idx: int, threshold: float) -> bool:
    if idx >= len(kps) or idx >= len(scores):
        return False
    return scores[idx] >= threshold


def _point(kps: list[tuple[float, float]], idx: int) -> tuple[float, float]:
    return kps[idx]


def angle_at_joint(
    kps: list[tuple[float, float]],
    scores: list[float],
    a: int,
    b: int,
    c: int,
    *,
    threshold: float = 0.3,
) -> float | None:
    """Angle at joint b formed by segments ba and bc (degrees)."""
    if not all(_valid(kps, scores, i, threshold) for i in (a, b, c)):
        return None
    ax, ay = _point(kps, a)
    bx, by = _point(kps, b)
    cx, cy = _point(kps, c)
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    m1 = math.hypot(v1x, v1y)
    m2 = math.hypot(v2x, v2y)
    if m1 < 1e-6 or m2 < 1e-6:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos_angle))


def torso_angle_deg(
    kps: list[tuple[float, float]],
    scores: list[float],
    *,
    threshold: float = 0.3,
) -> float | None:
    """Angle of shoulder-mid to hip-mid vector from vertical (degrees)."""
    if not (
        _valid(kps, scores, LEFT_SHOULDER, threshold)
        and _valid(kps, scores, RIGHT_SHOULDER, threshold)
        and _valid(kps, scores, LEFT_HIP, threshold)
        and _valid(kps, scores, RIGHT_HIP, threshold)
    ):
        return None
    sx = (_point(kps, LEFT_SHOULDER)[0] + _point(kps, RIGHT_SHOULDER)[0]) / 2.0
    sy = (_point(kps, LEFT_SHOULDER)[1] + _point(kps, RIGHT_SHOULDER)[1]) / 2.0
    hx = (_point(kps, LEFT_HIP)[0] + _point(kps, RIGHT_HIP)[0]) / 2.0
    hy = (_point(kps, LEFT_HIP)[1] + _point(kps, RIGHT_HIP)[1]) / 2.0
    dx, dy = hx - sx, hy - sy
    if math.hypot(dx, dy) < 1e-6:
        return None
    # Angle from vertical (y-axis down in image coords)
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


def wrist_near_face(
    kps: list[tuple[float, float]],
    scores: list[float],
    *,
    threshold: float = 0.3,
    face_ratio: float = 0.35,
    bbox_height: float,
) -> bool:
    """True when either wrist is near the head region."""
    head_refs = [NOSE, LEFT_EAR, RIGHT_EAR]
    head_pts = [_point(kps, i) for i in head_refs if _valid(kps, scores, i, threshold)]
    if not head_pts:
        return False
    hx = sum(p[0] for p in head_pts) / len(head_pts)
    hy = sum(p[1] for p in head_pts) / len(head_pts)
    max_dist = max(bbox_height * face_ratio, 20.0)
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        if not _valid(kps, scores, wrist, threshold):
            continue
        wx, wy = _point(kps, wrist)
        if math.hypot(wx - hx, wy - hy) <= max_dist:
            return True
    return False


def hand_near_mouth(
    kps: list[tuple[float, float]],
    scores: list[float],
    *,
    threshold: float = 0.3,
    mouth_ratio: float = 0.25,
    bbox_height: float,
) -> bool:
    if not _valid(kps, scores, NOSE, threshold):
        return False
    nx, ny = _point(kps, NOSE)
    max_dist = max(bbox_height * mouth_ratio, 15.0)
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        if not _valid(kps, scores, wrist, threshold):
            continue
        wx, wy = _point(kps, wrist)
        if math.hypot(wx - nx, wy - ny) <= max_dist:
            return True
    return False


def mean_keypoint_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
