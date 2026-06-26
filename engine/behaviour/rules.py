"""Pose and kinematic behaviour classification rules."""

from __future__ import annotations

from dataclasses import dataclass

from engine.behaviour.features import WindowFeatures
from engine.config import TimelineSettings
from engine.pose import keypoints as kp
from engine.types import PoseResult, TimelineBehaviour


@dataclass(frozen=True)
class WindowSignals:
    features: WindowFeatures
    pose: PoseResult | None
    bbox_height: float
    eating_cycles: int = 0
    speed_ratio: float = 1.0
    queue_score: float = 0.0


def classify_window(signals: WindowSignals, settings: TimelineSettings) -> tuple[TimelineBehaviour, float]:
    """Classify one 1-second sliding window using multi-signal scoring."""
    f = signals.features
    scores: dict[TimelineBehaviour, float] = {b: 0.0 for b in TimelineBehaviour}

    pose = signals.pose
    kp_thresh = settings.pose_keypoint_threshold
    pose_ok = pose is not None and f.pose_confidence >= kp_thresh

    if pose_ok and pose is not None:
        kps = pose.keypoints
        sc = pose.scores
        bh = max(signals.bbox_height, 1.0)

        if kp.wrist_near_face(kps, sc, threshold=kp_thresh, face_ratio=settings.phone_wrist_face_ratio, bbox_height=bh):
            elbow_l = kp.angle_at_joint(kps, sc, kp.LEFT_SHOULDER, kp.LEFT_ELBOW, kp.LEFT_WRIST, threshold=kp_thresh)
            elbow_r = kp.angle_at_joint(kps, sc, kp.RIGHT_SHOULDER, kp.RIGHT_ELBOW, kp.RIGHT_WRIST, threshold=kp_thresh)
            if (elbow_l is not None and elbow_l < 120) or (elbow_r is not None and elbow_r < 120):
                scores[TimelineBehaviour.PHONE_USAGE] += 0.8

        if signals.eating_cycles >= settings.eating_cycle_min:
            scores[TimelineBehaviour.EATING] += 0.75
        elif kp.hand_near_mouth(kps, sc, threshold=kp_thresh, mouth_ratio=settings.eating_hand_mouth_ratio, bbox_height=bh):
            scores[TimelineBehaviour.EATING] += 0.5

        torso = kp.torso_angle_deg(kps, sc, threshold=kp_thresh)
        knee_l = kp.angle_at_joint(kps, sc, kp.LEFT_HIP, kp.LEFT_KNEE, kp.LEFT_ANKLE, threshold=kp_thresh)
        knee_r = kp.angle_at_joint(kps, sc, kp.RIGHT_HIP, kp.RIGHT_KNEE, kp.RIGHT_ANKLE, threshold=kp_thresh)
        knee_angles = [a for a in (knee_l, knee_r) if a is not None]

        if torso is not None and torso >= settings.sleeping_torso_angle_deg and f.stop_duration_s > 1.0:
            scores[TimelineBehaviour.SLEEPING] += 0.7
        if knee_angles and min(knee_angles) < settings.sitting_knee_angle_deg and f.speed_mean < settings.walking_speed_threshold:
            scores[TimelineBehaviour.SITTING] += 0.65

    stop_ratio = f.stop_duration_s / max(f.window_duration_s, 0.1)
    relative_slow = signals.speed_ratio < 0.6
    moving = (
        f.speed_mean >= settings.walking_speed_threshold
        or signals.speed_ratio >= 1.1
        or (
            f.trajectory_variance >= settings.trajectory_variance_threshold
            and not relative_slow
        )
        or (
            f.pose_motion >= settings.pose_motion_threshold
            and signals.speed_ratio >= 0.75
        )
    )
    active_motion = (
        f.speed_std >= settings.speed_std_threshold
        or f.acceleration_std >= settings.acceleration_std_threshold
        or f.pose_motion >= settings.pose_motion_threshold * 0.6
    )

    if moving and active_motion and (signals.speed_ratio >= 0.7 or f.speed_mean >= settings.walking_speed_threshold):
        walk_strength = min(1.0, 0.4 + signals.speed_ratio * 0.25 + min(f.trajectory_variance / 200.0, 0.2))
        scores[TimelineBehaviour.WALKING] += 0.55 + walk_strength * 0.35

    if relative_slow and stop_ratio >= 0.3 and f.trajectory_variance < settings.trajectory_variance_threshold * 1.5:
        scores[TimelineBehaviour.STANDING] += 0.45 + min(stop_ratio, 0.25)

    if signals.queue_score >= settings.queue_score_threshold and f.speed_mean < settings.walking_speed_threshold:
        scores[TimelineBehaviour.QUEUEING] += 0.5 + signals.queue_score * 0.4

    if (
        f.speed_mean < settings.standing_speed_threshold
        and stop_ratio >= settings.waiting_stop_ratio
        and f.dwell_duration_s >= settings.waiting_dwell_s * 0.5
        and scores[TimelineBehaviour.QUEUEING] < 0.45
    ):
        scores[TimelineBehaviour.WAITING] += 0.55 + min(stop_ratio, 0.35)

    if (
        f.speed_mean < settings.standing_speed_threshold
        and stop_ratio >= 0.25
        and f.trajectory_variance < settings.trajectory_variance_threshold
        and f.pose_motion < settings.pose_motion_threshold
    ):
        scores[TimelineBehaviour.STANDING] += 0.5 + min(stop_ratio, 0.3)

    if not moving and f.speed_mean < settings.standing_speed_threshold * 1.5:
        scores[TimelineBehaviour.STANDING] += 0.25

    best = max(scores, key=scores.get)
    best_score = scores[best]
    total = sum(scores.values()) or 1.0
    confidence = round(min(1.0, best_score / total), 4)

    if best_score < settings.behaviour_confidence_threshold:
        if stop_ratio >= 0.4 and (relative_slow or f.speed_mean < settings.standing_speed_threshold * 2):
            return TimelineBehaviour.STANDING, max(confidence, 0.35)
        if moving and active_motion and signals.speed_ratio >= 0.75:
            return TimelineBehaviour.WALKING, max(confidence, 0.35)
        if stop_ratio >= 0.25:
            return TimelineBehaviour.STANDING, max(confidence, 0.3)
        return TimelineBehaviour.UNKNOWN, confidence

    return best, confidence
