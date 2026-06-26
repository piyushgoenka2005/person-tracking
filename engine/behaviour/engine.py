"""Sliding-window behaviour classification per track."""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timedelta, timezone

from engine.features.schemas import TrajectorySample
from engine.tracking.schemas import TrackAggregate, TrackedDetection

from engine.behaviour.features import extract_window_features
from engine.behaviour.rules import WindowSignals, classify_window
from engine.config import TimelineSettings
from engine.pose.keypoints import hand_near_mouth
from engine.types import BehaviourLabel, PoseResult


class BehaviourEngine:
    """Produces per-window behaviour labels from tracks, poses, and kinematics."""

    def __init__(self, settings: TimelineSettings) -> None:
        self._settings = settings

    def classify_tracks(
        self,
        tracks: list[TrackAggregate],
        poses: dict[tuple[int, int], PoseResult],
        *,
        video_fps: float,
    ) -> list[BehaviourLabel]:
        labels: list[BehaviourLabel] = []
        window_ms = self._settings.window_s * 1000.0
        hop_ms = self._settings.window_hop_s * 1000.0

        for track in tracks:
            if not track.observations:
                continue
            labels.extend(
                self._classify_single_track(
                    track,
                    poses,
                    window_ms=window_ms,
                    hop_ms=hop_ms,
                )
            )
        return labels

    def _classify_single_track(
        self,
        track: TrackAggregate,
        poses: dict[tuple[int, int], PoseResult],
        *,
        window_ms: float,
        hop_ms: float,
    ) -> list[BehaviourLabel]:
        observations = sorted(track.observations, key=lambda o: o.timestamp_ms)
        start_ms = observations[0].timestamp_ms
        end_ms = observations[-1].timestamp_ms

        raw_windows: list[tuple[float, float, WindowSignals]] = []
        cursor = start_ms
        while cursor + hop_ms <= end_ms:
            win_end = cursor + window_ms
            window_obs = [o for o in observations if cursor <= o.timestamp_ms < win_end]
            if len(window_obs) >= 1:
                signals = self._build_signals(track.local_track_id, window_obs, poses)
                raw_windows.append((cursor, win_end, signals))
            cursor += hop_ms

        if not raw_windows:
            return []

        speed_medians = [w[2].features.speed_mean for w in raw_windows]
        track_speed_baseline = statistics.median(speed_medians) if speed_medians else 1.0
        baseline = max(track_speed_baseline, 1.0)

        labels: list[BehaviourLabel] = []
        for win_start, win_end, signals in raw_windows:
            speed_ratio = signals.features.speed_mean / baseline
            adjusted = WindowSignals(
                features=signals.features,
                pose=signals.pose,
                bbox_height=signals.bbox_height,
                eating_cycles=signals.eating_cycles,
                speed_ratio=speed_ratio,
                queue_score=signals.queue_score,
            )
            behaviour, confidence = classify_window(adjusted, self._settings)
            labels.append(
                BehaviourLabel(
                    track_id=track.local_track_id,
                    start_ms=win_start,
                    end_ms=win_end,
                    behaviour=behaviour,
                    confidence=confidence,
                )
            )
        return labels

    def _build_signals(
        self,
        track_id: int,
        window_obs: list[TrackedDetection],
        poses: dict[tuple[int, int], PoseResult],
    ) -> WindowSignals:
        samples = self._to_trajectory_samples(window_obs, track_id)
        features = extract_window_features(
            window_obs,
            poses,
            track_id=track_id,
            samples=samples,
            settings=self._settings,
        )
        queue_score = self._estimate_queue_score(features)
        mid_obs = window_obs[len(window_obs) // 2]
        pose = poses.get((mid_obs.frame_index, track_id))
        bbox_h = max(mid_obs.y2 - mid_obs.y1, 1.0)
        eating_cycles = self._count_eating_cycles(window_obs, poses, track_id)

        return WindowSignals(
            features=features,
            pose=pose,
            bbox_height=bbox_h,
            eating_cycles=eating_cycles,
            queue_score=queue_score,
        )

    def _to_trajectory_samples(
        self, observations: list[TrackedDetection], track_id: int
    ) -> list[TrajectorySample]:
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        identity_id = uuid.UUID(int=0)
        tid = uuid.UUID(int=track_id & 0xFFFFFFFFFFFFFFFF)
        return [
            TrajectorySample(
                recorded_at=base + timedelta(milliseconds=obs.timestamp_ms),
                x=obs.centroid_x,
                y=obs.centroid_y,
                identity_id=identity_id,
                track_id=tid,
            )
            for obs in observations
        ]

    def _estimate_queue_score(self, features) -> float:
        cfg = self._settings
        if features.speed_mean > cfg.standing_speed_threshold * 2:
            return 0.0
        waiting = min(1.0, features.stop_duration_s / max(features.window_duration_s, 0.1))
        low_turn = 1.0 if features.turn_frequency < cfg.queue_turn_frequency_cap else 0.3
        low_var = 1.0 if features.trajectory_variance < cfg.trajectory_variance_threshold else 0.2
        return round(min(1.0, 0.35 * waiting + 0.35 * low_turn + 0.3 * low_var), 4)

    def _count_eating_cycles(
        self,
        observations: list[TrackedDetection],
        poses: dict[tuple[int, int], PoseResult],
        track_id: int,
    ) -> int:
        cycles = 0
        in_cycle = False
        thresh = self._settings.pose_keypoint_threshold
        for obs in observations:
            pose = poses.get((obs.frame_index, track_id))
            if pose is None:
                in_cycle = False
                continue
            bh = max(obs.y2 - obs.y1, 1.0)
            near = hand_near_mouth(
                pose.keypoints,
                pose.scores,
                threshold=thresh,
                mouth_ratio=self._settings.eating_hand_mouth_ratio,
                bbox_height=bh,
            )
            if near and not in_cycle:
                cycles += 1
                in_cycle = True
            elif not near:
                in_cycle = False
        return cycles
