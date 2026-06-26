"""Self-contained ByteTrack implementation for detection-to-track association."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.tracking.schemas import DetectionInput, TrackedDetection


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_cost(track_box: np.ndarray, det: DetectionInput) -> float:
    det_box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
    iou = _iou(track_box, det_box)
    iou_cost = 1.0 - iou

    tcx = (track_box[0] + track_box[2]) / 2.0
    tcy = (track_box[1] + track_box[3]) / 2.0
    dcx = (det_box[0] + det_box[2]) / 2.0
    dcy = (det_box[1] + det_box[3]) / 2.0
    tw = max(1.0, track_box[2] - track_box[0])
    th = max(1.0, track_box[3] - track_box[1])
    dist_cost = min(1.0, (((tcx - dcx) / tw) ** 2 + ((tcy - dcy) / th) ** 2) ** 0.5)

    return min(iou_cost, dist_cost)


def _linear_assignment(
    cost: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(cost)
    except ImportError:
        row_idx = np.arange(cost.shape[0])
        col_idx = np.argmin(cost, axis=1)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for r, c in zip(row_idx, col_idx):
        r_int, c_int = int(r), int(c)
        if cost[r_int, c_int] <= threshold and r_int not in matched_rows and c_int not in matched_cols:
            matches.append((r_int, c_int))
            matched_rows.add(r_int)
            matched_cols.add(c_int)

    unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


@dataclass
class _TrackState:
    track_id: int
    box: np.ndarray
    confidence: float
    frame_index: int
    timestamp_ms: float
    hits: int = 1
    time_since_update: int = 0
    is_lost: bool = False


@dataclass
class ByteTrackTracker:
    track_activation_threshold: float = 0.5
    low_threshold: float = 0.1
    match_threshold: float = 0.55
    lost_track_buffer: int = 30
    min_consecutive_frames: int = 1
    _next_id: int = 1
    _active: list[_TrackState] = field(default_factory=list)
    _lost: list[_TrackState] = field(default_factory=list)

    def update(self, frame_index: int, detections: list[DetectionInput]) -> list[TrackedDetection]:
        high = [d for d in detections if d.confidence >= self.track_activation_threshold]
        low = [
            d
            for d in detections
            if self.low_threshold <= d.confidence < self.track_activation_threshold
        ]

        tracked = [t for t in self._active if not t.is_lost]
        lost = list(self._lost)

        matched, unmatched_tracked, unmatched_high = self._associate(tracked, high)
        for t_idx, d_idx in matched:
            self._apply_detection(tracked[t_idx], high[d_idx], frame_index)

        unmatched_high_dets = [high[i] for i in unmatched_high]
        matched_lost, unmatched_lost, unmatched_high2 = self._associate(lost, unmatched_high_dets)
        for t_idx, d_idx in matched_lost:
            track = lost[t_idx]
            det = unmatched_high_dets[d_idx]
            self._apply_detection(track, det, frame_index)
            track.is_lost = False
            track.time_since_update = 0
            if track not in self._active:
                self._active.append(track)
            if track in self._lost:
                self._lost.remove(track)

        matched_low, _, _ = self._associate(
            [lost[i] for i in unmatched_lost],
            low,
        )
        for t_idx, d_idx in matched_low:
            track = lost[unmatched_lost[t_idx]]
            det = low[d_idx]
            self._apply_detection(track, det, frame_index)
            track.is_lost = False
            track.time_since_update = 0
            if track not in self._active:
                self._active.append(track)
            if track in self._lost:
                self._lost.remove(track)

        for d_idx in unmatched_high2:
            self._start_track(unmatched_high_dets[d_idx], frame_index)

        for t_idx in unmatched_tracked:
            track = tracked[t_idx]
            track.is_lost = True
            track.time_since_update += 1
            if track in self._active:
                self._active.remove(track)
            if track not in self._lost:
                self._lost.append(track)

        for track in list(self._lost):
            if track not in self._active:
                track.time_since_update += 1
            if track.time_since_update > self.lost_track_buffer:
                self._lost.remove(track)
                if track in self._active:
                    self._active.remove(track)

        output: list[TrackedDetection] = []
        for track in self._active:
            if not track.is_lost and track.hits >= self.min_consecutive_frames:
                output.append(self._to_tracked_detection(track, frame_index))
        return output

    def _associate(
        self,
        tracks: list[_TrackState],
        detections: list[DetectionInput],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost[i, j] = _match_cost(track.box, det)

        return _linear_assignment(cost, 1.0 - self.match_threshold)

    def _apply_detection(self, track: _TrackState, det: DetectionInput, frame_index: int) -> None:
        track.box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
        track.confidence = det.confidence
        track.frame_index = frame_index
        track.timestamp_ms = det.timestamp_ms
        track.hits += 1
        track.time_since_update = 0
        track.is_lost = False

    def _start_track(self, det: DetectionInput, frame_index: int) -> None:
        track = _TrackState(
            track_id=self._next_id,
            box=np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32),
            confidence=det.confidence,
            frame_index=frame_index,
            timestamp_ms=det.timestamp_ms,
        )
        self._next_id += 1
        self._active.append(track)

    @staticmethod
    def _to_tracked_detection(track: _TrackState, frame_index: int) -> TrackedDetection:
        return TrackedDetection(
            frame_index=frame_index,
            timestamp_ms=track.timestamp_ms,
            x1=float(track.box[0]),
            y1=float(track.box[1]),
            x2=float(track.box[2]),
            y2=float(track.box[3]),
            confidence=track.confidence,
            local_track_id=track.track_id,
        )
