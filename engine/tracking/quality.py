"""Track quality scoring."""

from __future__ import annotations

from engine.tracking.schemas import TrackAggregate


def compute_track_quality(track: TrackAggregate) -> float:
    if not track.observations:
        return 0.0

    frame_count = track.frame_count
    confidences = [obs.confidence for obs in track.observations]
    areas = [max(1.0, (obs.x2 - obs.x1) * (obs.y2 - obs.y1)) for obs in track.observations]

    length_score = min(1.0, frame_count / 15.0)
    confidence_score = sum(confidences) / len(confidences)

    expected_span = track.observations[-1].frame_index - track.observations[0].frame_index + 1
    continuity_score = frame_count / expected_span if expected_span > 0 else 0.0
    continuity_score = max(0.0, min(1.0, continuity_score))

    mean_area = sum(areas) / len(areas)
    variance = sum((a - mean_area) ** 2 for a in areas) / len(areas)
    coefficient_of_variation = (variance ** 0.5) / mean_area if mean_area > 0 else 1.0
    stability_score = max(0.0, 1.0 - min(1.0, coefficient_of_variation))
    gap_penalty = max(0.0, 1.0 - min(1.0, track.gap_count / max(1, frame_count)))

    score = (
        0.30 * length_score
        + 0.25 * confidence_score
        + 0.25 * continuity_score
        + 0.10 * stability_score
        + 0.10 * gap_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)
