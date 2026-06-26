"""Pipeline stage and output validation."""

from __future__ import annotations

from pathlib import Path

from engine.types import PoseResult, TimelineBehaviour, TimelineSegment


class QAReport:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_detection(detection_frames: int, person_frames: int, report: QAReport) -> None:
    if detection_frames == 0:
        report.warn("No frames were processed for detection")
    if person_frames == 0:
        report.warn("No person detections found in video")


def validate_tracking(
    observations_by_frame: dict[int, list],
    report: QAReport,
) -> None:
    for frame_index, obs_list in observations_by_frame.items():
        track_ids = [o[0] for o in obs_list]
        if len(track_ids) != len(set(track_ids)):
            report.error(f"Duplicate track IDs on frame {frame_index}")


def validate_pose(poses: dict[tuple[int, int], PoseResult], report: QAReport) -> None:
    if not poses:
        report.warn("No pose estimates produced")
        return
    confident = sum(1 for p in poses.values() if p.mean_score >= 0.3)
    ratio = confident / len(poses)
    if ratio < 0.5:
        report.warn(f"Low pose confidence on {ratio:.0%} of person-frames (threshold 50%)")


def validate_timeline(segments: list[TimelineSegment], report: QAReport) -> None:
    allowed = {b.value for b in TimelineBehaviour}
    by_person: dict[int, list[TimelineSegment]] = {}
    for seg in segments:
        if seg.behaviour.value not in allowed:
            report.error(f"Invalid behaviour: {seg.behaviour}")
        by_person.setdefault(seg.person_id, []).append(seg)

    for person_id, person_segs in by_person.items():
        ordered = sorted(person_segs, key=lambda s: s.start_ms)
        for i in range(1, len(ordered)):
            prev, curr = ordered[i - 1], ordered[i]
            if (
                curr.start_ms == prev.start_ms
                and curr.end_ms == prev.end_ms
                and curr.behaviour == prev.behaviour
            ):
                continue
            if curr.start_ms < prev.end_ms - 1:
                report.error(
                    f"Overlapping segments for person {person_id}: "
                    f"{prev.start_time}-{prev.end_time} vs {curr.start_time}-{curr.end_time}"
                )
            if curr.start_ms < prev.start_ms:
                report.error(f"Non-monotonic timeline for person {person_id}")


def validate_exports(
    output_dir: Path,
    segments: list[TimelineSegment],
    report: QAReport,
) -> None:
    required = [
        "annotated_video.mp4",
        "person_timeline.csv",
        "person_transitions.csv",
        "person_summary.xlsx",
        "session_summary.json",
    ]
    for name in required:
        path = output_dir / name
        if not path.is_file():
            report.error(f"Missing output file: {name}")

    csv_path = output_dir / "person_timeline.csv"
    if csv_path.is_file() and not segments:
        report.warn("person_timeline.csv exists but no segments were generated")

    xlsx_path = output_dir / "person_summary.xlsx"
    if xlsx_path.is_file():
        try:
            from openpyxl import load_workbook

            load_workbook(xlsx_path)
        except Exception as exc:
            report.error(f"person_summary.xlsx is not a valid workbook: {exc}")

    json_path = output_dir / "session_summary.json"
    if json_path.is_file():
        try:
            import json

            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "video" not in data or "session" not in data:
                report.error("session_summary.json missing required keys")
        except Exception as exc:
            report.error(f"session_summary.json invalid: {exc}")
