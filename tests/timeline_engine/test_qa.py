"""Tests for QA validators."""

from __future__ import annotations

import json
from pathlib import Path

from engine.export.csv_exporter import export_timeline_csv, export_transitions_csv
from engine.export.json_exporter import export_session_json
from engine.export.xlsx_exporter import export_person_summary_xlsx
from engine.qa.validators import (
    QAReport,
    validate_exports,
    validate_pose,
    validate_timeline,
    validate_tracking,
)
from engine.types import PoseResult, SessionContext, TimelineBehaviour, TimelineSegment, VideoInfo


def test_validate_tracking_duplicate_ids():
    report = QAReport()
    validate_tracking({0: [(1, 0, 0, 10, 10, 0), (1, 5, 5, 15, 15, 0)]}, report)
    assert not report.ok
    assert len(report.errors) == 1


def test_validate_timeline_overlap():
    report = QAReport()
    segments = [
        TimelineSegment(1, "00:00:00", "00:00:10", TimelineBehaviour.STANDING, 0.8, 10, 0, 10_000),
        TimelineSegment(1, "00:00:05", "00:00:15", TimelineBehaviour.WALKING, 0.9, 10, 5_000, 15_000),
    ]
    validate_timeline(segments, report)
    assert not report.ok


def test_validate_pose_low_confidence():
    report = QAReport()
    poses = {(0, 1): PoseResult(0, 1, [], [0.1])}
    validate_pose(poses, report)
    assert report.warnings


def test_validate_exports_schema(tmp_path: Path):
    report = QAReport()
    segments = [
        TimelineSegment(
            1, "00:00:10", "00:00:20", TimelineBehaviour.STANDING, 0.87, 10, 10_000, 20_000
        ),
    ]
    export_timeline_csv(tmp_path / "person_timeline.csv", segments)
    export_transitions_csv(tmp_path / "person_transitions.csv", [])
    export_person_summary_xlsx(tmp_path / "person_summary.xlsx", segments)
    context = SessionContext(
        video=VideoInfo("t.mp4", 25, 640, 480, 100, 4),
        yolo_model="yolo11x.pt",
        pose_model="m.onnx",
        device="cpu",
        frame_stride=1,
    )
    export_session_json(tmp_path / "session_summary.json", context, segments)
    (tmp_path / "annotated_video.mp4").write_bytes(b"\x00")
    validate_exports(tmp_path, segments, report)
    assert report.ok

    data = json.loads((tmp_path / "session_summary.json").read_text())
    assert "video" in data
    assert "session" in data
    assert "qa" in data
