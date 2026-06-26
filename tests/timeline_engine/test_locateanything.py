"""Tests for LocateAnything detection integration."""

from __future__ import annotations

from engine.detection.locateanything import parse_boxes


def test_parse_boxes_normalized_coords():
    answer = 'Found person<box><100><200><300><400></box> and another<box><500><100><700><800></box>'
    boxes = parse_boxes(answer, image_width=1000, image_height=500)
    assert len(boxes) == 2
    assert boxes[0]["x1"] == 100.0
    assert boxes[0]["y1"] == 100.0
    assert boxes[0]["x2"] == 300.0
    assert boxes[0]["y2"] == 200.0
