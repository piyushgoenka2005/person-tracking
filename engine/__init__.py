"""Behaviour Timeline Engine — standalone video behaviour analytics."""

from engine.config import TimelineSettings

__all__ = ["TimelineSettings", "TimelinePipeline"]


def __getattr__(name: str):
    if name == "TimelinePipeline":
        from engine.pipeline import TimelinePipeline

        return TimelinePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
