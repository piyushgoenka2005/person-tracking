"""Engine exception types."""


class EngineError(Exception):
    """Base engine error."""


class PerceptionError(EngineError):
    """Detection, tracking, or pose failure."""


class IngestionError(EngineError):
    """Video intake failure."""


class StorageError(EngineError):
    """Artifact read/write failure."""
