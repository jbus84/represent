"""Parameter optimization functionality has been deprecated."""

from typing import Any

OPTIMIZATION_AVAILABLE = False


class ParameterOptimizer:  # pragma: no cover - simple stub
    """Stub maintained for backward compatibility.

    RuntimeError is raised immediately because the historical tstrends-based
    optimization flow has been removed from represent.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Parameter optimization has been removed from represent. "
            "The previous implementation depended on tstrends and has been deprecated."
        )


def optimize_all_methods(*args: Any, **kwargs: Any) -> None:
    """Compatibility helper that always raises RuntimeError."""
    raise RuntimeError(
        "Parameter optimization has been removed from represent. "
        "The previous implementation depended on tstrends and has been deprecated."
    )
