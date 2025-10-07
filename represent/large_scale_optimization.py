"""Large-scale optimization functionality has been deprecated."""

from typing import Any

LARGE_SCALE_OPTIMIZATION_AVAILABLE = False


class LargeScaleParameterOptimizer:  # pragma: no cover - simple stub
    """Stub retained for backwards compatibility. Always raises RuntimeError."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Large-scale parameter optimization has been removed from represent. "
            "The previous implementation depended on tstrends and has been deprecated."
        )


def run_large_scale_optimization(*args: Any, **kwargs: Any) -> None:
    """Compatibility wrapper that always raises RuntimeError."""
    raise RuntimeError(
        "Large-scale parameter optimization has been removed from represent. "
        "The previous implementation depended on tstrends and has been deprecated."
    )
