"""onsetwindow: individual risk under an accumulating, whole-body burden model."""

__version__ = "0.1.0"

from .params import ParameterError, Parameters, load_parameters  # noqa: E402
from .model import (  # noqa: E402
    Intervention,
    Trajectory,
    risk_by_age,
    simulate,
    window_of_opportunity,
)

__all__ = [
    "ParameterError",
    "Parameters",
    "load_parameters",
    "Intervention",
    "Trajectory",
    "risk_by_age",
    "simulate",
    "window_of_opportunity",
    "__version__",
]
