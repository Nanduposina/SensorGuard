"""SensorGuard sensor data quality pipeline."""

from .autoencoder import AutoencoderDetector
from .config import SensorGuardConfig
from .detection import merge_alerts
from .experiment import evaluate_track
from .historical import voltage_case_studies
from .ingestion import iter_records, read_chunks
from .injection import FaultSpec, InjectionResult, inject_faults
from .isolation import IsolationForestDetector
from .rules import RuleDetector, contextual_deviations, fit_contextual_baselines

__all__ = [
    "SensorGuardConfig",
    "iter_records",
    "read_chunks",
    "RuleDetector",
    "contextual_deviations",
    "fit_contextual_baselines",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "merge_alerts",
    "FaultSpec",
    "InjectionResult",
    "inject_faults",
    "evaluate_track",
    "voltage_case_studies",
]
