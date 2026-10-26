"""Unified orchestration while preserving individual detector outputs."""

from .detection import merge_alerts


class DetectionEnsemble:
    def __init__(self, detectors):
        self.detectors = dict(detectors)

    def fit(self, train, validation=None):
        for detector in self.detectors.values():
            if hasattr(detector, "fit"):
                try:
                    detector.fit(train, validation=validation)
                except TypeError:
                    detector.fit(train)
        return self

    def predict(self, frame):
        individual = {name: detector.predict(frame) for name, detector in self.detectors.items()}
        return individual, merge_alerts(*individual.values())
