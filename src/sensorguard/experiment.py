"""Leakage-resistant supervised evaluation of controlled injections."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score

from .autoencoder import AutoencoderDetector
from .detection import merge_alerts, normalize_input
from .evaluation import binary_metrics, chronological_split
from .injection import FaultSpec, inject_faults
from .isolation import FEATURES, IsolationForestDetector
from .rules import RuleDetector


def choose_threshold(scores, labels, *, objective="f1"):
    """Choose a threshold from validation scores only (max F1 or precision-recall)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    if not len(scores):
        raise ValueError("validation scores are empty")
    if not labels.any():
        return float(np.quantile(scores, 0.99))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return float(np.max(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if objective != "f1":
        raise ValueError("objective must be f1")
    return float(thresholds[int(np.argmax(f1))])


def _scores_by_pair(output, frame, score_col="anomaly_score"):
    if output.empty:
        return pd.Series(dtype=float)
    return output.groupby(["timestamp", "sensor_id", "feature"])[score_col].max()


def evaluate_track(
    frame: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    seed=42,
    threshold_quantile=0.99,
    autoencoder_epochs=30,
    n_estimators=100,
):
    """Fit on unmodified train, calibrate on unmodified validation, and test injections.

    Evaluation unit is feature-reading pair. Missing or absent detector rows are treated
    as negative predictions. Unmodified historical data is not verified fault-free, so
    its no-injection alert rate is descriptive and its unlabeled pairs may include anomalies.
    """
    data = normalize_input(frame).reset_index(drop=True).sort_values("timestamp", kind="stable")
    train, val, test = chronological_split(data, 0.6, 0.2)
    # Controlled injection deliberately targets held-out test only.
    specs = []
    usable = test.groupby("sensor_id").size()
    if usable.empty:
        raise ValueError("no test sensors")
    sid = int(usable[usable >= 4].index[0]) if (usable >= 4).any() else int(usable.index[0])
    group = test[test.sensor_id == sid].sort_values("timestamp")
    span = max(1, min(5, len(group) // 4))
    start = max(1, len(group) // 2)
    specs = [
        FaultSpec("spike", sid, "temperature", start, 1, 18),
        FaultSpec("gradual_drift", sid, "humidity", start + 2, span, 10),
        FaultSpec(
            "missing_block",
            sid,
            "light",
            start + span + 3,
            min(span, len(group) - (start + span + 3)),
            None,
        ),
    ]
    specs = [s for s in specs if s.duration > 0 and s.start_index + s.duration <= len(group)]
    injected = inject_faults(test, specs, seed=seed)
    rules = RuleDetector().fit(train)
    iso = IsolationForestDetector(
        random_state=seed, n_estimators=n_estimators, contamination="auto"
    ).fit(train)
    ae = AutoencoderDetector(
        seed=seed,
        epochs=autoencoder_epochs,
        threshold_quantile=threshold_quantile,
        patience=max(2, min(8, autoencoder_epochs // 3)),
    ).fit(train, val)
    # Calibrate IF cutoff only on unmodified validation scores, without injection labels.
    val_scores = iso.score_samples(val)[1]
    iso_cut = float(np.quantile(val_scores, threshold_quantile))
    iso.threshold = iso_cut
    # AE's threshold was calibrated exclusively on its unmodified validation partition.
    predictions = {
        "rules": rules.predict(injected.frame),
        "isolation_forest": iso.predict(injected.frame),
        "autoencoder": ae.predict(injected.frame),
    }
    predictions["combined_ml_if_ae"] = merge_alerts(
        predictions["isolation_forest"], predictions["autoencoder"]
    )
    predictions["combined_rules_if"] = merge_alerts(
        predictions["rules"], predictions["isolation_forest"]
    )
    predictions["combined_rules_if_ae"] = merge_alerts(*predictions.values())
    clean = {
        "rules": rules.predict(test),
        "isolation_forest": iso.predict(test),
        "autoencoder": ae.predict(test),
    }
    clean["combined_ml_if_ae"] = merge_alerts(clean["isolation_forest"], clean["autoencoder"])
    clean["combined_rules_if"] = merge_alerts(clean["rules"], clean["isolation_forest"])
    clean["combined_rules_if_ae"] = merge_alerts(*clean.values())
    # Evaluate each feature-reading pair with stable modified timestamps.
    truth = {
        (r.timestamp, r.sensor_id, r.feature) for r in injected.labels.itertuples() if r.is_fault
    }
    universe = []
    for row in injected.frame.itertuples():
        for f in FEATURES:
            if f in injected.frame:
                universe.append((row.timestamp, row.sensor_id, f))
        # Record-level detections (gaps, order, duplicates) share the same
        # reading unit and need their own explicitly negative/positive marker.
        universe.append((row.timestamp, row.sensor_id, "record"))
    # Keep sensor outages/missing pairs in the denominator even though values are null.
    universe = sorted(
        set(universe) | truth,
        key=lambda key: (key[0], key[1], str(key[2])),
    )
    type_by_pair = {}
    for r in injected.labels.itertuples():
        type_by_pair[(r.timestamp, r.sensor_id, r.feature)] = r.fault_type
    metrics = []
    predictions_table = []
    pr_curves = {}
    window_type = {}
    for fault in injected.fault_manifest.itertuples():
        features = fault.features.split(";")
        for key in universe:
            if (
                key[1] == fault.sensor_id
                and key[2] in features
                and pd.Timestamp(fault.start_timestamp)
                <= pd.Timestamp(key[0])
                <= pd.Timestamp(fault.end_timestamp)
            ):
                window_type.setdefault(key, fault.fault_type)
    error_cases = []
    for name, out in predictions.items():
        scores = _scores_by_pair(out, injected.frame)
        decisions = {
            (r.timestamp, r.sensor_id, r.feature): bool(r.is_anomaly) for r in out.itertuples()
        }
        detector_details = {}
        if not out.empty:
            for alert in out.itertuples():
                detector_details[(alert.timestamp, alert.sensor_id, alert.feature)] = {
                    "observed_value": getattr(alert, "observed_value", np.nan),
                    "reason": getattr(alert, "reason", ""),
                    "severity": getattr(alert, "severity", "unrated"),
                    "rule_id": getattr(alert, "rule_id", ""),
                }
        y = np.array([key in truth for key in universe])
        pred = np.array([decisions.get(key, False) for key in universe])
        summary = binary_metrics(y, pred)
        metrics.append(
            {
                "detector": name,
                "unit": "sensor-feature-reading pair (including record marker)",
                "injected_faults": len(injected.fault_manifest),
                "fault_events": len(injected.fault_manifest),
                "evaluated_records": len(injected.frame),
                **summary,
                # Combining unrelated score scales with max() does not yield
                # a meaningful ranking, so PR-AUC is reported per detector.
                "pr_auc": None
                if name.startswith("combined_")
                else _pr_auc(scores, truth, universe),
            }
        )
        if not name.startswith("combined_"):
            pr_curves[name] = _pr_curve(scores, truth, universe)
        for key in universe:
            is_true = key in truth
            is_pred = decisions.get(key, False)
            detail = detector_details.get(key, {})
            predictions_table.append(
                {
                    "detector": name,
                    "timestamp": key[0],
                    "sensor_id": key[1],
                    "feature": key[2],
                    "ground_truth": is_true,
                    "fault_type": type_by_pair.get(key),
                    "fault_window_type": window_type.get(key),
                    "predicted": is_pred,
                    "score": scores.get(key, np.nan),
                    "observed_value": detail.get("observed_value", np.nan),
                    "reason": detail.get("reason", ""),
                    "severity": detail.get("severity", "unrated"),
                    "rule_id": detail.get("rule_id", ""),
                }
            )
            if is_true != is_pred:
                error_cases.append(
                    {
                        "detector": name,
                        "error_type": "false_negative" if is_true else "false_positive",
                        "timestamp": key[0],
                        "sensor_id": key[1],
                        "feature": key[2],
                        "fault_type": type_by_pair.get(key) or window_type.get(key),
                        "score": scores.get(key, np.nan),
                    }
                )
    metrics_df = pd.DataFrame(metrics)
    per_fault = []
    per_sensor = []
    pred_df = pd.DataFrame(predictions_table)
    for (detector, ftype), g in pred_df[pred_df.fault_window_type.notna()].groupby(
        ["detector", "fault_window_type"]
    ):
        yy = g.ground_truth.to_numpy(bool)
        pp = g.predicted.to_numpy(bool)
        per_fault.append(
            {
                "detector": detector,
                "fault_type": ftype,
                "fault_events": int((injected.fault_manifest.fault_type == ftype).sum()),
                "evaluated_pairs": len(g),
                "positive_pairs": int(yy.sum()),
                "precision": float(precision_score(yy, pp, zero_division=0)),
                "recall": float(recall_score(yy, pp, zero_division=0)),
                "f1": float(f1_score(yy, pp, zero_division=0)),
                "false_positives": int((~yy & pp).sum()),
            }
        )
    for (detector, sid), g in pred_df.groupby(["detector", "sensor_id"]):
        yy = g.ground_truth.to_numpy(bool)
        pp = g.predicted.to_numpy(bool)
        positives = g[g.ground_truth]
        per_sensor.append(
            {
                "detector": detector,
                "sensor_id": sid,
                "fault_events": int((injected.fault_manifest.sensor_id == sid).sum()),
                "evaluated_pairs": len(g),
                "positive_pairs": len(positives),
                "precision": float(precision_score(yy, pp, zero_division=0)),
                "recall": float(recall_score(yy, pp, zero_division=0)),
                "f1": float(f1_score(yy, pp, zero_division=0)),
                "false_positives": int((~yy & pp).sum()),
            }
        )
    # No-injection held-out alert baseline; historical negatives are unverified.
    clean_metrics = []
    for name, out in clean.items():
        dec = (
            out.groupby(["timestamp", "sensor_id", "feature"]).is_anomaly.any()
            if not out.empty
            else pd.Series(dtype=bool)
        )
        n = len(
            set(
                (r.timestamp, r.sensor_id, f)
                for r in test.itertuples()
                for f in (*FEATURES, "record")
            )
        )
        clean_metrics.append(
            {
                "detector": name,
                "clean_test_pairs": n,
                "evaluated_records": len(test),
                "fault_events": 0,
                "positive_pairs": 0,
                "false_positives": int(dec.sum()),
                "false_positive_rate": float(dec.sum() / n) if n else None,
            }
        )
    clean_rows = []
    for detector_name, out in clean.items():
        if out.empty:
            continue
        item = out.copy()
        item["experiment_detector"] = detector_name
        item["predicted"] = item["is_anomaly"]
        item["score"] = item["anomaly_score"]
        item["is_injected_test"] = False
        clean_rows.append(item)
    clean_detail = pd.concat(clean_rows, ignore_index=True) if clean_rows else pd.DataFrame()
    delay = _detection_delays(pred_df, injected.fault_manifest)
    return {
        "metrics": metrics_df,
        "per_fault": pd.DataFrame(per_fault),
        "per_sensor": pd.DataFrame(per_sensor),
        "clean_baseline": pd.DataFrame(clean_metrics),
        "fault_manifest": injected.fault_manifest,
        "injection_observations": injected.observations,
        "prediction_details": pred_df,
        "clean_prediction_details": clean_detail,
        "error_cases": pd.DataFrame(error_cases),
        "detection_delay": pd.DataFrame(delay),
        "pr_curves": {k: v for k, v in pr_curves.items() if v is not None},
        "predictions": predictions,
        "clean_predictions": clean,
        "test_frame": test,
        "injected_frame": injected.frame,
        "thresholds": {
            "isolation_forest_validation_quantile": threshold_quantile,
            "isolation_forest_threshold": iso_cut,
            "autoencoder_validation_threshold": ae.threshold,
        },
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "training_range": [str(train.timestamp.min()), str(train.timestamp.max())],
        "validation_range": [str(val.timestamp.min()), str(val.timestamp.max())],
        "test_range": [str(test.timestamp.min()), str(test.timestamp.max())],
    }


def _aligned_scores(scores, truth, universe):
    if not truth:
        return None, None
    keys = list(universe)
    y = np.array([k in truth for k in keys])
    present = scores.to_dict() if not scores.empty else {}
    floor = min(present.values()) - 1.0 if present else 0.0
    return y, np.array([present.get(k, floor) for k in keys], dtype=float)


def _pr_auc(scores, truth, universe):
    y, s = _aligned_scores(scores, truth, universe)
    if y is None or not y.any():
        return None
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p))


def _pr_curve(scores, truth, universe):
    y, s = _aligned_scores(scores, truth, universe)
    if y is None or not y.any():
        return None
    p, r, t = precision_recall_curve(y, s)
    thresholds = np.append(t, np.nan)
    return pd.DataFrame({"precision": p, "recall": r, "threshold": thresholds})


def _detection_delays(pred, manifest):
    results = []
    for fault in manifest.itertuples():
        subset = pred[
            (pred.fault_type.str.contains(fault.fault_type, na=False))
            & (pred.sensor_id == fault.sensor_id)
        ]
        for detector, g in subset.groupby("detector"):
            detected = g[g.predicted]
            delay = (
                None
                if detected.empty
                else max(
                    0,
                    (
                        pd.Timestamp(detected.timestamp.min()) - pd.Timestamp(fault.start_timestamp)
                    ).total_seconds(),
                )
            )
            if fault.fault_type in {"gradual_drift", "sensor_outage"}:
                results.append(
                    {
                        "fault_id": fault.fault_id,
                        "fault_type": fault.fault_type,
                        "detector": detector,
                        "delay_seconds": delay,
                    }
                )
    return results
