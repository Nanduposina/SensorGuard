"""Quality annotation and explicit, provenance-preserving cleaning policies."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import SensorGuardConfig

MEASUREMENTS = ("temperature", "humidity", "light", "voltage")
POLICIES = {"preserve", "flag", "drop", "impute"}


def validate_frame(frame: pd.DataFrame, config: SensorGuardConfig | None = None) -> pd.DataFrame:
    """Return a copy with quality flags; never removes or repairs observations."""
    cfg = config or SensorGuardConfig()
    result = frame.copy()
    if "moteid" not in result and "sensor_id" in result:
        result["moteid"] = result["sensor_id"]
    if "timestamp" not in result:
        result["timestamp"] = pd.NaT
    if "moteid" not in result:
        result["moteid"] = pd.NA
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    if "issues" not in result:
        result["issues"] = [[] for _ in range(len(result))]
    result["issues"] = result["issues"].map(lambda v: list(v) if isinstance(v, list) else [])
    for idx, row in result.iterrows():
        issues = row["issues"]
        mote = row.get("moteid")
        if pd.notna(mote) and int(mote) not in cfg.sensor_ids:
            issues.append("invalid_sensor_id")
        for feature, (low, high) in cfg.physical_limits.items():
            value = row.get(feature)
            if pd.notna(value) and not low <= float(value) <= high:
                issues.append(f"suspicious_{feature}")
    result["issues"] = result["issues"].map(lambda x: list(dict.fromkeys(x)))
    result["duplicate"] = False
    keyed = result.dropna(subset=["timestamp", "moteid"])
    result.loc[keyed.index, "duplicate"] = keyed.duplicated(
        subset=["timestamp", "moteid"], keep="first"
    )
    result["out_of_order"] = False
    result["sampling_gap_seconds"] = float("nan")
    # Keep source order here: the per-sensor stream is continued across chunks
    # by the cleaning CLI, so a gap/order check has the same meaning at any chunk size.
    valid = result.dropna(subset=["timestamp", "moteid"])
    gaps = valid.groupby("moteid")["timestamp"].diff().dt.total_seconds()
    result.loc[valid.index, "sampling_gap_seconds"] = gaps
    result.loc[valid.index[gaps > cfg.sampling_gap_seconds], "issues"] = result.loc[
        valid.index[gaps > cfg.sampling_gap_seconds], "issues"
    ].map(lambda x: x + ["sampling_gap"])
    backwards = valid.groupby("moteid", sort=False)["timestamp"].diff() < pd.Timedelta(0)
    result.loc[valid.index[backwards], "out_of_order"] = True
    return result


def fit_imputation_statistics(
    frame: pd.DataFrame,
    training_mask: pd.Series | np.ndarray,
) -> dict[str, dict[int | None, float]]:
    """Fit per-sensor and global medians on explicitly selected training rows.

    The caller must provide a chronological training mask. Per-sensor medians are
    preferred; a training-only feature median is the fallback for sparse sensors.
    No values are interpolated across time or sensor boundaries.
    """
    if len(training_mask) != len(frame):
        raise ValueError("training_mask length must match the input frame")
    mask = np.asarray(training_mask, dtype=bool)
    if not mask.any():
        raise ValueError("training_mask must select at least one row")
    source = frame.loc[mask].copy()
    if "sensor_id" not in source and "moteid" in source:
        source["sensor_id"] = source["moteid"]
    if "sensor_id" not in source:
        raise ValueError("frame must contain sensor_id or moteid")
    result: dict[str, dict[int | None, float]] = {}
    for feature in MEASUREMENTS:
        if feature not in source:
            continue
        values = pd.to_numeric(source[feature], errors="coerce")
        stats: dict[int | None, float] = {}
        valid = values.notna()
        if valid.any():
            stats[None] = float(values[valid].median())
        for sensor_id, indexes in source.groupby("sensor_id", dropna=True).groups.items():
            sensor_values = values.loc[indexes].dropna()
            if not sensor_values.empty:
                stats[int(sensor_id)] = float(sensor_values.median())
        result[feature] = stats
    return result


def clean_frame(
    frame: pd.DataFrame,
    *,
    policies: Mapping[str, str] | None = None,
    imputation_statistics: Mapping[str, Mapping[int | None, float]] | None = None,
    config: SensorGuardConfig | None = None,
) -> pd.DataFrame:
    """Apply explicit row/measurement policies and retain an audit row per input.

    Policy keys are issue categories: ``timestamp``, ``out_of_order``,
    ``duplicate_record``, ``duplicate_key``, ``missing_measurement``,
    ``implausible_value``, ``sampling_gap``, ``missing_sensor_id``, ``unexpected_sensor_id``, and
    ``malformed_record``. Unspecified issues default to ``flag``. ``impute`` is
    only supported for missing measurements and requires training-fitted
    ``imputation_statistics``. A row is marked ``retained=False`` if any of its
    issue policies is ``drop``; callers can still export its provenance.
    """
    cfg = config or SensorGuardConfig()
    choices = dict(policies or {})
    invalid = {key: value for key, value in choices.items() if value not in POLICIES}
    if invalid:
        raise ValueError(f"unknown cleaning policy value(s): {invalid}")
    if choices.get("impute") == "impute":
        raise ValueError("impute is not an issue category; use missing_measurement=impute")
    if any(value == "impute" for value in choices.values()) and not imputation_statistics:
        raise ValueError("imputation requires training-fitted imputation_statistics")
    unsupported_impute = [
        key for key, value in choices.items() if value == "impute" and key != "missing_measurement"
    ]
    if unsupported_impute:
        raise ValueError(f"impute is only supported for missing_measurement: {unsupported_impute}")

    source = frame.copy().reset_index(drop=True)
    if "sensor_id" in source and "moteid" not in source:
        source["moteid"] = source["sensor_id"]
    result = validate_frame(source, cfg).reset_index(drop=True)
    original_fields = [name for name in ("timestamp", "moteid", *MEASUREMENTS) if name in result]
    for name in original_fields:
        original = source[name] if name in source else result[name]
        result[f"original_{name}"] = original
    if "original_timestamp" in result:
        result["original_timestamp"] = result.original_timestamp.map(
            lambda value: value.isoformat(sep=" ") if isinstance(value, pd.Timestamp) else value
        )
    if "sampling_gap_seconds" in source:
        source_gaps = pd.to_numeric(source.sampling_gap_seconds, errors="coerce").reset_index(
            drop=True
        )
        local_gaps = pd.to_numeric(result.sampling_gap_seconds, errors="coerce")
        result["sampling_gap_seconds"] = local_gaps.where(local_gaps.notna(), source_gaps)
    result["retained"] = True
    result["policy_actions"] = ""
    result["quality_issues"] = ""
    result["cleaning_provenance"] = ""

    if "raw" in result:
        raw_values = result.raw.fillna("")
        result["_exact_duplicate"] = raw_values.ne("") & raw_values.duplicated(keep="first")
    else:
        identity_fields = [
            column
            for column in result.columns
            if column
            not in {"issues", "line_number", "duplicate", "out_of_order", "sampling_gap_seconds"}
            and not column.startswith("original_")
        ]
        result["_exact_duplicate"] = result[identity_fields].duplicated(keep="first")
    result["_duplicate_key"] = False
    keyed = result.dropna(subset=["timestamp", "moteid"])
    result.loc[keyed.index, "_duplicate_key"] = keyed.duplicated(
        ["timestamp", "moteid"], keep="first"
    )

    def category(issue: str) -> str:
        if issue in {"invalid_timestamp", "missing_timestamp"}:
            return "timestamp"
        if issue in {"out_of_order"}:
            return "out_of_order"
        if issue in {"duplicate_record"}:
            return "duplicate_record"
        if issue == "duplicate_key":
            return "duplicate_key"
        if issue.startswith("missing_") and issue.removeprefix("missing_") in MEASUREMENTS:
            return "missing_measurement"
        if issue.startswith(("suspicious_", "implausible_")):
            return "implausible_value"
        if issue in {"sampling_gap"}:
            return "sampling_gap"
        if issue in {"invalid_sensor_id", "unexpected_sensor_id"}:
            return "unexpected_sensor_id"
        if issue in {"missing_moteid", "missing_sensor_id"}:
            return "missing_sensor_id"
        return "malformed_record"

    for index, row in result.iterrows():
        issues = list(row.get("issues", [])) if isinstance(row.get("issues"), (list, tuple)) else []
        if pd.isna(row.get("timestamp")):
            original_timestamp = row.get("original_timestamp")
            timestamp_issue = (
                "invalid_timestamp"
                if "invalid_timestamp" in issues or not pd.isna(original_timestamp)
                else "missing_timestamp"
            )
            issues.append(timestamp_issue)
        if pd.isna(row.get("moteid")) and "invalid_moteid" not in issues:
            issues.append("missing_sensor_id")
        if bool(row["_exact_duplicate"]):
            issues.append("duplicate_record")
        if bool(row["_duplicate_key"]):
            issues.append("duplicate_key")
        if pd.notna(row.get("moteid")) and int(row.moteid) not in cfg.sensor_ids:
            issues.append("unexpected_sensor_id")
        for feature in MEASUREMENTS:
            value = row.get(feature)
            if pd.isna(value):
                issues.append(f"missing_{feature}")
            elif feature in cfg.physical_limits:
                lower, upper = cfg.physical_limits[feature]
                if not lower <= float(value) <= upper:
                    issues.append(f"implausible_{feature}")
        if bool(row.get("out_of_order", False)):
            issues.append("out_of_order")
        gap = row.get("sampling_gap_seconds")
        if pd.notna(gap) and gap > cfg.sampling_gap_seconds:
            issues.append("sampling_gap")

        issues = sorted(set(str(item) for item in issues))
        result.at[index, "duplicate"] = "duplicate_key" in issues
        result.at[index, "out_of_order"] = (
            bool(row.get("out_of_order", False)) or "out_of_order" in issues
        )
        actions: list[dict[str, object]] = []
        flagged: list[str] = []
        should_drop = False
        sensor_id = row.get("moteid")
        for issue in issues:
            issue_category = category(issue)
            action = choices.get(issue_category, "flag")
            detail: dict[str, object] = {
                "issue": issue,
                "category": issue_category,
                "policy": action,
                "original": None,
                "result": None,
                "retained": True,
            }
            if issue.startswith("missing_") and issue.removeprefix("missing_") in MEASUREMENTS:
                feature = issue.removeprefix("missing_")
                original = row.get(feature)
                detail["original"] = None if pd.isna(original) else original
                if action == "impute":
                    stats = (imputation_statistics or {}).get(feature, {})
                    replacement = stats.get(int(sensor_id)) if pd.notna(sensor_id) else None
                    if replacement is None:
                        replacement = stats.get(None)
                    if replacement is None or not np.isfinite(float(replacement)):
                        raise ValueError(f"no training median available to impute {feature}")
                    result.at[index, feature] = float(replacement)
                    detail["result"] = float(replacement)
                else:
                    detail["result"] = None
            elif action in {"flag", "preserve"}:
                field = issue.removeprefix("implausible_").removeprefix("suspicious_")
                detail["original"] = row.get(field) if field in result else row.get("raw")
                detail["result"] = detail["original"]
            if action == "drop":
                should_drop = True
            if action == "flag":
                flagged.append(issue)
            actions.append(detail)

        # A drop policy always excludes the entire row from the clean view, but
        # the audit/provenance view below retains it and explains every action.
        result.at[index, "retained"] = not should_drop
        result.at[index, "quality_issues"] = ";".join(issues)
        result.at[index, "policy_actions"] = ";".join(
            dict.fromkeys(str(item["policy"]) for item in actions)
        )
        result.at[index, "cleaning_provenance"] = json.dumps(
            {
                "line_number": row.get("line_number"),
                "sensor_id": None if pd.isna(sensor_id) else int(sensor_id),
                "timestamp": None if pd.isna(row.get("timestamp")) else str(row.get("timestamp")),
                "issues": issues,
                "actions": actions,
                "retained": not should_drop,
            },
            sort_keys=True,
            default=str,
        )
        if flagged:
            result.at[index, "issues"] = list(dict.fromkeys([*issues, *flagged]))

    return result.drop(columns=["_exact_duplicate", "_duplicate_key"])


def cleaning_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return deterministic aggregate counts from ``clean_frame`` output."""
    if "retained" not in frame:
        raise ValueError("cleaning summary requires clean_frame output")
    issue_counts: Counter[str] = Counter()
    for issues in frame.get("quality_issues", pd.Series(dtype=str)).fillna(""):
        issue_counts.update(issue for issue in str(issues).split(";") if issue)
    policy_counts = Counter(
        action
        for actions in frame.get("policy_actions", pd.Series(dtype=str)).fillna("")
        for action in str(actions).split(";")
        if action
    )
    return {
        "input_rows": int(len(frame)),
        "retained_rows": int(frame.retained.sum()),
        "excluded_rows": int((~frame.retained).sum()),
        "issue_counts": dict(sorted(issue_counts.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "policies": "Per-row actions are preserved in cleaning_provenance.",
    }
