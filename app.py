"""SensorGuard monitoring and investigation dashboard."""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sensorguard.dashboard_data import (
    FEATURES,
    align_context,
    configured_paths,
    confusion_counts,
    filter_alerts,
    load_sensor_records,
    normalize_predictions,
    profile_row_count,
    read_csv_optional,
    read_json_optional,
    sensor_summary,
)

st.set_page_config(
    page_title="SensorGuard | Sensor quality",
    page_icon="SG",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
.stApp { background: #f5f7fa; color: #172b4d; }
[data-testid="stSidebar"] { background: #edf2f7; border-right: 1px solid #dbe3ec; }
h1,h2,h3 { color: #17324d; letter-spacing: -0.02em; }
[data-testid="stMetric"] { background: white; border: 1px solid #dbe3ec; padding: 14px 16px; border-radius: 10px; }
.sg-note { background: #eaf3f7; border-left: 4px solid #168c8c; padding: 12px 16px; border-radius: 6px; margin: 8px 0 16px; }
.sg-warn { background: #fff5e5; border-left: 4px solid #d58c22; padding: 12px 16px; border-radius: 6px; margin: 8px 0 16px; }
</style>
""",
    unsafe_allow_html=True,
)

PATHS = configured_paths()


def cache_version(path: str | Path) -> tuple[int | None, int | None]:
    """Invalidate cached reads when an artifact appears or changes on disk."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None, None
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False, ttl=600)
def cached_csv(path: str, version: tuple[int | None, int | None]) -> pd.DataFrame:
    return read_csv_optional(path)


@st.cache_data(show_spinner="Reading the selected sensor from the source dataset…", ttl=1800)
def cached_sensor(
    path: str, sensor_id: int, max_records: int, version: tuple[int | None, int | None]
) -> pd.DataFrame:
    return load_sensor_records(path, sensor_id, max_records)


@st.cache_data(show_spinner=False, ttl=600)
def cached_profile_count(path: str, version: tuple[int | None, int | None]):
    return profile_row_count(path)


@st.cache_data(show_spinner=False, ttl=600)
def cached_json(path: str, version: tuple[int | None, int | None]):
    return read_json_optional(path)


def load_artifacts():
    exp = PATHS["experiment"]

    def csv(name: str) -> pd.DataFrame:
        path = exp / name
        return cached_csv(str(path), cache_version(path))

    return {
        "metrics": csv("metrics.csv"),
        "per_fault": csv("per_fault_type.csv"),
        "per_sensor": csv("per_sensor.csv"),
        "clean_alerts": csv("clean_test_prediction_details.csv"),
        "synthetic_alerts": csv("prediction_details.csv"),
        "clean_records": csv("test_records.csv"),
        "injected_records": csv("injected_test_records.csv"),
        "curves": csv("precision_recall_curves.csv"),
        "delays": csv("detection_delay.csv"),
        "candidates": csv("historical_voltage_candidates.csv"),
        "metadata": cached_json(
            str(exp / "experiment_metadata.json"),
            cache_version(exp / "experiment_metadata.json"),
        ),
    }


def empty_message(title, message):
    st.subheader(title)
    st.info(message)


def normalize_records(frame):
    data = frame.copy()
    if data.empty:
        return data
    if "sensor_id" not in data and "moteid" in data:
        data["sensor_id"] = data["moteid"]
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data


def selected_records(sensor_id, artifacts, max_records=5000):
    if PATHS["data"].exists():
        try:
            result = cached_sensor(
                str(PATHS["data"]), int(sensor_id), max_records, cache_version(PATHS["data"])
            )
            if not result.empty:
                return normalize_records(result), "bounded source sample"
        except Exception as exc:
            st.warning(f"Could not read source file: {exc}")
    result = normalize_records(artifacts["clean_records"])
    if not result.empty:
        return result[result.sensor_id == sensor_id], "saved clean experiment test subset"
    return pd.DataFrame(), "no records available"


def severity_counts(alerts):
    if alerts.empty:
        return pd.DataFrame(columns=["severity", "count"])
    data = normalize_predictions(alerts)
    if "severity" not in data:
        return pd.DataFrame({"severity": ["unrated"], "count": [len(data)]})
    return data.groupby("severity", dropna=False).size().rename("count").reset_index()


def show_overview(a):
    st.title("SensorGuard")
    st.caption(
        "Sensor data quality and anomaly investigation · Intel Berkeley Research Lab dataset"
    )
    clean = normalize_predictions(a["clean_alerts"])
    metrics = a["metrics"]
    metadata = a["metadata"]
    sample_records = (
        int(metadata.get("sampled_rows", len(a["clean_records"])))
        if metadata
        else len(a["clean_records"])
    )
    test_records = (
        int(metadata.get("split_sizes", {}).get("test", len(a["clean_records"])))
        if metadata
        else len(a["clean_records"])
    )
    flagged = (
        clean[clean.get("predicted", pd.Series(dtype=bool))]
        if not clean.empty and "predicted" in clean
        else pd.DataFrame()
    )
    flagged_records = (
        flagged[["timestamp", "sensor_id"]].drop_duplicates().shape[0] if not flagged.empty else 0
    )
    sensors = normalize_records(a["clean_records"])
    sensor_count = (
        int(sensors.sensor_id.nunique()) if not sensors.empty and "sensor_id" in sensors else 0
    )
    profile_count = cached_profile_count(str(PATHS["profile"]), cache_version(PATHS["profile"]))
    st.markdown(
        '<div class="sg-note">Dashboard views distinguish unmodified test-set detections from controlled synthetic fault experiments. Historical readings have no verified anomaly labels.</div>',
        unsafe_allow_html=True,
    )
    c = st.columns(5)
    c[0].metric(
        "Source rows in last profile", f"{profile_count:,}" if profile_count else "Not profiled"
    )
    c[1].metric("Records in experiment sample", f"{sample_records:,}")
    c[2].metric("Sensors in test subset", f"{sensor_count:,}")
    c[3].metric("Unmodified test records flagged", f"{flagged_records:,}")
    pct = 100 * flagged_records / test_records if test_records else 0
    c[4].metric(
        "Flagged record rate",
        f"{pct:.1f}%",
        help=f"{flagged_records} of {test_records} records in the saved unmodified test subset",
    )
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Alerts by detector")
        if clean.empty:
            st.info("No unmodified test prediction artifact is available.")
        else:
            detector_col = "detector_group" if "detector_group" in clean else "detector"
            counts = (
                clean[clean.predicted]
                .groupby(detector_col)
                .size()
                .rename("alerts")
                .reset_index()
                .rename(columns={detector_col: "detector"})
            )
            st.plotly_chart(
                px.bar(
                    counts,
                    x="detector",
                    y="alerts",
                    color="detector",
                    color_discrete_sequence=["#168c8c", "#277da1", "#d58c22", "#586b83"],
                    title="Unmodified test subset · predicted alerts",
                ),
                width="stretch",
            )
        st.subheader("Data quality snapshot")
        records = normalize_records(a["clean_records"])
        if records.empty:
            st.info("Unmodified test record sample is unavailable.")
        else:
            summary = sensor_summary(records)
            missing = int(summary.missing_measurement_cells.sum()) if not summary.empty else 0
            gaps = int(summary.gaps_over_threshold.sum()) if not summary.empty else 0
            st.write(
                f"Missing measurement cells: **{missing:,}** · timestamp gaps over 300 seconds: **{gaps:,}** in the saved unmodified test subset."
            )
            if not clean.empty and "rule_id" in clean:
                rule_col = "detector_group" if "detector_group" in clean else "detector"
                rule_rows = clean[(clean[rule_col] == "rules") & clean.predicted]
                if not rule_rows.empty:
                    issue_counts = (
                        rule_rows.rule_id.fillna("rule finding")
                        .str.split("; ")
                        .explode()
                        .value_counts()
                        .head(8)
                        .rename_axis("quality issue")
                        .rename("count")
                        .reset_index()
                    )
                    st.plotly_chart(
                        px.bar(
                            issue_counts,
                            x="quality issue",
                            y="count",
                            title="Rule-based data-quality findings",
                            color_discrete_sequence=["#d58c22"],
                        ),
                        width="stretch",
                    )
                else:
                    st.caption(
                        "No rule-based quality findings were exported for this unmodified test sample."
                    )
    with c2:
        st.subheader("Alert severity")
        if clean.empty:
            st.info("Severity is available when prediction details are present.")
        else:
            levels = severity_counts(clean[clean.predicted])
            if levels.empty:
                st.info("No alerts in the unmodified test subset.")
            else:
                st.plotly_chart(
                    px.bar(
                        levels,
                        x="severity",
                        y="count",
                        color="severity",
                        color_discrete_map={
                            "critical": "#b42318",
                            "high": "#df6d47",
                            "medium": "#d58c22",
                            "low": "#168c8c",
                            "unrated": "#73849a",
                        },
                        title="Alert counts by severity",
                    ),
                    width="stretch",
                )
        st.subheader("Processing status")
        st.write("Dataset:", "available" if PATHS["data"].exists() else "source file not found")
        st.write("Saved experiment:", "available" if not metrics.empty else "results not found")
        checkpoint = any(PATHS["models"].glob("*")) if PATHS["models"].exists() else False
        st.write(
            "Model checkpoints:",
            "available"
            if checkpoint
            else "not present; dashboard uses saved predictions and does not retrain",
        )
        if not metrics.empty:
            st.caption(
                f"Saved experiment contains {len(metrics)} detector comparisons and {int(metrics.evaluated_records.max())} held-out records."
            )


def show_explorer(a):
    st.title("Sensor explorer")
    records = normalize_records(a["clean_records"])
    if records.empty:
        return empty_message(
            "No sensor sample",
            "The saved experiment does not include unmodified test records. Run the experiment script or configure another dataset path.",
        )
    sensors = sorted(int(x) for x in records.sensor_id.dropna().unique())
    id_col, feature_col, scope = st.columns([1, 1, 2])
    sid = id_col.selectbox("Sensor", sensors)
    feature = feature_col.selectbox("Measurement", FEATURES)
    raw, source_label = selected_records(sid, a)
    if raw.empty:
        return empty_message(
            "No readings", f"No readings for sensor {sid} were found in {source_label}."
        )
    st.caption(
        f"Showing a bounded {source_label}. Dataset timestamps are timezone-naive wall-clock values; no timezone metadata is provided by the source."
    )
    mind = raw.timestamp.min().date()
    maxd = raw.timestamp.max().date()
    if mind == maxd:
        dates = (mind, maxd)
    else:
        dates = st.date_input("Date range", value=(mind, maxd), min_value=mind, max_value=maxd)
    t1, t2 = st.columns(2)
    start_time = t1.time_input("Start time", time(0, 0))
    end_time = t2.time_input("End time", time(23, 59, 59))
    if isinstance(dates, tuple) and len(dates) == 2:
        begin = pd.Timestamp(datetime.combine(dates[0], start_time))
        end = pd.Timestamp(datetime.combine(dates[1], end_time))
    else:
        begin = pd.Timestamp(datetime.combine(mind, time.min))
        end = pd.Timestamp(datetime.combine(maxd, time.max))
    view = raw[(raw.timestamp >= begin) & (raw.timestamp <= end)].copy()
    synthetic = st.toggle(
        "Overlay controlled-injection test alerts",
        value=False,
        help="These markers are model predictions on a test copy that contains injected faults.",
    )
    marks = normalize_predictions(a["synthetic_alerts"] if synthetic else a["clean_alerts"])
    if not marks.empty:
        marks = marks[(marks.sensor_id == sid) & (marks.feature == feature) & marks.predicted]
        marks = marks[(marks.timestamp >= begin) & (marks.timestamp <= end)]
    fig = go.Figure()
    if feature not in view or view.empty:
        st.info("No readings for this feature in the selected interval.")
    else:
        fig.add_trace(
            go.Scatter(
                x=view.timestamp,
                y=pd.to_numeric(view[feature], errors="coerce"),
                mode="lines+markers",
                name=feature.title(),
                line={"color": "#277da1", "width": 2},
                marker={"size": 5},
                customdata=view.get("issues", pd.Series([[]] * len(view))).astype(str),
                hovertemplate="%{x}<br>value=%{y}<br>input notes=%{customdata}<extra></extra>",
            )
        )
        if not marks.empty:
            fig.add_trace(
                go.Scatter(
                    x=marks.timestamp,
                    y=marks.get("observed_value"),
                    mode="markers",
                    name="Flagged reading",
                    marker={"color": "#d9534f", "size": 11, "symbol": "x"},
                    customdata=marks.get(
                        "reason", marks.get("fault_type", pd.Series([""] * len(marks)))
                    ),
                    hovertemplate="%{x}<br>score=%{customdata}<extra>Flag</extra>",
                )
            )
        fig.update_layout(
            title=f"Sensor {sid} · {feature.title()}",
            xaxis_title="Dataset timestamp (timezone-naive)",
            yaxis_title=feature.title(),
            hovermode="x unified",
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )
        st.plotly_chart(fig, width="stretch")
    if len(view):
        view["gap_seconds"] = view.timestamp.diff().dt.total_seconds()
        missing = int(view[feature].isna().sum()) if feature in view else 0
        gaps = view[view.gap_seconds > 300]
        vals = pd.to_numeric(view[feature], errors="coerce").dropna()
        longest = (
            int(vals.eq(vals.shift()).groupby(vals.ne(vals.shift()).cumsum()).sum().max())
            if len(vals)
            else 0
        )
        c = st.columns(3)
        c[0].metric("Readings in range", len(view))
        c[1].metric(f"Missing {feature}", missing)
        c[2].metric("Gaps > 300s", len(gaps))
        st.caption(
            f"Longest exact repeated-value run for {feature}: {longest} observations. Constant readings can be legitimate; this is descriptive, not a fault diagnosis."
        )
    st.subheader("Contextual sensor comparison")
    peers = [x for x in sensors if x != sid]
    chosen = st.multiselect(
        "Reference sensors", peers, default=peers[: min(2, len(peers))], max_selections=3
    )
    if chosen and not view.empty:
        figc = go.Figure()
        figc.add_trace(
            go.Scatter(x=view.timestamp, y=view[feature], mode="lines", name=f"Sensor {sid}")
        )
        for peer_id in chosen:
            peer, _ = selected_records(peer_id, a)
            aligned = (
                align_context(view, peer, feature, "2min") if not peer.empty else pd.DataFrame()
            )
            if not aligned.empty:
                figc.add_trace(
                    go.Scatter(
                        x=aligned.timestamp,
                        y=aligned.reference_value,
                        mode="lines",
                        name=f"Sensor {peer_id} · latest prior ≤2 min",
                        line={"dash": "dot"},
                    )
                )
        figc.update_layout(
            template="plotly_white",
            title="Backward-aligned reference readings",
            xaxis_title="Naive dataset timestamp",
            yaxis_title=feature.title(),
            hovermode="x unified",
        )
        st.plotly_chart(figc, width="stretch")


def show_alerts(a):
    st.title("Alert investigation")
    data = normalize_predictions(a["clean_alerts"])
    injected = st.toggle(
        "Inspect controlled-injection predictions (synthetic test copy)", value=False
    )
    if injected:
        st.warning(
            "This view includes predictions on test records modified by synthetic fault injection. It is not a record of verified historical faults."
        )
        data = normalize_predictions(a["synthetic_alerts"])
    else:
        st.info(
            "Showing predictions on the unmodified held-out test subset. This is still an unlabeled historical sample; a detector flag is not verified ground truth."
        )
    if data.empty:
        return empty_message(
            "No alert artifact",
            "No saved prediction details were found. Configure SENSORGUARD_EXPERIMENT_DIR or run the experiment script.",
        )
    detector_col = "detector_group" if "detector_group" in data else "detector"
    detectors = sorted(data[detector_col].dropna().unique())
    c = st.columns([1.2, 1, 1, 1.5])
    det = c[0].multiselect("Detector", detectors, default=detectors[-1:] if detectors else [])
    ids = sorted(int(v) for v in data.sensor_id.dropna().unique())
    sids = c[1].multiselect("Sensor", ids)
    features = c[2].multiselect("Feature", sorted(data.feature.dropna().unique()))
    search = c[3].text_input("Search reason or feature")
    filtered = filter_alerts(
        data,
        sensor_ids=sids or None,
        detectors=det or None,
        features=features or None,
        search=search,
    )
    if filtered.empty:
        return st.info("No alerts match these filters.")
    if "reason" not in filtered:
        filtered["reason"] = "Detector flagged this reading; see its score and detector output."
    filtered = filtered.copy()
    filtered["severity"] = filtered.get(
        "severity", pd.Series("unrated", index=filtered.index)
    ).fillna("unrated")
    filtered["explanation"] = filtered["reason"].fillna("")
    filtered["anomaly_score"] = filtered.get(
        "score", filtered.get("anomaly_score", pd.Series(index=filtered.index, dtype=float))
    )
    page_size = 50
    pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, pages, 1, 1)
    shown = filtered.iloc[(page - 1) * page_size : page * page_size]
    columns = [
        c
        for c in [
            "timestamp",
            "sensor_id",
            "feature",
            "observed_value",
            "anomaly_score",
            "severity",
            "detector_group",
            "detector",
            "explanation",
            "rule_id",
            "fault_type",
            "ground_truth",
        ]
        if c in shown
    ]
    st.dataframe(shown[columns], width="stretch", hide_index=True)
    st.download_button(
        "Export filtered alerts as CSV",
        filtered[columns].to_csv(index=False).encode("utf-8"),
        file_name="sensorguard_filtered_alerts.csv",
        mime="text/csv",
    )
    st.subheader("Selected alert details")
    choices = filtered.reset_index(drop=True)
    labels = [
        f"{r.timestamp} · sensor {r.sensor_id} · {r.feature} · {r.detector}"
        for r in choices.itertuples()
    ]
    selected = st.selectbox("Alert", range(len(labels)), format_func=lambda i: labels[i])
    row = choices.iloc[selected]
    c = st.columns(4)
    c[0].metric("Observed value", str(row.get("observed_value", "not available")))
    c[1].metric(
        "Anomaly score",
        f"{row.get('anomaly_score', float('nan')):.5g}"
        if pd.notna(row.get("anomaly_score"))
        else "not available",
    )
    c[2].metric("Severity", str(row.get("severity", "unrated")))
    c[3].metric("Detector output", str(row.get("detector", "unknown")))
    st.write("**Explanation:**", row.get("explanation", "No explanation was exported."))
    st.write("**Rule:**", row.get("rule_id", "—"))
    raw, _ = selected_records(int(row.sensor_id), a)
    if not raw.empty:
        center = pd.Timestamp(row.timestamp)
        near = raw[
            (raw.timestamp >= center - pd.Timedelta(minutes=5))
            & (raw.timestamp <= center + pd.Timedelta(minutes=5))
        ]
        if row.feature in FEATURES and not near.empty:
            fig = px.line(
                near,
                x="timestamp",
                y=row.feature,
                markers=True,
                title="Five-minute surrounding readings",
            )
            st.plotly_chart(fig, width="stretch")
        peers = [s for s in ids if s != int(row.sensor_id)][:3]
        if row.feature in FEATURES and peers:
            comparisons = []
            for peer_id in peers:
                peer, _ = selected_records(peer_id, a)
                aligned = (
                    align_context(near, peer, row.feature, "2min")
                    if not near.empty and not peer.empty
                    else pd.DataFrame()
                )
                if not aligned.empty:
                    aligned["peer_sensor_id"] = peer_id
                    comparisons.append(aligned)
            if comparisons:
                st.dataframe(
                    pd.concat(comparisons, ignore_index=True), width="stretch", hide_index=True
                )


def show_evaluation(a):
    st.title("Model evaluation")
    st.markdown(
        '<div class="sg-warn"><b>Controlled experiment:</b> all labels shown here come from deliberate synthetic edits to a held-out test copy. The historical sensor dataset is otherwise unlabeled.</div>',
        unsafe_allow_html=True,
    )
    metrics = a["metrics"]
    if metrics.empty:
        return empty_message(
            "Evaluation unavailable",
            "No metrics.csv was found. Run scripts/run_experiment.py to generate the evaluation artifacts.",
        )
    st.caption(
        "Evaluation unit is a sensor-feature-reading pair, including a record marker for record-level rules. Missing prediction rows count as negative. No-injection pairs are treated as negative, but their natural anomaly status is unknown."
    )
    chosen = st.selectbox(
        "Detector comparison", metrics.detector.tolist(), index=max(0, len(metrics) - 1)
    )
    row = metrics[metrics.detector == chosen].iloc[0]
    cols = st.columns(5)
    for col, key, label in zip(
        cols,
        ["precision", "recall", "f1", "false_positive_rate", "evaluated_pairs"],
        ["Precision", "Recall", "F1", "Apparent false-positive rate", "Evaluated pairs"],
    ):
        col.metric(label, f"{row[key]:.3f}" if key != "evaluated_pairs" else f"{int(row[key]):,}")
    c1, c2 = st.columns(2)
    with c1:
        vals = metrics.melt(
            id_vars="detector",
            value_vars=["precision", "recall", "f1", "false_positive_rate"],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(
            px.bar(
                vals,
                x="detector",
                y="value",
                color="metric",
                barmode="group",
                title="Detector comparison",
                range_y=[0, 1],
            ),
            width="stretch",
        )
    with c2:
        fault = a["per_fault"]
        if fault.empty:
            st.info("Per-fault results not available.")
        else:
            st.plotly_chart(
                px.bar(
                    fault,
                    x="fault_type",
                    y="recall",
                    color="detector",
                    barmode="group",
                    range_y=[0, 1],
                    title="Recall by injected fault type",
                ),
                width="stretch",
            )
    details = normalize_predictions(a["synthetic_alerts"])
    if not details.empty:
        counts = confusion_counts(details, chosen)
        matrix = [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]]
        fig = px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted no alert", "Predicted alert"],
            y=["No injected fault label", "Injected fault"],
            color_continuous_scale="Blues",
            title=f"Confusion matrix · {chosen}",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Non-injected pairs are counted as negative for these synthetic metrics; historical anomalies are unlabeled and may be included."
        )
    curves = a["curves"]
    if not curves.empty:
        plot = curves[curves.detector == chosen]
        if not plot.empty:
            st.plotly_chart(
                px.line(
                    plot, x="recall", y="precision", title=f"Precision-recall curve · {chosen}"
                ),
                width="stretch",
            )
    if not a["per_fault"].empty:
        st.dataframe(a["per_fault"], width="stretch", hide_index=True)
    if not a["delays"].empty:
        st.subheader("Detection delay (drift/outage cases)")
        st.dataframe(a["delays"], width="stretch", hide_index=True)
    meta = a["metadata"]
    if meta:
        with st.expander("Thresholds and sample configuration"):
            st.json(
                {
                    "thresholds": meta.get("thresholds"),
                    "split_sizes": meta.get("split_sizes"),
                    "training_range": meta.get("training_range"),
                    "validation_range": meta.get("validation_range"),
                    "test_range": meta.get("test_range"),
                    "seed": meta.get("seed"),
                    "threshold_method": meta.get("threshold_method"),
                    "evaluation_unit": meta.get("evaluation_unit"),
                }
            )
    st.caption(
        "Synthetic amplitudes and durations were chosen by the experimenter and do not estimate natural failure prevalence. Scores are not probabilities of failure."
    )


def show_health(a):
    st.title("Sensor health")
    records = normalize_records(a["clean_records"])
    if records.empty:
        return empty_message(
            "Health summary unavailable", "No unmodified test records were exported."
        )
    summaries = sensor_summary(records)
    alerts = normalize_predictions(a["clean_alerts"])
    if not alerts.empty:
        summary = (
            alerts[alerts.predicted]
            .groupby("sensor_id")
            .agg(
                anomaly_pairs=("predicted", "size"),
                rule_findings=(
                    "experiment_detector",
                    lambda s: int((s == "rules").sum()) if "experiment_detector" in alerts else 0,
                ),
            )
            .reset_index()
        )
        summaries = summaries.merge(summary, on="sensor_id", how="left")
    else:
        summaries["anomaly_pairs"] = 0
        summaries["rule_findings"] = 0
    summaries[["anomaly_pairs", "rule_findings"]] = (
        summaries[["anomaly_pairs", "rule_findings"]].fillna(0).astype(int)
    )
    summaries["anomaly_rate_per_pair"] = summaries.anomaly_pairs / (summaries.records_in_sample * 5)
    candidates = a["candidates"]
    if not candidates.empty:
        cols = [
            "sensor_id",
            "temperature_voltage_correlation",
            "temperature_adjusted_voltage_slope_per_day",
            "candidate_decline",
            "candidate_abrupt_change",
            "interpretation",
        ]
        summaries = summaries.merge(
            candidates[[c for c in cols if c in candidates]], on="sensor_id", how="left"
        )
    st.markdown(
        '<div class="sg-warn">This view summarizes a small unmodified test sample and precomputed detector outputs. It is not a verified hardware-health score. Battery labels are not available.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(summaries, width="stretch", hide_index=True)
    fig = px.line(
        records.sort_values("timestamp"),
        x="timestamp",
        y="voltage",
        color="sensor_id",
        markers=True,
        title="Voltage history in unmodified test subset",
    )
    st.plotly_chart(fig, width="stretch")
    if not candidates.empty:
        st.subheader("Temperature-adjusted candidate cases")
        st.dataframe(candidates, width="stretch", hide_index=True)
    st.caption(
        "Anomaly rate is a count of flagged feature-reading pairs divided by five markers per record. It is not a probability. Voltage trends are adjusted using a per-sensor linear temperature fit and remain exploratory."
    )


a = load_artifacts()
with st.sidebar:
    st.title("SensorGuard")
    page = st.radio(
        "Workspace",
        ["Overview", "Sensor explorer", "Alert investigation", "Model evaluation", "Sensor health"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Artifact configuration")
    st.code(f"data: {PATHS['data']}\nresults: {PATHS['experiment']}", language=None)
    st.caption("Override with SENSORGUARD_DATA_PATH and SENSORGUARD_EXPERIMENT_DIR.")

if page == "Overview":
    show_overview(a)
elif page == "Sensor explorer":
    show_explorer(a)
elif page == "Alert investigation":
    show_alerts(a)
elif page == "Model evaluation":
    show_evaluation(a)
else:
    show_health(a)
