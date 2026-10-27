"""SensorGuard monitoring and investigation dashboard."""

from __future__ import annotations

import sqlite3
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
    filter_alerts,
    load_sensor_records,
    normalize_predictions,
    profile_row_count,
    read_csv_optional,
    read_json_optional,
    sensor_summary,
)
from sensorguard.review_store import (
    LABEL_TITLES,
    LABELS,
    ReviewConflictError,
    ReviewImportError,
    ReviewStore,
    export_reviews,
    prepare_review_alerts,
    reviewed_alert_summary,
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
        "alert_breakdown": csv("alert_breakdown.csv"),
        "reading_alert_summary": csv("reading_alert_summary.csv"),
        "rule_breakdown": csv("rule_breakdown.csv"),
        "clean_alerts": csv("clean_test_prediction_details.csv"),
        "synthetic_alerts": csv("prediction_details.csv"),
        "clean_records": csv("test_records.csv"),
        "injected_records": csv("injected_test_records.csv"),
        "curves": csv("precision_recall_curves.csv"),
        "delays": csv("detection_delay.csv"),
        "candidates": csv("historical_voltage_candidates.csv"),
        "voltage_windows": csv("historical_voltage_windows.csv"),
        "contextual_coverage": csv("contextual_coverage.csv"),
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
    try:
        reviews = ReviewStore(PATHS["reviews"]).list_reviews()
    except (OSError, sqlite3.Error, ValueError) as exc:
        reviews = []
        st.warning(f"Human review feedback could not be loaded: {exc}")
    feedback = reviewed_alert_summary(a["clean_alerts"], reviews)
    with st.expander("Independently reviewed alert feedback", expanded=False):
        st.caption(
            "This summary matches human reviews to the current unmodified alert artifact by stable alert ID. It describes reviewed alerts only; review selection is not random and it is not a held-out model accuracy estimate. Uncertain reviews are excluded from the binary confirmation rate."
        )
        cols = st.columns(4)
        cols[0].metric("Matched reviews", feedback["matched_reviews"])
        cols[1].metric("Confirmed anomaly", feedback["confirmed_anomaly"])
        cols[2].metric("Normal behavior", feedback["normal_behavior"])
        cols[3].metric("Uncertain", feedback["uncertain"])
        rate = feedback["reviewed_alert_confirmation_rate"]
        st.metric("Reviewed-alert confirmation rate", "n/a" if rate is None else f"{rate:.1%}")
        normal_share = feedback["reviewed_alert_normal_share"]
        normal_share_text = "n/a" if normal_share is None else f"{normal_share:.1%}"
        st.caption(
            f"Binary reviewed sample: {feedback['binary_reviewed']} matched alerts; normal-labelled share: {normal_share_text}. Current alerts: {feedback['current_alerts']}; stored reviews unmatched to this artifact: {feedback['unmatched_stored_reviews']}. Reviewable observations are selected predicted alerts, so this share is not a false-positive rate, recall, or dataset-wide precision estimate."
        )
    if metrics.empty:
        return empty_message(
            "Evaluation unavailable",
            "No metrics.csv was found. Run scripts/run_experiment.py to generate the evaluation artifacts.",
        )
    st.caption(
        "Evaluation unit is a sensor-feature-reading pair, including a record marker for record-level rules. Only directly modified injected pairs have known labels. Unmodified historical pairs remain unlabeled; this page reports injected-pair recall and alerts on unlabeled pairs, not precision or verified false-positive rates."
    )
    chosen = st.selectbox(
        "Detector comparison", metrics.detector.tolist(), index=max(0, len(metrics) - 1)
    )
    row = metrics[metrics.detector == chosen].iloc[0]
    metric_rows = [
        ("recall", "Injected-pair recall", "rate"),
        ("true_positives", "Injected pairs detected", "count"),
        ("positive_pairs", "Known injected pairs", "count"),
        (
            "alerts_on_unlabeled_pairs"
            if "alerts_on_unlabeled_pairs" in metrics
            else "false_positives",
            "Alerts outside injected pairs",
            "count",
        ),
        (
            "unlabeled_alert_rate" if "unlabeled_alert_rate" in metrics else "false_positive_rate",
            "Unlabeled alert rate",
            "rate",
        ),
    ]
    cols = st.columns(len(metric_rows))
    for col, (key, label, kind) in zip(cols, metric_rows):
        value = row.get(key)
        rendered = (
            "n/a" if pd.isna(value) else f"{value:.3f}" if kind == "rate" else f"{int(value):,}"
        )
        col.metric(label, rendered)
    c1, c2 = st.columns(2)
    with c1:
        metric_columns = ["recall"]
        if "unlabeled_alert_rate" in metrics:
            metric_columns.append("unlabeled_alert_rate")
        elif "false_positive_rate" in metrics:
            metric_columns.append("false_positive_rate")
        vals = metrics.melt(
            id_vars="detector", value_vars=metric_columns, var_name="metric", value_name="value"
        )
        vals["metric"] = vals.metric.map(
            {
                "recall": "Injected-pair recall",
                "unlabeled_alert_rate": "Unlabeled alert rate",
                "false_positive_rate": "Legacy apparent alert rate",
            }
        )
        st.plotly_chart(
            px.bar(
                vals,
                x="detector",
                y="value",
                color="metric",
                barmode="group",
                title="Injected-pair recall and unlabeled alert rate",
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
                    y="injection_recall" if "injection_recall" in fault else "recall",
                    color="detector",
                    barmode="group",
                    range_y=[0, 1],
                    title="Recall by injected fault type",
                ),
                width="stretch",
            )
    st.info(
        "Confusion matrices and precision-recall curves are withheld because unmodified readings have no verified negative labels. Inspect detector contributions and per-sensor results below."
    )
    if not a["per_sensor"].empty:
        st.subheader("Per-sensor results")
        st.dataframe(
            a["per_sensor"][a["per_sensor"].detector == chosen],
            width="stretch",
            hide_index=True,
        )
    if not a["alert_breakdown"].empty:
        st.subheader("Alerts by sensor and feature")
        st.dataframe(
            a["alert_breakdown"][a["alert_breakdown"].detector == chosen],
            width="stretch",
            hide_index=True,
        )
    if not a["reading_alert_summary"].empty:
        st.caption(
            "Multivariate scores may appear on multiple feature rows. This table separates feature-pair alert counts from distinct sensor-timestamp readings."
        )
        st.dataframe(a["reading_alert_summary"], width="stretch", hide_index=True)
    if chosen == "rules" and not a["rule_breakdown"].empty:
        st.subheader("Rule contributions")
        st.dataframe(a["rule_breakdown"], width="stretch", hide_index=True)
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
                    "contextual_features_enabled": meta.get("contextual_features_enabled"),
                    "contextual_reference_method": meta.get("contextual_reference_method"),
                    "peer_radius_meters": meta.get("peer_radius_meters"),
                    "peer_minimum": meta.get("peer_minimum"),
                    "peer_tolerance": meta.get("peer_tolerance"),
                }
            )
    if not a["contextual_coverage"].empty:
        st.subheader("Physical peer-reference availability")
        st.caption(
            "Peer features use official mote coordinates and the configured radius. They are spatial comparison evidence, not confirmed faults; missing context is not scored as normal."
        )
        st.dataframe(a["contextual_coverage"], width="stretch", hide_index=True)
    st.caption(
        "Synthetic amplitudes and durations were chosen by the experimenter and do not estimate natural failure prevalence. Scores are not probabilities of failure."
    )


def show_review(a):
    """Collect and manage reviewer labels independently of model artifacts."""
    st.title("Alert review")
    st.markdown(
        '<div class="sg-note">Review unmodified historical alerts using independent evidence. Labels are stored separately and never retrain detectors or change thresholds.</div>',
        unsafe_allow_html=True,
    )
    raw_alerts = normalize_predictions(a["clean_alerts"])
    alerts, invalid = prepare_review_alerts(raw_alerts)
    try:
        store = ReviewStore(PATHS["reviews"])
        saved = store.list_reviews()
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f"The local review store is unavailable: {exc}")
        return
    saved_by_id = {review["alert_id"]: review for review in saved}

    if invalid:
        st.warning(
            f"{invalid} alert row(s) could not be reviewed because identity fields were missing or invalid."
        )
    if alerts.empty:
        empty_message(
            "No reviewable alerts",
            "No predicted rows were found in the unmodified alert artifact. Run the experiment to generate alert details. Synthetic-injection predictions are intentionally excluded from human historical review.",
        )
    else:
        alerts["review_status"] = alerts.alert_id.map(
            lambda alert_id: saved_by_id.get(alert_id, {}).get("label", "unreviewed")
        )
        c = st.columns([1, 1, 1, 1.5])
        sensors = sorted(int(value) for value in alerts.sensor_id.dropna().unique())
        sensor_filter = c[0].multiselect("Sensor", sensors, key="review_sensor_filter")
        detector_col = "detector_group" if "detector_group" in alerts else "detector"
        detectors = sorted(alerts[detector_col].dropna().astype(str).unique())
        detector_filter = c[1].multiselect("Detector", detectors, key="review_detector_filter")
        features = sorted(alerts.feature.dropna().astype(str).unique())
        feature_filter = c[2].multiselect("Feature", features, key="review_feature_filter")
        status_filter = c[3].selectbox(
            "Review status",
            ["All", "Unreviewed", *[LABEL_TITLES[label] for label in LABELS]],
            key="review_status_filter",
        )
        search = st.text_input("Search explanation or alert ID", key="review_search")
        filtered = alerts
        if sensor_filter:
            filtered = filtered[filtered.sensor_id.isin(sensor_filter)]
        if detector_filter:
            filtered = filtered[filtered[detector_col].astype(str).isin(detector_filter)]
        if feature_filter:
            filtered = filtered[filtered.feature.astype(str).isin(feature_filter)]
        if status_filter == "Unreviewed":
            filtered = filtered[filtered.review_status == "unreviewed"]
        elif status_filter != "All":
            wanted = next(label for label in LABELS if LABEL_TITLES[label] == status_filter)
            filtered = filtered[filtered.review_status == wanted]
        if search:
            explanation = filtered.get(
                "explanation", filtered.get("reason", pd.Series("", index=filtered.index))
            )
            mask = filtered.alert_id.str.contains(
                search, case=False, na=False
            ) | explanation.astype(str).str.contains(search, case=False, na=False)
            filtered = filtered[mask]
        st.caption(
            f"{len(filtered):,} matching reviewable alert(s); {len(saved):,} review record(s) in local store."
        )
        table_cols = [
            col
            for col in [
                "alert_id",
                "timestamp",
                "sensor_id",
                "feature",
                "observed_value",
                detector_col,
                "anomaly_score",
                "score",
                "severity",
                "reason",
                "explanation",
                "review_status",
            ]
            if col in filtered
        ]
        if filtered.empty:
            st.info("No alerts match these filters.")
        else:
            st.dataframe(filtered[table_cols].head(100), width="stretch", hide_index=True)
            if len(filtered) > 100:
                st.caption(
                    "Showing the first 100 matching alerts. Use filters or search to narrow the list."
                )
            choices = filtered.reset_index(drop=True)

            def alert_label(index):
                row = choices.iloc[index]
                return (
                    f"{row.timestamp} · sensor {int(row.sensor_id)} · {row.feature} · "
                    f"{row.get(detector_col, 'unknown')} · {row.alert_id[:16]}"
                )

            selected = st.selectbox(
                "Select alert to review",
                range(len(choices)),
                format_func=alert_label,
                key="review_alert_selection",
            )
            row = choices.iloc[selected]
            prior = saved_by_id.get(row.alert_id)
            st.subheader("Selected alert details")
            cols = st.columns(5)
            cols[0].metric("Sensor ID", str(int(row.sensor_id)))
            cols[1].metric("Timestamp", str(row.timestamp))
            cols[2].metric("Feature / value", f"{row.feature} / {row.get('observed_value', 'n/a')}")
            cols[3].metric("Detector", str(row.get(detector_col, "unknown")))
            score = row.get("anomaly_score")
            if score is None or pd.isna(score):
                score = row.get("score")
            cols[4].metric(
                "Score / severity",
                str(score if pd.notna(score) else row.get("severity", "n/a")),
            )
            explanation = row.get("explanation")
            if explanation is None or pd.isna(explanation):
                explanation = row.get("reason", "No explanation available.")
            st.write("**Explanation:**", explanation)
            st.code(row.alert_id, language=None)
            if prior:
                st.info(
                    f"Current saved decision: {LABEL_TITLES[prior['label']]} · {prior['reviewed_at']}"
                )
            default_index = LABELS.index(prior["label"]) if prior else None
            decision = st.radio(
                "Reviewer classification",
                LABELS,
                index=default_index,
                format_func=lambda label: LABEL_TITLES[label],
                key="review_decision",
            )
            notes = st.text_area(
                "Optional notes",
                value=(prior["notes"] or "") if prior else "",
                key="review_notes",
            )
            if st.button("Save review decision", type="primary", key="save_review"):
                if decision is None:
                    st.error("Choose a classification before saving.")
                else:
                    try:
                        saved_review = store.save_review(row.to_dict(), decision, notes)
                        st.session_state["review_save_feedback"] = (
                            f"Saved {LABEL_TITLES[saved_review['label']]} for alert {saved_review['alert_id'][:16]}."
                        )
                        st.rerun()
                    except (OSError, sqlite3.Error, ValueError) as exc:
                        st.error(f"Review could not be saved: {exc}")
    if "review_save_feedback" in st.session_state:
        st.success(st.session_state.pop("review_save_feedback"))

    st.divider()
    st.subheader("Export or import review data")
    st.caption(f"Local review store: {PATHS['reviews']}")
    st.download_button(
        "Download review backup (JSON)",
        data=export_reviews(saved),
        file_name="sensorguard_reviews.json",
        mime="application/json",
        key="export_reviews",
    )
    upload = st.file_uploader(
        "Import a SensorGuard review JSON", type=["json"], key="review_upload"
    )
    overwrite = st.checkbox(
        "Allow imported records to overwrite existing reviews with matching alert IDs",
        value=False,
        key="review_import_overwrite",
    )
    if st.button("Import review file", key="import_reviews"):
        if upload is None:
            st.error("Choose a JSON review export first.")
        else:
            try:
                result = store.import_reviews(upload.getvalue(), overwrite=overwrite)
                st.success(
                    f"Imported {result['imported']} new review(s) and updated {result['overwritten']} existing review(s)."
                )
                st.rerun()
            except ReviewConflictError as exc:
                st.error(
                    str(exc) + ". Confirm overwrite only if replacing these decisions is intended."
                )
            except (ReviewImportError, OSError, sqlite3.Error) as exc:
                st.error(f"Review import rejected; no records were written: {exc}")


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
    if not a["voltage_windows"].empty:
        st.subheader("Temperature-adjusted voltage candidates")
        st.caption(
            "Unverified candidate windows only. The dataset documentation reports strong temperature dependence; no battery-failure labels are available."
        )
        st.dataframe(a["voltage_windows"], width="stretch", hide_index=True)
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
        [
            "Overview",
            "Sensor explorer",
            "Alert investigation",
            "Alert review",
            "Model evaluation",
            "Sensor health",
        ],
        label_visibility="collapsed",
        key="workspace_navigation",
    )
    st.divider()
    st.caption("Artifact configuration")
    st.code(f"data: {PATHS['data']}\nresults: {PATHS['experiment']}", language=None)
    st.caption(
        "Override paths with SENSORGUARD_DATA_PATH, SENSORGUARD_EXPERIMENT_DIR and SENSORGUARD_REVIEW_STORE."
    )

if page == "Overview":
    show_overview(a)
elif page == "Sensor explorer":
    show_explorer(a)
elif page == "Alert investigation":
    show_alerts(a)
elif page == "Alert review":
    show_review(a)
elif page == "Model evaluation":
    show_evaluation(a)
else:
    show_health(a)
