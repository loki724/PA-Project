"""
High-Risk Detection and Anomaly Analysis Module.
Uses Isolation Forest to detect multivariate pollution anomalies and statistical
thresholding (90th percentile) to identify high-risk conditions.
Adapts dynamically to both UCI multi-sensor datasets and general air-quality datasets.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def classify_risk_metric(val: float, median_threshold: float, p90_threshold: float) -> str:
    """
    Classifies a measurement into statistical dataset-relative risk categories:
    - Normal: <= median_threshold
    - Elevated: > median_threshold and <= p90_threshold
    - High: > p90_threshold
    Note: Dataset-relative statistical risk, not official regulatory AQI.
    """
    if pd.isna(val):
        return "Unknown"
    if val <= median_threshold:
        return "Normal"
    elif val <= p90_threshold:
        return "Elevated"
    else:
        return "High"


def classify_co_risk(co_val: float, median_threshold: float = 2.2, p90_threshold: float = 4.2) -> str:
    """Convenience alias for CO risk classification."""
    return classify_risk_metric(co_val, median_threshold, p90_threshold)


def run_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.05,
    co_median: float = 2.2,
    co_p90: float = 4.2
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[IsolationForest]]:
    """
    Fits an Isolation Forest model on available numeric pollutant columns.
    Computes multivariate anomaly flags and dataset-relative statistical risk categories.
    Works seamlessly whether the dataset is UCI Air Quality or a general air-quality dataset.
    """
    res_df = df.copy()

    # Identify numeric candidate columns for anomaly detection
    exclude_cols = {"Hour", "Month", "DayOfWeek", "IsWeekend", "is_anomaly", "anomaly_score"}
    numeric_cols = [
        c for c in res_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(res_df[c]) and res_df[c].dropna().nunique() > 1
    ]

    # If no numeric columns are present, return safe defaults
    if not numeric_cols:
        res_df["is_anomaly"] = False
        res_df["anomaly_score"] = 0.0
        res_df["risk_category"] = "Normal"
        stats = {
            "total_records": len(res_df),
            "anomaly_count": 0,
            "anomaly_pct": 0.0,
            "high_risk_count": 0,
            "high_risk_pct": 0.0,
            "risk_breakdown": {"Normal": len(res_df)},
            "risk_basis_column": "None",
            "co_thresholds": {"median": co_median, "90th_percentile": co_p90}
        }
        return res_df, stats, None

    # Handle missing values using SimpleImputer (median)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(res_df[numeric_cols])

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    raw_preds = iso_forest.fit_predict(X_imputed) # -1: anomaly, 1: normal
    anomaly_scores = iso_forest.decision_function(X_imputed)

    res_df["is_anomaly"] = raw_preds == -1
    res_df["anomaly_score"] = anomaly_scores.round(4)

    # Determine risk basis column (prefer CO(GT), then CO, PM2.5, AQI, NO2, or first numeric col)
    risk_col = None
    for cand in ["CO(GT)", "CO", "PM2.5", "AQI", "AQI Value", "NO2(GT)", "NO2", "C6H6(GT)"]:
        if cand in res_df.columns and res_df[cand].dropna().notna().sum() > 10:
            risk_col = cand
            break

    if risk_col is None:
        risk_col = numeric_cols[0]

    risk_series = pd.to_numeric(res_df[risk_col], errors="coerce").dropna()
    if len(risk_series) > 0:
        med_th = float(np.percentile(risk_series, 50))
        p90_th = float(np.percentile(risk_series, 90))
    else:
        med_th = co_median
        p90_th = co_p90

    res_df["risk_category"] = res_df[risk_col].apply(
        lambda v: classify_risk_metric(v, median_threshold=med_th, p90_threshold=p90_th)
    )

    total_records = len(res_df)
    anomaly_count = int(res_df["is_anomaly"].sum())
    anomaly_pct = float(round((anomaly_count / total_records) * 100, 2)) if total_records > 0 else 0.0

    risk_counts = res_df["risk_category"].value_counts().to_dict()
    high_risk_count = risk_counts.get("High", 0)
    high_risk_pct = float(round((high_risk_count / total_records) * 100, 2)) if total_records > 0 else 0.0

    stats = {
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_pct": anomaly_pct,
        "high_risk_count": high_risk_count,
        "high_risk_pct": high_risk_pct,
        "risk_breakdown": risk_counts,
        "risk_basis_column": risk_col,
        "co_thresholds": {
            "median": med_th,
            "90th_percentile": p90_th
        }
    }

    # Save fitted Isolation Forest model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))

    # Save static anomaly plot for PDF report
    save_static_anomaly_chart(res_df, p90_th, risk_col)

    return res_df, stats, iso_forest


def create_plotly_anomaly_scatter(
    df: pd.DataFrame,
    p90_threshold: float = 4.2,
    target_column: Optional[str] = None
) -> go.Figure:
    """Interactive time-series scatter highlighting Isolation Forest anomalies and high-risk spikes."""
    if "is_anomaly" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Anomaly detection has not been executed yet.", x=0.5, y=0.5, showarrow=False)
        return fig

    # Determine x-axis (DateTime if valid, else index)
    has_dt = "DateTime" in df.columns and df["DateTime"].notna().sum() > 0
    x_col_name = "DateTime" if has_dt else "_Index"
    plot_df = df.copy()
    if not has_dt:
        plot_df["_Index"] = plot_df.index

    # Determine y-axis column
    if target_column is None or target_column not in plot_df.columns:
        for cand in ["CO(GT)", "CO", "PM2.5", "AQI", "NO2(GT)", "NO2", "anomaly_score"]:
            if cand in plot_df.columns:
                target_column = cand
                break
        if target_column is None:
            target_column = "anomaly_score"

    normal_df = plot_df[~plot_df["is_anomaly"]]
    anom_df = plot_df[plot_df["is_anomaly"]]

    fig = go.Figure()

    # Normal points
    fig.add_trace(go.Scatter(
        x=normal_df[x_col_name],
        y=normal_df[target_column],
        mode="markers",
        name="Normal Pattern",
        marker=dict(color="#10B981", size=4, opacity=0.4)
    ))

    # Anomalous points
    fig.add_trace(go.Scatter(
        x=anom_df[x_col_name],
        y=anom_df[target_column],
        mode="markers",
        name="Detected Anomaly",
        marker=dict(color="#EF4444", size=7, symbol="cross", opacity=0.9)
    ))

    # Statistical 90th percentile threshold line
    if target_column != "anomaly_score":
        fig.add_hline(
            y=p90_threshold,
            line_dash="dash",
            line_color="#B91C1C",
            annotation_text=f"Dataset 90th Percentile Cutoff ({p90_threshold:.2f})",
            annotation_position="top left"
        )

    fig.update_layout(
        title=f"Multivariate Anomaly Detection Timeline ({target_column})",
        xaxis_title="Date & Time" if has_dt else "Observation Sequence (Index)",
        yaxis_title=f"Concentration / Level: {target_column}",
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_risk_distribution(stats: Dict[str, Any]) -> go.Figure:
    """Generates a donut chart representing distribution of pollution risk categories."""
    breakdown = stats.get("risk_breakdown", {"Normal": 1})
    labels = list(breakdown.keys())
    values = list(breakdown.values())

    color_map = {
        "Normal": "#10B981",
        "Elevated": "#F59E0B",
        "High": "#EF4444",
        "Unknown": "#9CA3AF"
    }
    colors = [color_map.get(k, "#6B7280") for k in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.45,
        marker=dict(colors=colors)
    )])
    basis = stats.get("risk_basis_column", "Target")
    fig.update_layout(
        title=f"Dataset-Relative Risk Distribution (Basis: {basis})",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def save_static_anomaly_chart(
    df: pd.DataFrame,
    p90_threshold: float,
    target_column: Optional[str] = None
) -> str:
    """Saves static anomaly timeline chart for inclusion in the PDF report."""
    os.makedirs(VIS_DIR, exist_ok=True)
    out_path = os.path.join(VIS_DIR, "anomaly_timeline.png")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 4.5))

    has_dt = "DateTime" in df.columns and df["DateTime"].notna().sum() > 0
    x_axis = df["DateTime"] if has_dt else df.index

    if target_column is None or target_column not in df.columns:
        for cand in ["CO(GT)", "CO", "PM2.5", "AQI", "NO2(GT)", "anomaly_score"]:
            if cand in df.columns:
                target_column = cand
                break
        if target_column is None:
            target_column = "anomaly_score"

    is_anom = df.get("is_anomaly", pd.Series(False, index=df.index))
    normal_mask = ~is_anom
    anom_mask = is_anom

    ax.scatter(x_axis[normal_mask], df[target_column][normal_mask], color="#2ecc71", alpha=0.3, s=8, label="Normal Observations")
    ax.scatter(x_axis[anom_mask], df[target_column][anom_mask], color="#e74c3c", alpha=0.8, s=25, marker="x", label="Detected Anomalies")

    if target_column != "anomaly_score":
        ax.axhline(p90_threshold, color="#c0392b", linestyle="--", linewidth=1.5, label=f"90th %ile ({p90_threshold:.2f})")

    ax.set_xlabel("Observation Timeline" if has_dt else "Observation Sequence (Index)", fontweight="bold")
    ax.set_ylabel(f"{target_column}", fontweight="bold")
    ax.set_title(f"Isolation Forest Anomaly Detection ({target_column})", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path
