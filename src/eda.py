"""
Exploratory Data Analysis (EDA) Module.
Computes descriptive statistics, correlation matrices, diurnal/seasonal trends,
and provides both Plotly interactive figures and Matplotlib static charts.
Dynamically adapts to any available numeric environmental columns and temporal configurations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def get_numeric_pollutant_columns(df: pd.DataFrame) -> List[str]:
    """Identifies all numeric pollutant and environmental columns in the dataset."""
    exclude = {"Hour", "Month", "DayOfWeek", "IsWeekend", "is_anomaly", "anomaly_score"}
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique() > 1:
            numeric_cols.append(c)
    return numeric_cols


def get_descriptive_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes comprehensive descriptive statistics for numeric environmental columns:
    Mean, Median, Standard Deviation, Min, 25%, 75%, Max, and Skewness.
    """
    if columns is None:
        columns = get_numeric_pollutant_columns(df)

    stats_list = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        stats_list.append({
            "Variable": col,
            "Count": len(s),
            "Mean": round(float(s.mean()), 2),
            "Std Dev": round(float(s.std()), 2) if len(s) > 1 else 0.0,
            "Median": round(float(s.median()), 2),
            "Min": round(float(s.min()), 2),
            "25th %": round(float(s.quantile(0.25)), 2),
            "75th %": round(float(s.quantile(0.75)), 2),
            "Max": round(float(s.max()), 2),
            "Skewness": round(float(s.skew()), 2) if len(s) > 2 else 0.0
        })

    if not stats_list:
        return pd.DataFrame([{"Message": "No numeric measurements available for statistical summary"}])

    return pd.DataFrame(stats_list)


def get_hourly_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean pollutant concentrations by hour of day (0-23) if Hour exists."""
    if "Hour" not in df.columns or df["Hour"].dropna().empty:
        return pd.DataFrame()
    cols = get_numeric_pollutant_columns(df)
    if not cols:
        return pd.DataFrame()
    return df.groupby("Hour")[cols].mean().reset_index()


def get_monthly_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean pollutant concentrations by month (1-12) if Month exists."""
    if "Month" not in df.columns or df["Month"].dropna().empty:
        return pd.DataFrame()
    cols = get_numeric_pollutant_columns(df)
    if not cols:
        return pd.DataFrame()
    return df.groupby("Month")[cols].mean().reset_index()


def create_plotly_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
    """Generates an interactive Plotly correlation heatmap for available numeric columns."""
    if columns is None:
        columns = get_numeric_pollutant_columns(df)[:15] # Top 15 numeric columns

    if len(columns) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="At least two numeric pollutant columns are required to generate a correlation heatmap.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#64748B")
        )
        fig.update_layout(template="plotly_white")
        return fig

    corr = df[columns].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Cross-Pollutant & Variable Correlation Heatmap"
    )
    fig.update_layout(
        font=dict(family="sans-serif", size=11),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_trend_chart(df: pd.DataFrame, column: str, title: Optional[str] = None) -> go.Figure:
    """Creates an interactive time series line chart with a rolling average."""
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' not found in dataset.", x=0.5, y=0.5, showarrow=False)
        return fig

    has_dt = "DateTime" in df.columns and df["DateTime"].notna().sum() > 0
    valid_cols = ["DateTime", column] if has_dt else [column]
    valid_df = df.dropna(subset=valid_cols).copy()

    if valid_df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No valid observations for '{column}'.", x=0.5, y=0.5, showarrow=False)
        return fig

    x_axis = valid_df["DateTime"] if has_dt else valid_df.index
    x_title = "Date & Time" if has_dt else "Observation Sequence (Index)"

    valid_df["RollingAvg"] = valid_df[column].rolling(window=min(24, max(3, len(valid_df)//20)), min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=valid_df[column],
        mode="lines",
        name="Observation",
        line=dict(color="rgba(59, 130, 246, 0.4)", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=valid_df["RollingAvg"],
        mode="lines",
        name="Moving Average",
        line=dict(color="#10B981", width=2.5)
    ))

    fig.update_layout(
        title=title or f"Concentration Trend: {column}",
        xaxis_title=x_title,
        yaxis_title="Concentration / Level",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_hourly_diurnal(df: pd.DataFrame) -> go.Figure:
    """Creates an interactive diurnal profile showing hourly variations."""
    if "Hour" not in df.columns or df["Hour"].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Hourly diurnal analysis is unavailable (the dataset contains dates/observations only, no hourly timestamps).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#64748B")
        )
        fig.update_layout(template="plotly_white")
        return fig

    hourly_df = get_hourly_profiles(df)
    if hourly_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No numeric pollutant columns available for diurnal profile.", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()
    palette = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    
    # Priority targets to plot
    priority_targets = ["CO(GT)", "CO", "NO2(GT)", "NO2", "PM2.5", "PM10", "C6H6(GT)", "C6H6", "O3", "SO2"]
    plotted = 0

    for col in priority_targets:
        if col in hourly_df.columns:
            color = palette[plotted % len(palette)]
            fig.add_trace(go.Scatter(
                x=hourly_df["Hour"],
                y=hourly_df[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=2.5)
            ))
            plotted += 1
            if plotted >= 4:
                break

    # If none of the priority targets were present, plot first 3 available numeric columns
    if plotted == 0:
        for col in get_numeric_pollutant_columns(df)[:3]:
            if col in hourly_df.columns:
                fig.add_trace(go.Scatter(
                    x=hourly_df["Hour"],
                    y=hourly_df[col],
                    mode="lines+markers",
                    name=col
                ))
                plotted += 1

    fig.update_layout(
        title="Diurnal Air Pollution Profile (Hourly Variations & Commute Peaks)",
        xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Pollutant Concentration"),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Generates an interactive histogram and boxplot for a given column."""
    if column not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{column}' not found in dataset.", x=0.5, y=0.5, showarrow=False)
        return fig

    valid_series = pd.to_numeric(df[column], errors="coerce").dropna()
    fig = px.histogram(
        valid_series,
        x=valid_series.values,
        marginal="box",
        nbins=40,
        color_discrete_sequence=["#059669"],
        title=f"Statistical Distribution & Outlier Boxplot: {column}"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=column,
        yaxis_title="Observation Count",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def save_static_eda_charts(df: pd.DataFrame) -> Dict[str, str]:
    """
    Renders and saves publication-ready static charts (PNG) to visualizations/
    for inclusion in downloadable PDF reports. Adapts to available columns.
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    saved_files = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    numeric_cols = get_numeric_pollutant_columns(df)

    # 1. Correlation Heatmap
    if len(numeric_cols) >= 2:
        cols_to_plot = numeric_cols[:10]
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df[cols_to_plot].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Cross-Pollutant & Environmental Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        corr_path = os.path.join(VIS_DIR, "correlation_matrix.png")
        fig.savefig(corr_path, dpi=200)
        plt.close(fig)
        saved_files["correlation_matrix"] = corr_path

    # 2. Hourly Profile Chart
    if "Hour" in df.columns and df["Hour"].dropna().nunique() > 1:
        hourly_df = get_hourly_profiles(df)
        if not hourly_df.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            plotted = False
            for target, color in [("CO(GT)", "#d9534f"), ("CO", "#d9534f"), ("C6H6(GT)", "#2e6da4"), ("PM2.5", "#f0ad4e"), ("NO2(GT)", "#5bc0de")]:
                if target in hourly_df.columns:
                    ax.plot(hourly_df["Hour"], hourly_df[target], marker="o", linewidth=2.0, label=target, color=color)
                    plotted = True
            if plotted:
                ax.set_xlabel("Hour of Day (0-23)", fontweight="bold")
                ax.set_ylabel("Concentration", fontweight="bold")
                ax.set_title("Diurnal Hourly Pollution Variation", fontsize=12, fontweight="bold", pad=10)
                ax.set_xticks(range(0, 24, 2))
                ax.legend(frameon=True)
                plt.tight_layout()
                hourly_path = os.path.join(VIS_DIR, "hourly_profile.png")
                fig.savefig(hourly_path, dpi=200)
                plt.close(fig)
                saved_files["hourly_profile"] = hourly_path

    # 3. Boxplots of primary numeric pollutants
    if numeric_cols:
        cols_box = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(cols_box), figsize=(3.5 * len(cols_box), 4))
        if len(cols_box) == 1:
            axes = [axes]
        for i, col in enumerate(cols_box):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            sns.boxplot(y=s, ax=axes[i], color="#5cb85c")
            axes[i].set_title(col, fontweight="bold")
            axes[i].set_ylabel("Value")
        plt.suptitle("Pollutant Distributions & Outliers (IQR Method)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        box_path = os.path.join(VIS_DIR, "pollutant_boxplots.png")
        fig.savefig(box_path, dpi=200)
        plt.close(fig)
        saved_files["boxplots"] = box_path

    return saved_files
