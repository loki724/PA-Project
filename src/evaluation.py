"""
Evaluation and Metrics Visualization Module.
Calculates MAE, MSE, RMSE, R² Score, and generates both interactive Plotly
and publication-ready static charts for models and residuals.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes genuine regression evaluation metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of Determination
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(round(mae, 4)),
        "MSE": float(round(mse, 4)),
        "RMSE": float(round(rmse, 4)),
        "R2": float(round(r2, 4))
    }


def create_plotly_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Champion Model"
) -> go.Figure:
    """Generates an interactive scatter plot of Actual vs Predicted CO with y = x reference line."""
    df_eval = pd.DataFrame({"Actual CO (mg/m³)": y_true, "Predicted CO (mg/m³)": y_pred})
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig = px.scatter(
        df_eval,
        x="Actual CO (mg/m³)",
        y="Predicted CO (mg/m³)",
        opacity=0.6,
        color_discrete_sequence=["#2563EB"],
        title=f"Actual vs Predicted CO(GT) Concentration ({model_name})"
    )
    # Add ideal 45-degree reference line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Prediction (y = x)",
        line=dict(color="#DC2626", dash="dash", width=2)
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Creates a grouped bar chart comparing MAE, RMSE, and R² across trained models."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["MAE"],
        name="MAE (Lower is Better)",
        marker_color="#3B82F6"
    ))
    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["RMSE"],
        name="RMSE (Lower is Better)",
        marker_color="#F59E0B"
    ))
    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["R2"],
        name="R² Score (Higher is Better)",
        marker_color="#10B981"
    ))

    fig.update_layout(
        title="Machine Learning Model Performance Comparison (Chronological Held-Out Test Set)",
        barmode="group",
        template="plotly_white",
        yaxis_title="Metric Value",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Generates an interactive horizontal bar chart of feature importances."""
    importance_sorted = importance_df.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importance_sorted,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Predictor Feature Importance Ranking"
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_plotly_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Champion Model"
) -> go.Figure:
    """Generates an interactive residual distribution plot (errors = Actual - Predicted)."""
    residuals = y_true - y_pred
    fig = px.histogram(
        residuals,
        nbins=40,
        marginal="rug",
        color_discrete_sequence=["#6366F1"],
        title=f"Prediction Residuals Distribution ({model_name})"
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Residual Error (Actual - Predicted CO) [mg/m³]",
        yaxis_title="Frequency",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def save_static_evaluation_charts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    comparison_df: pd.DataFrame,
    importance_df: Optional[pd.DataFrame] = None,
    champion_name: str = "Random Forest Regressor"
) -> Dict[str, str]:
    """
    Renders and saves high-resolution static evaluation charts to visualizations/
    for inclusion in downloadable PDF reports.
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    saved = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Model Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(comparison_df))
    width = 0.25

    ax.bar(x - width, comparison_df["MAE"], width, label="MAE", color="#3498db")
    ax.bar(x, comparison_df["RMSE"], width, label="RMSE", color="#e67e22")
    ax.bar(x + width, comparison_df["R2"], width, label="R² Score", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], fontweight="bold")
    ax.set_ylabel("Score / Error", fontweight="bold")
    ax.set_title("Model Leaderboard on Chronological Test Split", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    comp_path = os.path.join(VIS_DIR, "model_comparison.png")
    fig.savefig(comp_path, dpi=200)
    plt.close(fig)
    saved["model_comparison"] = comp_path

    # 2. Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#1f77b4", edgecolors="none", s=20)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="y = x (Perfect)")
    ax.set_xlabel("Actual CO(GT) [mg/m³]", fontweight="bold")
    ax.set_ylabel("Predicted CO(GT) [mg/m³]", fontweight="bold")
    ax.set_title(f"Actual vs Predicted CO ({champion_name})", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    avp_path = os.path.join(VIS_DIR, "actual_vs_predicted.png")
    fig.savefig(avp_path, dpi=200)
    plt.close(fig)
    saved["actual_vs_predicted"] = avp_path

    # 3. Feature Importance
    if importance_df is not None and len(importance_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sorted_imp = importance_df.sort_values(by="Importance", ascending=True)
        ax.barh(sorted_imp["Feature"], sorted_imp["Importance"], color="#27ae60")
        ax.set_xlabel("Relative Importance Score", fontweight="bold")
        ax.set_title(f"Sensor & Meteorological Feature Importance ({champion_name})", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        imp_path = os.path.join(VIS_DIR, "feature_importance.png")
        fig.savefig(imp_path, dpi=200)
        plt.close(fig)
        saved["feature_importance"] = imp_path

    # 4. Residuals Histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    residuals = y_true - y_pred
    ax.hist(residuals, bins=35, color="#8e44ad", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.set_xlabel("Residual Error (Actual - Predicted) [mg/m³]", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Prediction Error / Residual Distribution ({champion_name})", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    res_path = os.path.join(VIS_DIR, "residuals.png")
    fig.savefig(res_path, dpi=200)
    plt.close(fig)
    saved["residuals"] = res_path

    return saved
