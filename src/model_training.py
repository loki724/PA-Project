"""
Model Training and Champion Selection Pipeline.
Trains Linear Regression, Random Forest, and Gradient Boosting Regressors
using strictly chronological splits, evaluates performance, and serializes the best model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.evaluation import calculate_metrics, save_static_evaluation_charts

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: Any,
    split_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Trains Linear Regression, Random Forest, and Gradient Boosting models.
    Compares their performance on the chronological test set and selects the champion.
    Serializes models and saves metadata to models/ directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    models_to_train = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            random_state=42
        )
    }

    results = {}
    comparison_rows = []
    trained_estimators = {}
    predictions_dict = {}

    for name, model in models_to_train.items():
        # Train model
        model.fit(X_train, y_train)
        trained_estimators[name] = model

        # Evaluate on test set
        y_pred = model.predict(X_test)
        predictions_dict[name] = y_pred
        metrics = calculate_metrics(y_test.values, y_pred)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred
        }

        comparison_rows.append({
            "Model": name,
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

        # Save individual model
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{clean_name}.pkl"))

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    # Champion model is the one with lowest RMSE / highest R2
    champion_name = comparison_df.iloc[0]["Model"]
    champion_model = trained_estimators[champion_name]
    champion_metrics = results[champion_name]["metrics"]
    champion_predictions = predictions_dict[champion_name]

    # Compute Feature Importances
    features = split_info["feature_names"]
    if hasattr(champion_model, "feature_importances_"):
        raw_importances = champion_model.feature_importances_
    elif hasattr(champion_model, "coef_"):
        raw_importances = np.abs(champion_model.coef_)
        if raw_importances.sum() > 0:
            raw_importances = raw_importances / raw_importances.sum()
    else:
        raw_importances = np.ones(len(features)) / len(features)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": [float(round(imp, 4)) for imp in raw_importances]
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Save champion package containing preprocessor + model + thresholds + metadata
    champion_package = {
        "champion_name": champion_name,
        "model": champion_model,
        "preprocessor": preprocessor,
        "feature_names": features,
        "metrics": champion_metrics,
        "co_median_threshold": split_info["co_median_threshold"],
        "co_90th_percentile_threshold": split_info["co_90th_percentile_threshold"],
        "split_info": {
            "train_samples": split_info["train_samples"],
            "test_samples": split_info["test_samples"],
            "train_dates": split_info["train_date_range"],
            "test_dates": split_info["test_date_range"]
        }
    }
    joblib.dump(champion_package, os.path.join(MODELS_DIR, "best_model.pkl"))

    # Save metadata JSON
    metadata = {
        "champion_model": champion_name,
        "comparison": comparison_df.to_dict(orient="records"),
        "champion_metrics": champion_metrics,
        "thresholds": {
            "median_co": split_info["co_median_threshold"],
            "co_90th_percentile": split_info["co_90th_percentile_threshold"]
        },
        "feature_importances": importance_df.to_dict(orient="records"),
        "training_metadata": {
            "train_samples": split_info["train_samples"],
            "test_samples": split_info["test_samples"],
            "features_used": features,
            "target": "CO(GT)"
        }
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Save static charts for report
    save_static_evaluation_charts(
        y_true=y_test.values,
        y_pred=champion_predictions,
        comparison_df=comparison_df,
        importance_df=importance_df,
        champion_name=champion_name
    )

    return {
        "results": results,
        "comparison_df": comparison_df,
        "champion_name": champion_name,
        "champion_metrics": champion_metrics,
        "champion_package": champion_package,
        "importance_df": importance_df,
        "y_test": y_test.values,
        "champion_predictions": champion_predictions
    }


def load_best_model_package() -> Dict[str, Any]:
    """Loads the serialized best model package from models/best_model.pkl."""
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Champion model not found at {path}. Please train models first.")
    return joblib.load(path)
