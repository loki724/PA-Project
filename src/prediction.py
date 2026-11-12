"""
Interactive Prediction Module.
Validates real-time sensor inputs, transforms them through the fitted preprocessor pipeline,
estimates CO(GT) concentration using the champion ML model, computes the risk category,
and suggests contextual abatement measures.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from src.recommendation import get_abatement_recommendations
from src.anomaly_detection import classify_co_risk

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Sensor and weather physical bounds for validation
FEATURE_RANGES = {
    "PT08.S1(CO)": (300.0, 3000.0, 1098.0),
    "PT08.S2(NMHC)": (300.0, 2500.0, 946.0),
    "PT08.S3(NOx)": (300.0, 2800.0, 835.0),
    "PT08.S4(NO2)": (400.0, 3000.0, 1459.0),
    "PT08.S5(O3)": (200.0, 3000.0, 1022.0),
    "T": (-10.0, 50.0, 18.3),
    "RH": (0.0, 100.0, 48.9),
    "AH": (0.0, 3.0, 1.02),
    "Hour": (0, 23, 12),
    "Month": (1, 12, 6),
    "DayOfWeek": (0, 6, 2)
}


def validate_input_data(input_dict: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates user-submitted sensor measurements and meteorological variables.
    Checks values against known physical ranges and returns clean DataFrame with any warnings.
    """
    warnings = []
    clean_vals = {}

    for feature, (min_v, max_v, default_v) in FEATURE_RANGES.items():
        val = input_dict.get(feature, default_v)
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            val_float = float(default_v)
            warnings.append(f"Invalid input for {feature}; defaulted to {default_v}.")

        if val_float < min_v or val_float > max_v:
            warnings.append(
                f"Feature '{feature}' value {val_float:.2f} is outside typical range [{min_v}, {max_v}]."
            )

        clean_vals[feature] = val_float

    feature_cols = list(FEATURE_RANGES.keys())
    input_df = pd.DataFrame([clean_vals])[feature_cols]
    return input_df, warnings


def predict_co(
    input_dict: Dict[str, Any],
    model_package: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Runs full inference pipeline:
    1. Validates inputs
    2. Scales using fitted training preprocessor
    3. Predicts CO(GT) concentration in mg/m³
    4. Computes dataset-relative risk category
    5. Retrieves contextual abatement recommendations
    """
    if model_package is None:
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Train models before making predictions."
            )
        model_package = joblib.load(model_path)

    input_df, warnings = validate_input_data(input_dict)

    # Transform features using preprocessor fitted on training data
    preprocessor = model_package["preprocessor"]
    features_ordered = model_package["feature_names"]
    
    # Ensure column ordering matches training
    X_ordered = input_df[features_ordered]
    X_transformed = preprocessor.transform(X_ordered)
    X_transformed_df = pd.DataFrame(X_transformed, columns=features_ordered)

    # Predict using champion model
    model = model_package["model"]
    raw_pred = float(model.predict(X_transformed_df)[0])

    # Concentration cannot physically be negative
    predicted_co = max(0.0, round(raw_pred, 2))

    # Evaluate dataset-relative risk category
    co_median = model_package.get("co_median_threshold", 2.2)
    co_p90 = model_package.get("co_90th_percentile_threshold", 4.2)
    risk_category = classify_co_risk(predicted_co, median_threshold=co_median, p90_threshold=co_p90)

    # Retrieve tailored recommendations
    recs = get_abatement_recommendations(
        co=predicted_co,
        is_anomaly=(risk_category == "High"),
        risk_category=risk_category
    )

    badge_colors = {
        "Normal": "#10B981",    # Emerald
        "Elevated": "#F59E0B",  # Amber
        "High": "#EF4444"       # Red
    }

    return {
        "predicted_co": predicted_co,
        "unit": "mg/m³",
        "risk_category": risk_category,
        "badge_color": badge_colors.get(risk_category, "#6B7280"),
        "champion_model_name": model_package.get("champion_name", "Champion Model"),
        "co_thresholds": {
            "median": co_median,
            "90th_percentile": co_p90
        },
        "recommendations": recs,
        "warnings": warnings,
        "input_features": clean_vals if "clean_vals" in locals() else input_df.to_dict(orient="records")[0]
    }
