"""
Preprocessing Module for Air Quality Data.
Performs intelligent schema detection, robust temporal column identification (Date, Time, Timestamp),
safe column normalization, leakage-free imputation pipelines, and outlier audits.
Supports both authentic UCI Air Quality and arbitrary general air-quality datasets.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CLEANED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "air_quality_cleaned.csv")

# UCI Signature Columns
UCI_SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
UCI_METEO = ["T", "RH", "AH"]
UCI_TARGET = "CO(GT)"
UCI_POLLUTANTS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
UCI_ALL_NUMERIC = UCI_POLLUTANTS + UCI_SENSORS + UCI_METEO

# General Pollutant & Meteorological Canonical Mapping
POLLUTANT_PATTERNS = {
    "CO": [r"^co(\(gt\))?$", r"^carbon_?monoxide", r"^co_?ppm", r"^co_?mgm3"],
    "NO2": [r"^no2(\(gt\))?$", r"^nitrogen_?dioxide", r"^pt08\.s4\(no2\)"],
    "NOx": [r"^nox(\(gt\))?$", r"^nitrogen_?oxides", r"^pt08\.s3\(nox\)"],
    "C6H6": [r"^c6h6(\(gt\))?$", r"^benzene", r"^pt08\.s2\(nmhc\)"],
    "NMHC": [r"^nmhc(\(gt\))?$", r"^non_?methane_?hydrocarbons"],
    "O3": [r"^o3$", r"^ozone", r"^pt08\.s5\(o3\)"],
    "PM2.5": [r"^pm2\.?5$", r"^pm_?25", r"^particulate_?matter_?2\.?5"],
    "PM10": [r"^pm10$", r"^pm_?10", r"^particulate_?matter_?10"],
    "SO2": [r"^so2$", r"^sulfur_?dioxide"],
    "AQI": [r"^aqi$", r"^aqi_?value", r"^air_?quality_?index"],
    "Temperature": [r"^t$", r"^temp$", r"^temperature$"],
    "Relative Humidity": [r"^rh$", r"^humidity$", r"^relative_?humidity$"],
    "Absolute Humidity": [r"^ah$", r"^absolute_?humidity$"]
}


def normalize_col_name(col: str) -> str:
    """Strips whitespace, BOM, and converts to clean lower string for matching."""
    c = str(col).strip().replace("\ufeff", "")
    return re.sub(r"[\s\-_]+", "_", c).lower()


def detect_dataset_schema(df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    """
    Detects whether the dataset is UCI Air Quality or General Air Quality,
    and returns a mapping of standard keys to the actual dataframe column names.
    """
    col_map = {}
    norm_to_orig = {normalize_col_name(c): c for c in df.columns}

    for standard_name, patterns in POLLUTANT_PATTERNS.items():
        for norm_col, orig_col in norm_to_orig.items():
            for pat in patterns:
                if re.search(pat, norm_col, re.IGNORECASE):
                    col_map[standard_name] = orig_col
                    break
            if standard_name in col_map:
                break

    # Check for UCI sensor signatures
    uci_sensor_count = sum(1 for s in UCI_SENSORS if s in df.columns)
    if uci_sensor_count >= 3 or "CO(GT)" in df.columns:
        schema_type = "uci_air_quality"
    else:
        schema_type = "general_air_quality"

    return schema_type, col_map


def detect_and_parse_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robustly identifies date and time columns from diverse naming conventions:
    - Combined: Timestamp, DateTime, Datetime, Date_Time, datetime
    - Separate Date: Date, date, DATE, Observation_Date
    - Separate Time: Time, time, TIME, Observation_Time
    
    Returns enriched dataframe with DateTime (if available), along with temporal metadata:
    has_datetime, has_time, has_date, and temporal_status message.
    Never creates fake timestamps.
    """
    df_out = df.copy()
    norm_cols = {normalize_col_name(c): c for c in df_out.columns}

    # Combined candidate check
    combined_candidates = ["datetime", "date_time", "timestamp", "time_stamp", "timedate", "observation_datetime"]
    dt_col = None
    for cand in combined_candidates:
        if cand in norm_cols:
            dt_col = norm_cols[cand]
            break

    # Date candidate check
    date_candidates = ["date", "observation_date", "obs_date", "reading_date"]
    date_col = None
    for cand in date_candidates:
        if cand in norm_cols:
            date_col = norm_cols[cand]
            break

    # Time candidate check
    time_candidates = ["time", "observation_time", "obs_time", "reading_time", "time_gmt", "time_local"]
    time_col = None
    for cand in time_candidates:
        if cand in norm_cols:
            time_col = norm_cols[cand]
            break

    temporal_info = {
        "has_datetime": False,
        "has_time": False,
        "has_date": False,
        "detected_date_col": date_col,
        "detected_time_col": time_col,
        "detected_combined_col": dt_col,
        "status_message": ""
    }

    # Case 1: Combined DateTime column exists
    if dt_col is not None:
        parsed_dt = pd.to_datetime(df_out[dt_col], errors="coerce", dayfirst=True)
        if parsed_dt.notna().sum() > 0:
            df_out["DateTime"] = parsed_dt
            temporal_info["has_datetime"] = True
            temporal_info["has_date"] = True
            # Determine if time is meaningful (not all 00:00:00)
            if (df_out["DateTime"].dt.hour != 0).any() or (df_out["DateTime"].dt.minute != 0).any():
                temporal_info["has_time"] = True
                temporal_info["status_message"] = f"Using combined date-time column '{dt_col}'."
            else:
                temporal_info["has_time"] = False
                temporal_info["status_message"] = f"Using date column '{dt_col}' (time components not present)."
            return df_out, temporal_info

    # Case 2: Separate Date and Time columns exist
    if date_col is not None and time_col is not None:
        # Handle formats where time uses dots (e.g. 18.00.00 in Italian UCI dataset)
        time_clean = df_out[time_col].astype(str).str.replace(".", ":", regex=False)
        combined_str = df_out[date_col].astype(str) + " " + time_clean
        
        parsed_dt = pd.to_datetime(combined_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if parsed_dt.isna().sum() > len(df_out) * 0.5:
            # Fallback to general parser
            parsed_dt = pd.to_datetime(combined_str, errors="coerce", dayfirst=True)

        if parsed_dt.notna().sum() > 0:
            df_out["DateTime"] = parsed_dt
            temporal_info["has_datetime"] = True
            temporal_info["has_date"] = True
            temporal_info["has_time"] = True
            temporal_info["status_message"] = f"Combined separate columns '{date_col}' and '{time_col}' into DateTime."
            return df_out, temporal_info

    # Case 3: Only Date column exists (No Time column)
    if date_col is not None:
        parsed_d = pd.to_datetime(df_out[date_col], errors="coerce", dayfirst=True)
        if parsed_d.notna().sum() > 0:
            df_out["DateTime"] = parsed_d
            temporal_info["has_datetime"] = True
            temporal_info["has_date"] = True
            temporal_info["has_time"] = False
            temporal_info["status_message"] = (
                f"Date column '{date_col}' detected without time component. "
                "Date-based trends are enabled; hourly diurnal analysis is disabled."
            )
            return df_out, temporal_info

    # Case 4: Neither Date nor Time exists
    temporal_info["has_datetime"] = False
    temporal_info["has_date"] = False
    temporal_info["has_time"] = False
    temporal_info["status_message"] = (
        "No date or timestamp column detected. Temporal and diurnal analyses are disabled. "
        "Observations will be sequenced by row index."
    )
    return df_out, temporal_info


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    Computes IQR-based outlier bounds and flags for specified numeric columns.
    Preserves actual values without dropping genuine high-pollution peaks.
    """
    outlier_summary = {}
    total_outliers = 0

    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            outlier_summary[col] = {"q1": np.nan, "q3": np.nan, "iqr": np.nan, "lower_bound": np.nan, "upper_bound": np.nan, "count": 0, "pct": 0.0}
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        count = int(outlier_mask.sum())
        pct = float((count / len(series)) * 100) if len(series) > 0 else 0.0
        total_outliers += count

        outlier_summary[col] = {
            "q1": float(round(q1, 3)),
            "q3": float(round(q3, 3)),
            "iqr": float(round(iqr, 3)),
            "lower_bound": float(round(lower_bound, 3)),
            "upper_bound": float(round(upper_bound, 3)),
            "count": count,
            "pct": float(round(pct, 2))
        }

    outlier_summary["_total_count"] = total_outliers
    return outlier_summary


def clean_and_prepare_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust Preprocessing Pipeline:
    1. Safe column normalization.
    2. Dataset schema identification (UCI vs General).
    3. Temporal column detection and parsing (handles Combined, Separate, Date-only, or None).
    4. Numeric conversion of all pollutant and environmental measurements.
    5. UCI -200 sentinel replacement only where applicable.
    6. IQR outlier detection audit without dropping genuine peaks.
    7. Time-based feature engineering (Hour, Month, DayOfWeek) when temporal data exists.
    8. Exports cleaned dataset to CSV.
    """
    df = raw_df.copy()
    original_records = len(df)

    if original_records == 0:
        raise ValueError("Cannot clean an empty dataset.")

    # Detect schema type and column mappings
    schema_type, col_mapping = detect_dataset_schema(df)

    # Replace sentinel -200 with NaN where applicable
    if schema_type == "uci_air_quality":
        df = df.replace([-200, -200.0, "-200"], np.nan)
    else:
        # For general datasets, only replace -200 if values of -200 actually exist in numeric columns
        for col in df.columns:
            if df[col].dtype.kind in "ifc":
                if (df[col] == -200).any():
                    df[col] = df[col].replace(-200, np.nan)

    # Detect and parse temporal columns
    df, temporal_info = detect_and_parse_temporal(df)

    # Sort chronologically if DateTime is available
    if temporal_info["has_datetime"] and "DateTime" in df.columns:
        df = df.sort_values("DateTime").reset_index(drop=True)
        duplicate_count = int(df.duplicated(subset=["DateTime"]).sum())
        if duplicate_count > 0:
            df = df.drop_duplicates(subset=["DateTime"]).reset_index(drop=True)
    else:
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)

    # Feature engineering for temporal data
    if temporal_info["has_time"] and "DateTime" in df.columns:
        df["Hour"] = df["DateTime"].dt.hour
    if temporal_info["has_date"] and "DateTime" in df.columns:
        df["Month"] = df["DateTime"].dt.month
        df["DayOfWeek"] = df["DateTime"].dt.dayofweek
        df["IsWeekend"] = df["DayOfWeek"].apply(lambda d: 1 if d >= 5 else 0)

    # Convert all pollutant and measurement candidate columns to numeric
    numeric_cols = []
    non_numeric_cols = []
    date_col_names = [temporal_info.get("detected_date_col"), temporal_info.get("detected_time_col"), temporal_info.get("detected_combined_col"), "DateTime"]

    for col in df.columns:
        if col in date_col_names or col in ["Hour", "Month", "DayOfWeek", "IsWeekend"]:
            continue
        # Check if column is numeric or can be safely converted to numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        # If column has valid numeric numbers and isn't predominantly text
        if converted.notna().sum() > len(df) * 0.1:
            df[col] = converted
            numeric_cols.append(col)
        else:
            non_numeric_cols.append(col)

    # Missing value audit
    missing_by_col = df[numeric_cols].isna().sum().to_dict() if numeric_cols else {}
    total_missing = int(df[numeric_cols].isna().sum().sum()) if numeric_cols else 0

    # Detect outliers using IQR method for audit report
    outlier_report = detect_outliers_iqr(df, numeric_cols)

    date_range = {"start": "N/A", "end": "N/A"}
    if temporal_info["has_datetime"] and "DateTime" in df.columns:
        min_dt = df["DateTime"].min()
        max_dt = df["DateTime"].max()
        date_range = {
            "start": str(min_dt) if pd.notna(min_dt) else "N/A",
            "end": str(max_dt) if pd.notna(max_dt) else "N/A"
        }

    cleaning_report = {
        "original_records": original_records,
        "cleaned_records": len(df),
        "duplicate_records": duplicate_count,
        "schema_type": schema_type,
        "temporal_info": temporal_info,
        "detected_pollutants": col_mapping,
        "numeric_columns": numeric_cols,
        "non_numeric_columns": non_numeric_cols,
        "total_missing_values": total_missing,
        "missing_by_column": missing_by_col,
        "outliers_detected": outlier_report.get("_total_count", 0),
        "outlier_details": outlier_report,
        "date_range": date_range
    }

    # Save cleaned data to disk
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(CLEANED_CSV_PATH, index=False)

    return df, cleaning_report


def prepare_ml_splits(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[Pipeline], Dict[str, Any]]:
    """
    Prepares Train/Test splits for Machine Learning regression.
    Evaluates whether the dataset contains the required features and target for the
    sensor-based Carbon Monoxide estimation model.
    
    If features are available:
    - Chronological 80/20 train/test split
    - Pipeline fitted strictly on X_train to prevent leakage
    - Returns splits, pipeline, and split metadata with is_trainable = True
    
    If required features/target are missing:
    - Returns is_trainable = False with informative explanation.
    - Does NOT crash.
    """
    # Check if target is present
    target_candidate = None
    if UCI_TARGET in df.columns:
        target_candidate = UCI_TARGET
    elif "CO" in df.columns and pd.to_numeric(df["CO"], errors="coerce").notna().sum() > 50:
        target_candidate = "CO"

    # Check required sensor and meteorological features
    sensor_features = [s for s in UCI_SENSORS if s in df.columns]
    meteo_features = [m for m in UCI_METEO if m in df.columns]
    time_features = [t for t in ["Hour", "Month", "DayOfWeek"] if t in df.columns]

    available_features = sensor_features + meteo_features + time_features

    # Requirements for sensor-based regression: target present + at least 3 chemical sensors
    if target_candidate is None or len(sensor_features) < 3:
        missing_reasons = []
        if target_candidate is None:
            missing_reasons.append(f"Target column '{UCI_TARGET}' or 'CO' not found.")
        if len(sensor_features) < 3:
            missing_reasons.append(
                f"Requires metal oxide chemical sensor columns (found {len(sensor_features)}/5: {sensor_features})."
            )

        split_info = {
            "is_trainable": False,
            "reason": " ".join(missing_reasons),
            "target": target_candidate,
            "available_features": available_features,
            "required_features": UCI_SENSORS + UCI_METEO,
            "total_supervised_samples": 0,
            "train_samples": 0,
            "test_samples": 0,
            "co_median_threshold": 2.2,
            "co_90th_percentile_threshold": 4.2
        }
        return None, None, None, None, None, split_info

    # Filter rows with non-null target
    ml_df = df.dropna(subset=[target_candidate]).copy()
    if len(ml_df) < 50:
        split_info = {
            "is_trainable": False,
            "reason": f"Insufficient non-null target records ({len(ml_df)} samples found; minimum 50 required).",
            "target": target_candidate,
            "co_median_threshold": 2.2,
            "co_90th_percentile_threshold": 4.2
        }
        return None, None, None, None, None, split_info

    if "DateTime" in ml_df.columns and ml_df["DateTime"].notna().sum() > 0:
        ml_df = ml_df.sort_values("DateTime").reset_index(drop=True)
    else:
        ml_df = ml_df.reset_index(drop=True)

    X = ml_df[available_features].copy()
    y = ml_df[target_candidate].copy()

    split_idx = int(len(ml_df) * (1 - test_ratio))
    X_train_raw = X.iloc[:split_idx].copy()
    X_test_raw = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    train_dates = ("N/A", "N/A")
    test_dates = ("N/A", "N/A")
    if "DateTime" in ml_df.columns and ml_df["DateTime"].notna().sum() > 0:
        train_dates = (str(ml_df["DateTime"].iloc[0]), str(ml_df["DateTime"].iloc[split_idx - 1]))
        test_dates = (str(ml_df["DateTime"].iloc[split_idx]), str(ml_df["DateTime"].iloc[-1]))

    # Pipeline fitted strictly on training data
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pipeline.fit(X_train_raw)

    X_train_transformed = pd.DataFrame(
        pipeline.transform(X_train_raw),
        columns=available_features,
        index=X_train_raw.index
    )
    X_test_transformed = pd.DataFrame(
        pipeline.transform(X_test_raw),
        columns=available_features,
        index=X_test_raw.index
    )

    co_p90 = float(np.percentile(y_train.dropna(), 90))
    co_median = float(np.percentile(y_train.dropna(), 50))

    split_info = {
        "is_trainable": True,
        "target": target_candidate,
        "feature_names": available_features,
        "total_supervised_samples": len(ml_df),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "train_date_range": train_dates,
        "test_date_range": test_dates,
        "co_median_threshold": co_median,
        "co_90th_percentile_threshold": co_p90
    }

    return X_train_transformed, X_test_transformed, y_train, y_test, pipeline, split_info
