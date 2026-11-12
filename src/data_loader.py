"""
Data Loader Module for Air Quality Datasets.
Supports authentic UCI Air Quality dataset and arbitrary user-uploaded CSV/XLSX files.
Features automatic delimiter detection (semicolon, comma, tab), encoding fallbacks,
UTF-8 BOM stripping, and robust error handling.
"""

import os
import zipfile
import urllib.request
import io
import csv
import pandas as pd
from typing import Tuple, Optional, Union, Any, Dict

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_CSV_PATH = os.path.join(RAW_DATA_DIR, "AirQualityUCI.csv")
DEFAULT_ZIP_PATH = os.path.join(RAW_DATA_DIR, "air_quality.zip")

DATASET_METADATA = {
    "source_name": "UCI Machine Learning Repository - Air Quality Dataset",
    "source_url": "https://archive.ics.uci.edu/dataset/360/air+quality",
    "download_url": UCI_DATASET_URL,
    "location": "Significantly polluted area in an Italian city (road-level sensor station)",
    "collection_period": "March 10, 2004 to April 4, 2005 (1 year, 9,358 hourly instances)",
    "sensors_used": "5 metal oxide chemical sensors (PT08.S1 to PT08.S5) + Reference analyzers",
    "citation": "Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59Q63"
}


def download_uci_dataset(dest_dir: str = RAW_DATA_DIR) -> str:
    """
    Downloads the authentic UCI Air Quality zip archive and extracts it.
    Returns the path to the extracted CSV file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "air_quality.zip")
    csv_path = os.path.join(dest_dir, "AirQualityUCI.csv")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10000:
        return csv_path

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(UCI_DATASET_URL, headers=headers)
    
    with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Extracted archive did not contain AirQualityUCI.csv in {dest_dir}")

    return csv_path


def detect_csv_dialect_and_encoding(buffer_or_path: Union[str, io.BytesIO]) -> Tuple[str, str, str]:
    """
    Inspects raw CSV content to detect delimiter (;, ,, \\t), decimal point (., ,),
    and character encoding (utf-8, utf-8-sig, latin1).
    """
    sample_bytes = b""
    if isinstance(buffer_or_path, str):
        with open(buffer_or_path, "rb") as f:
            sample_bytes = f.read(16384)
    else:
        # File-like / BytesIO object
        buffer_or_path.seek(0)
        sample_bytes = buffer_or_path.read(16384)
        buffer_or_path.seek(0)

    # Check for UTF-8 BOM
    encoding = "utf-8"
    if sample_bytes.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        try:
            sample_bytes.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            try:
                sample_bytes.decode("cp1252")
                encoding = "cp1252"
            except UnicodeDecodeError:
                encoding = "latin1"

    sample_text = sample_bytes.decode(encoding, errors="ignore")
    lines = [l for l in sample_text.splitlines() if l.strip()][:10]

    # Delimiter detection
    sep = ","
    decimal = "."

    if lines:
        header = lines[0]
        # Count candidate delimiters
        cnt_semi = header.count(";")
        cnt_comma = header.count(",")
        cnt_tab = header.count("\t")
        cnt_pipe = header.count("|")

        if cnt_semi > cnt_comma and cnt_semi > cnt_tab:
            sep = ";"
        elif cnt_tab > cnt_comma and cnt_tab > cnt_semi:
            sep = "\t"
        elif cnt_pipe > cnt_comma and cnt_pipe > cnt_semi:
            sep = "|"
        else:
            sep = ","

        # Decimal separator detection:
        # If delimiter is semicolon, commas in data rows frequently represent decimal points (European format)
        if sep == ";" and len(lines) > 1:
            data_sample = " ".join(lines[1:])
            # Look for digit,digit pattern like 2,6 or 11,9
            import re
            if re.search(r"\d,\d", data_sample):
                decimal = ","

    return sep, decimal, encoding


def load_raw_data(source: Optional[Union[str, io.BytesIO, Any]] = None) -> pd.DataFrame:
    """
    Loads air quality data from:
    1. Default authentic UCI dataset (downloaded automatically if not present)
    2. Uploaded Streamlit file buffer (CSV or XLSX)
    3. Explicit file path

    Robustly handles:
    - Delimiter detection (comma, semicolon, tab, pipe)
    - European decimal notation (',') vs standard ('.')
    - UTF-8 with BOM, Latin1, CP1252 encodings
    - Stripping leading/trailing whitespace and BOM characters from column headers
    - Dropping completely empty trailing columns and rows
    - Error logging with actual exception details
    """
    if source is None:
        if not os.path.exists(DEFAULT_CSV_PATH):
            source = download_uci_dataset(RAW_DATA_DIR)
        else:
            source = DEFAULT_CSV_PATH

    is_xlsx = False
    filename = ""
    if isinstance(source, str):
        filename = source.lower()
    elif hasattr(source, "name"):
        filename = getattr(source, "name", "").lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        is_xlsx = True

    try:
        if is_xlsx:
            df = pd.read_excel(source)
        else:
            sep, decimal, encoding = detect_csv_dialect_and_encoding(source)
            if hasattr(source, "seek"):
                source.seek(0)
            
            # Read CSV with detected formatting
            try:
                df = pd.read_csv(
                    source,
                    sep=sep,
                    decimal=decimal,
                    encoding=encoding,
                    skipinitialspace=True,
                    low_memory=False
                )
            except Exception:
                # Fallback to python engine with sep=None
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(
                    source,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    skipinitialspace=True
                )

        if df is None or df.empty:
            raise ValueError("The loaded dataset is empty (0 records found).")

        # Drop any trailing columns that are completely null (e.g. Unnamed: 15, Unnamed: 16)
        df = df.dropna(how="all", axis=1)

        # Drop rows where all values are null
        df = df.dropna(how="all", axis=0)

        # Clean and sanitize column names
        cleaned_columns = []
        for col in df.columns:
            c_str = str(col).strip().replace("\ufeff", "")
            cleaned_columns.append(c_str)
        df.columns = cleaned_columns

        # Reset index
        df = df.reset_index(drop=True)

        return df

    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset: {str(exc)}") from exc


def get_dataset_metadata() -> dict:
    """Returns dataset provenance and documentation metadata."""
    return DATASET_METADATA.copy()
