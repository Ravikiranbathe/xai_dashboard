"""
utils/data_processing.py
Handles loading, cleaning, encoding, and scaling of datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List


def load_and_preview(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numeric columns  → median
      - Categorical cols → mode (most frequent)
    """
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def encode_features(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "Label Encoding",
) -> Tuple[pd.DataFrame, np.ndarray, Dict, List[str], List[str]]:
    """
    Encode categorical features and separate target.

    Returns
    -------
    X_encoded   : feature DataFrame (encoded)
    y           : target array (int)
    encoders    : dict of fitted LabelEncoders keyed by column name
    cat_cols    : list of original categorical feature columns
    num_cols    : list of numeric feature columns
    """
    df = df.copy()

    # ── Encode target ──────────────────────────────────────────────────────
    if df[target_col].dtype == object or str(df[target_col].dtype) == "category":
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target_col].astype(str))
    else:
        le_target = LabelEncoder()
        le_target.fit(df[target_col])
        y = df[target_col].values.astype(int)

    encoders: Dict = {"target": le_target}

    # ── Separate features ──────────────────────────────────────────────────
    X = df.drop(columns=[target_col]).copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # ── Encode features ────────────────────────────────────────────────────
    if strategy == "Label Encoding":
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    else:  # One-Hot Encoding
        # First store label encoders for inverse-transform in prediction UI
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            encoders[col] = le
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y, encoders, cat_cols, num_cols


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Apply StandardScaler to feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary (dtype, missing %, unique count)."""
    summary = pd.DataFrame({
        "dtype":   df.dtypes,
        "missing %": df.isnull().mean().mul(100).round(2),
        "unique":  df.nunique(),
    })
    summary.index.name = "column"
    return summary.reset_index()
