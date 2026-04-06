"""
utils/train_model.py
Trains, evaluates, and visualises ML models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, List, Tuple, Any

# ── Theme ──────────────────────────────────────────────────────────────────
PALETTE = {
    "Logistic Regression": "#7efff5",
    "Random Forest":       "#a78bfa",
    "XGBoost":             "#fb923c",
    "LightGBM":            "#4ade80",
}
BG_COLOR   = "#0f0c29"
CARD_BG    = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#ffffff18"


def _dark_style(fig: plt.Figure, ax) -> None:
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5)


def split_data(
    X: np.ndarray, y: np.ndarray,
    test_size: float = 0.2, random_state: int = 42,
) -> Tuple:
    try:
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)
    except ValueError:
        # Fallback when stratify is not possible (very small classes)
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)


def _build_model(name: str, random_state: int = 42):
    registry = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=150, random_state=random_state,
            eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=150, random_state=random_state, verbose=-1),
    }
    return registry[name]


def train_all_models(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    selected: List[str],
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """Train each selected model and return fitted models + evaluation dicts."""
    fitted_models: Dict[str, Any] = {}
    results: Dict[str, Dict] = {}
    n_classes = len(np.unique(y_train))

    for name in selected:
        model = _build_model(name, random_state)
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = None
        roc_val = None

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            try:
                if n_classes == 2:
                    roc_val = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_val = roc_auc_score(
                        y_test, y_prob, multi_class="ovr", average="macro")
            except Exception:
                roc_val = None

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred, average="macro"),
            "roc_auc":  roc_val,
            "y_test":   y_test,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "report":   classification_report(y_test, y_pred),
        }

    return fitted_models, results


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report":   classification_report(y_test, y_pred),
        "y_pred":   y_pred,
    }


def plot_confusion_matrix(y_test, y_pred) -> plt.Figure:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax,
                annot_kws={"color": "white", "size": 13})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=13)
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_prob, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    n_classes = y_prob.shape[1]

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        color = PALETTE.get(model_name, "#7efff5")
        ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_test)
        y_bin = label_binarize(y_test, classes=classes)
        colors = list(PALETTE.values())
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f"Class {cls} AUC={roc_auc:.2f}")

    ax.plot([0, 1], [0, 1], "w--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12)
    ax.legend(loc="lower right", facecolor=CARD_BG,
              labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_model_comparison(results: Dict) -> plt.Figure:
    names     = list(results.keys())
    accuracy  = [results[n]["accuracy"] for n in names]
    f1_scores = [results[n]["f1"] for n in names]
    roc_aucs  = [results[n]["roc_auc"] or 0 for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - width, accuracy,  width, label="Accuracy",  color="#7efff5", alpha=0.85)
    b2 = ax.bar(x,          f1_scores, width, label="F1 (macro)", color="#a78bfa", alpha=0.85)
    b3 = ax.bar(x + width,  roc_aucs,  width, label="ROC-AUC",   color="#fb923c", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison", fontsize=14)
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig
