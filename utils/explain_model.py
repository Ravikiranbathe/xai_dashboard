"""
utils/explain_model.py
SHAP-based global and local explanations for trained models.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from matplotlib.patches import Patch
from typing import List

BG_COLOR   = "#0f0c29"
CARD_BG    = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT     = "#7efff5"
NEG_COLOR  = "#fb923c"
GRID_COLOR = "#ffffff18"


def _apply_dark(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BG_COLOR)
    for ax in fig.axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)


def compute_shap_values(model, X_sample: np.ndarray, model_name: str):
    """
    Compute SHAP values using the appropriate explainer.
    Returns (shap_values_2d, explainer) where shap_values_2d is shape (n, features).
    """
    try:
        if model_name in ("Random Forest", "XGBoost", "LightGBM"):
            explainer = shap.TreeExplainer(model)
            shap_out = explainer.shap_values(X_sample)
        else:
            explainer = shap.LinearExplainer(
                model, X_sample, feature_dependence="independent")
            shap_out = explainer.shap_values(X_sample)

        # Normalise to 2-D array
        if isinstance(shap_out, list):
            # Binary → keep class-1; multi-class → mean abs across classes
            if len(shap_out) == 2:
                shap_values = shap_out[1]
            else:
                shap_values = np.mean([np.abs(sv) for sv in shap_out], axis=0)
        else:
            shap_values = shap_out

        return shap_values, explainer

    except Exception:
        # Universal fallback — slow but always works
        predict_fn = (model.predict_proba
                      if hasattr(model, "predict_proba")
                      else model.predict)
        background = shap.sample(X_sample, min(50, len(X_sample)))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_out = explainer.shap_values(X_sample[:50])
        if isinstance(shap_out, list):
            shap_values = shap_out[1] if len(shap_out) == 2 else shap_out[0]
        else:
            shap_values = shap_out
        return shap_values, explainer


def plot_feature_importance(
    shap_values: np.ndarray, feature_names: List[str]
) -> plt.Figure:
    """Bar chart of mean |SHAP| per feature — global importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    
    # Ensure mean_abs is 1D
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.flatten()
    
    # Ensure we don't exceed available features
    top_n = min(20, len(feature_names), len(mean_abs))
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]
    
    # Filter sorted_idx to only include valid indices
    sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
    
    # Extract values for plotting
    values = mean_abs[sorted_idx]
    
    # Ensure values is 1D array
    if hasattr(values, 'flatten'):
        values = values.flatten()
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(values) * 0.38)))
    colors = [ACCENT if i == 0 else "#a78bfa" for i in range(len(values))]
    ax.barh(np.arange(len(values)), values, color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx],
                       color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance")
    ax.invert_yaxis()
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, axis="x")
    _apply_dark(fig)
    plt.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
) -> plt.Figure:
    """SHAP beeswarm summary plot."""
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    fig = plt.gcf()
    _apply_dark(fig)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    sample_idx: int,
    feature_names: List[str],
) -> plt.Figure:
    """Waterfall chart for a single prediction."""
    sv = shap_values[sample_idx]
    
    # Ensure sv is 1D
    if sv.ndim > 1:
        sv = sv.flatten()
    
    # Ensure we don't exceed available features
    top_n = min(15, len(feature_names), len(sv))
    sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    
    # Filter sorted_idx to only include valid indices
    sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
    
    # Extract values for plotting
    values = sv[sorted_idx]
    if hasattr(values, 'flatten'):
        values = values.flatten()

    fig, ax = plt.subplots(figsize=(8, max(4, len(values) * 0.45)))
    colors = [ACCENT if v >= 0 else NEG_COLOR for v in values]
    ax.barh(np.arange(len(values)), values, color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx],
                       color=TEXT_COLOR, fontsize=9)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(f"Local Explanation — Sample #{sample_idx}")
    ax.invert_yaxis()
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, axis="x")

    legend_elements = [
        Patch(facecolor=ACCENT,    label="Increases prediction"),
        Patch(facecolor=NEG_COLOR, label="Decreases prediction"),
    ]
    ax.legend(handles=legend_elements, facecolor=CARD_BG,
              labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    _apply_dark(fig)
    plt.tight_layout()
    return fig


def generate_plain_language_explanation(
    shap_values: np.ndarray,
    sample_idx: int,
    feature_names: List[str],
    predicted_class: str,
) -> str:
    """Return a human-readable explanation for a single prediction."""
    sv = shap_values[sample_idx]
    
    # Ensure sv is 1D
    if sv.ndim > 1:
        sv = sv.flatten()
    
    top_n = min(5, len(feature_names), len(sv))
    sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    
    # Filter sorted_idx to only include valid indices
    sorted_idx = sorted_idx[sorted_idx < len(feature_names)]

    pos_feats = [feature_names[i] for i in sorted_idx if sv[i] > 0]
    neg_feats = [feature_names[i] for i in sorted_idx if sv[i] < 0]

    lines = [f"The model predicted **{predicted_class}**."]
    if pos_feats:
        lines.append(
            f"Features that **increased** this prediction: {', '.join(pos_feats)}.")
    if neg_feats:
        lines.append(
            f"Features that **decreased** this prediction: {', '.join(neg_feats)}.")
    lines.append(
        "The bar chart above shows the magnitude of each feature's influence — "
        "longer bars mean stronger impact.")
    return " ".join(lines)
