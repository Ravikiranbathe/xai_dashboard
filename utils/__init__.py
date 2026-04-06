"""XAI Dashboard utilities package."""
from .data_processing import (
    load_and_preview,
    handle_missing_values,
    encode_features,
    scale_features,
    get_feature_summary,
)
from .train_model import (
    split_data,
    train_all_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
)
from .explain_model import (
    compute_shap_values,
    plot_feature_importance,
    plot_shap_summary,
    plot_shap_waterfall,
    generate_plain_language_explanation,
)

__all__ = [
    "load_and_preview", "handle_missing_values", "encode_features",
    "scale_features", "get_feature_summary",
    "split_data", "train_all_models", "evaluate_model",
    "plot_confusion_matrix", "plot_roc_curve", "plot_model_comparison",
    "compute_shap_values", "plot_feature_importance", "plot_shap_summary",
    "plot_shap_waterfall", "generate_plain_language_explanation",
]
