# 🧠 Explainable AI Dashboard

> An end-to-end Machine Learning dashboard with built-in explainability powered by SHAP — built for transparency, trust, and production quality.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📸 Screenshots

| Upload & Explore | Model Training | SHAP Explanations |
|---|---|---|
| *(Upload Data page)* | *(Train Model page)* | *(Explain Model page)* |

---

## 🎯 Project Overview

The **XAI Dashboard** is a Streamlit-based web application that makes machine learning interpretable and accessible. Upload any tabular CSV dataset, train multiple ML models with one click, compare their performance, and understand every prediction through SHAP-based global and local explanations.

---

## ✨ Features

### 📂 Data Handling
- CSV upload with automatic preview
- Missing value imputation (median for numeric, mode for categorical)
- Label Encoding and One-Hot Encoding support
- Feature scaling via `StandardScaler`
- Dataset summary, statistics, and feature summary table
- Target column validation (low-cardinality guard)

### 🏋️ Model Training

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |

Evaluation includes:
- Accuracy, F1 (macro), ROC-AUC
- Confusion Matrix heatmap
- ROC Curve (binary and multi-class)
- Full Classification Report
- Side-by-side model comparison chart

### 🔍 Explainable AI (XAI)
- **Global** — Feature importance bar chart + SHAP beeswarm summary plot
- **Local** — Per-sample SHAP waterfall chart with plain-language explanation
- TreeExplainer for RF / XGBoost / LightGBM (fast)
- LinearExplainer for Logistic Regression
- KernelExplainer fallback for any other model

### 🔮 Predictions
- Dynamic input form based on feature types
- Categorical inputs shown as dropdowns with original class labels
- Predicted class + confidence score
- Probability bar chart per class

### 💾 Model Persistence
- Trained models saved as `.joblib` files
- Scaler, encoders, and feature names persisted
- Ready for deployment or API wrapping

---

## 🧰 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost, LightGBM |
| XAI | SHAP |
| Visualisation | matplotlib, seaborn |
| Persistence | joblib |

---

## 🗂️ Folder Structure

```
xai_dashboard/
│
├── app.py                    # Main Streamlit entry point
│
├── utils/
│   ├── __init__.py
│   ├── data_processing.py    # Load, clean, encode, scale
│   ├── train_model.py        # Train, evaluate, plot metrics
│   └── explain_model.py      # SHAP global & local explanations
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── models/                   # Auto-created; stores .joblib artefacts
├── data/                     # Place sample datasets here (e.g. iris.csv)
│
├── .gitignore                # Git ignore rules
├── runtime.txt               # Python version for deployment
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Ravikiranbathe/xai_dashboard.git
cd xai_dashboard/xai_dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Quick Start with the bundled Iris dataset
1. Navigate to **Upload Data** → upload `data/iris.csv`
2. Select `species` as the target column → click **Process Dataset**
3. Go to **Train Model** → select all models → click **Train**
4. Go to **Predictions** → fill in petal/sepal values → click **Predict**
5. Go to **Explain Model** → click **Compute SHAP Values** → explore charts

---

## ☁️ Deployment

### Streamlit Cloud (Recommended - FREE)

This project is configured for easy deployment on Streamlit Cloud:

1. Push this repo to GitHub (already done!)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Connect your GitHub account and select `Ravikiranbathe/xai_dashboard`
4. Set **Main file path** to `xai_dashboard/app.py`
5. Click **Deploy**

Your app will be live at `https://your-app-name.streamlit.app` in minutes!

**Deployment files included:**
- `.streamlit/config.toml` - Custom theme configuration
- `runtime.txt` - Python version specification
- `.gitignore` - Excludes unnecessary files
- `requirements.txt` - All dependencies

### Other Platforms

For detailed deployment instructions on Railway, Render, Heroku, and more, see **[DEPLOYMENT.md](../DEPLOYMENT.md)** in the root directory.

> **Note**: The `models/` folder is ephemeral on Streamlit Cloud. For persistent storage, integrate with AWS S3 or Google Cloud Storage using `boto3` or `google-cloud-storage`.

---

## 🔮 Future Improvements

- [ ] LIME explanations as an alternative to SHAP
- [ ] Regression task support
- [ ] Hyperparameter tuning via Optuna
- [ ] Dataset auto-profiling with ydata-profiling
- [ ] REST API endpoint via FastAPI
- [ ] Docker containerisation
- [ ] MLflow experiment tracking integration
- [ ] Persistent model storage (S3/GCS integration)
- [ ] User authentication and multi-tenancy
- [ ] Custom model upload support

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

**Ravikiran Bathe**
- GitHub: [@Ravikiranbathe](https://github.com/Ravikiranbathe)
- Repository: [xai_dashboard](https://github.com/Ravikiranbathe/xai_dashboard)

---

## 📄 License

MIT License — free to use, modify, and distribute.
