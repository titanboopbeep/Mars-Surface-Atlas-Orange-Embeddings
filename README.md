# Mars Surface Atlas — Planetary Terrain Classifier

An image-based machine learning application for classifying Martian terrain into **Rocky** (navigational hazards and boulder fields) and **Smooth** (navigable regolith and sand plains), directly deployable to **Streamlit Community Cloud**.

---

## Key Features

1. **Direct Image Upload and Feature Extraction**
   - Upload raw Mars images (`.jpg`, `.jpeg`, `.png`) directly in the browser.
   - Extracts continuous 512-dimensional bottleneck visual representations using a pretrained **SqueezeNet v1.1** backbone (`features` block + Adaptive Average Pooling).
   - Does **not** use arbitrary ImageNet classification logits.

2. **Multi-Model Benchmark and Model Selection**
   - Compares **Random Forest (300 trees)**, **SVM with RBF Kernel**, **Linear SVM**, and **Logistic Regression Transfer Head**.
   - Evaluates performance using held-out validation accuracy, precision, recall, macro F1-score, and confusion matrices.
   - Automatically selects the best model or allows manual selection.

3. **Data Leakage and Contamination Guard**
   - Planetary rover imagery often contains burst frames (near-duplicates).
   - Perceptual difference hashing (`dHash`) automatically audits uploads to flag near-duplicate frames and alert if conflicting labels exist.

4. **Prediction with Uncertainty Warnings**
   - Upload new, unseen Mars surface imagery for batch classification.
   - Calibrated probability scores for Rocky and Smooth terrain.
   - Dynamic uncertainty threshold flags predictions where model confidence is below specified limits.
   - Export results as a downloadable CSV.

5. **Stateless Model Persistence**
   - Streamlit Community Cloud ephemeral worker containers do not maintain persistent local disk storage across reboots.
   - Models are serialized into self-contained `.joblib` bundles that can be downloaded and reloaded into any session instantly.

6. **Refined Light-Only UI**
   - Professional executive styling with warm ivory palette (`#F7F5F0`), white cards, dark charcoal text (`#1C1C18`), and muted terracotta accents (`#A94F37`).
   - Clean contrast, responsive typography, and zero emoji clutter.

---

## Streamlit Community Cloud Deployment

Deploying this repository to Streamlit Community Cloud takes less than 2 minutes:

1. **Push this repository to GitHub** under your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `titanboopbeep/Mars-Surface-Atlas-Orange-Embeddings` (or your fork/renamed repo).
5. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: `3.12` (or `3.11`)
6. Click **"Deploy!"**.

---

## Local Development Setup

To run locally on your computer:

```bash
# 1. Clone repository
git clone https://github.com/titanboopbeep/Mars-Surface-Atlas-Orange-Embeddings.git
cd Mars-Surface-Atlas-Orange-Embeddings

# 2. Create virtual environment (Python 3.11 or 3.12 recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit
streamlit run app.py
```

---

## Repository Structure

```
├── .streamlit/
│   └── config.toml          # Streamlit theme (warm ivory light palette) & server config
├── .gitignore               # Ignores bytecode, caches, models, and virtual environments
├── app.py                   # Streamlit web interface and session state management
├── ml_engine.py             # SqueezeNet feature extractor, perceptual hashing, and ML pipelines
├── requirements.txt         # Minimal, pinned dependencies (torch, torchvision, scikit-learn, etc.)
└── README.md                # Documentation and deployment guide
```

---

## Scientific and Engineering Methodology

- **Backbone**: SqueezeNet v1.1 (Iandola et al.) pretrained on ImageNet. Chosen for its compact parameter footprint (4.8MB), fast CPU inference in cloud environments, and strong spatial feature extraction in early convolutional fire modules.
- **Pooling**: Adaptive Average Pooling converts spatial feature activations `(512, 13, 13)` into a single 512-dimensional continuous embedding vector per image.
- **Honest Evaluation**: No manufactured 100% metrics. Accuracy and F1 scores are calculated strictly on unseen holdout splits.
