import io
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from ml_engine import (
    LABELS,
    SqueezeNetFeatureExtractor,
    compute_perceptual_hash,
    detect_duplicates,
    get_candidate_models,
    predict_images,
    train_and_evaluate_models,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mars Surface Atlas — Planetary Terrain Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Premium Light-Only Editorial Typography & Warm Ivory Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        color-scheme: light !important;
        --bg-paper: #F7F5F0;
        --card-white: #FFFFFF;
        --card-warm: #EFECE4;
        --border-subtle: #DFDACF;
        --border-strong: #C8C1B4;
        --ink-primary: #1C1C18;
        --ink-secondary: #57544C;
        --ink-tertiary: #7D796F;
        --terracotta: #A94F37;
        --terracotta-hover: #8E3E28;
        --terracotta-light: #F9EDE9;
        --status-safe: #2C5E3B;
        --status-warning: #8F5200;
    }

    /* Core Application Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--ink-primary) !important;
        background-color: var(--bg-paper) !important;
    }

    [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background-color: var(--bg-paper) !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--ink-primary) !important;
        letter-spacing: -0.025em !important;
        font-weight: 700 !important;
    }

    p, span, label, div, li {
        color: var(--ink-primary) !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] *, small {
        color: var(--ink-secondary) !important;
        font-size: 0.88rem !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--card-warm) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--ink-primary) !important;
    }

    /* Header Bar */
    [data-testid="stHeader"] {
        background-color: var(--bg-paper) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    /* Cards and Metric Tiles */
    div[data-testid="stMetric"] {
        background: var(--card-white) !important;
        border: 1px solid var(--border-subtle) !important;
        padding: 18px 22px !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--ink-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        background-color: transparent !important;
        border-bottom: 2px solid var(--border-subtle) !important;
        padding-bottom: 0px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0 !important;
        padding: 12px 22px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: var(--ink-secondary) !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--terracotta) !important;
        border-bottom: 3px solid var(--terracotta) !important;
        background-color: var(--card-white) !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: var(--card-white) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: var(--card-warm) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: var(--card-white) !important;
        color: var(--ink-primary) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        color: var(--ink-primary) !important;
        background-color: var(--card-white) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.4rem !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        letter-spacing: -0.01em !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--ink-primary) !important;
        background-color: #FDFCFA !important;
    }
    .stButton > button[kind="primary"] {
        color: #FFFFFF !important;
        background-color: var(--terracotta) !important;
        border-color: var(--terracotta) !important;
    }
    .stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--terracotta-hover) !important;
        border-color: var(--terracotta-hover) !important;
    }

    /* Form Controls */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input, textarea {
        background-color: var(--card-white) !important;
        color: var(--ink-primary) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="popover"], [role="listbox"], [role="option"] {
        background-color: var(--card-white) !important;
        color: var(--ink-primary) !important;
    }
    [role="option"]:hover, [aria-selected="true"] {
        background-color: var(--card-warm) !important;
    }

    /* Dataframe & Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: var(--card-white) !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background-color: var(--card-warm) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 8px !important;
    }

    /* Executive Hero Card */
    .atlas-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #EFECE4 100%);
        border: 1px solid var(--border-subtle);
        border-left: 5px solid var(--terracotta);
        padding: 24px 28px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .atlas-hero h1 {
        margin: 0 0 6px 0;
        font-size: 1.85rem;
    }
    .atlas-badge {
        display: inline-block;
        background-color: var(--terracotta-light);
        color: var(--terracotta) !important;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 3px 9px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }

    .prediction-card {
        background: var(--card-white);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    .status-badge {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        margin-top: 6px;
    }
    .status-safe {
        background-color: #EBF3ED;
        color: var(--status-safe);
    }
    .status-warn {
        background-color: #F8F1E5;
        color: var(--status-warning);
    }

    hr {
        border-color: var(--border-subtle) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Feature Extractor Singleton
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading SqueezeNet feature extractor...")
def load_feature_extractor() -> SqueezeNetFeatureExtractor:
    return SqueezeNetFeatureExtractor()


feature_extractor = load_feature_extractor()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 16px 0;">
            <span class="atlas-badge">System Core</span>
            <h2 style="margin: 4px 0 0 0; font-size: 1.35rem;">MARS ATLAS</h2>
            <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #57544C;">Surface Hazard and Terrain Classifier</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Architecture Specification**")
    st.markdown(
        """
        - **Vision Backbone**: Pretrained SqueezeNet v1.1
        - **Feature Layer**: Conv Fire Bottleneck (512-D)
        - **Pooling Method**: Adaptive Average Pooling
        - **Leakage Guard**: Perceptual dHash Audit
        - **Candidate Heads**: Random Forest, RBF/Linear SVM, Transfer Head
        """
    )
    st.divider()

    st.markdown("**Runtime Status**")
    active_bundle = st.session_state.get("active_model_bundle")
    if active_bundle:
        st.success(
            f"Active Model: **{active_bundle.get('model_name', 'Trained')}**\n\n"
            f"Training samples: {active_bundle.get('n_training_samples', 0)}"
        )
    else:
        st.info("No active model loaded. Train or upload a model bundle.")

    st.divider()
    st.caption("Deployment target: Streamlit Community Cloud (Python 3.12+ compatible).")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="atlas-hero">
        <span class="atlas-badge">Planetary Exploration Workspace</span>
        <h1>Mars Surface Atlas</h1>
        <p style="margin: 0; color: #57544C; font-size: 0.95rem; line-height: 1.5;">
            Autonomous classification of planetary terrain into <strong>Rocky</strong> (hazard/mobility restriction) 
            and <strong>Smooth</strong> (navigable regolith/sand). Trained via deep SqueezeNet visual feature embeddings.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "01 · Train Model",
    "02 · Classify Images",
    "03 · Model Benchmark and Evaluation",
    "04 · System Method and Integrity",
])


# =========================================================
# TAB 1: TRAIN MODEL
# =========================================================
with tabs[0]:
    st.subheader("1. Train from Raw Mars Images")
    st.markdown(
        "Upload labelled sets of **Rocky** and **Smooth** JPG or PNG images. "
        "The system extracts 512-dimensional bottleneck features, audits for duplicate contamination, "
        "and optimizes the classifier across candidate architectures."
    )

    col1, col2 = col_u = st.columns(2)
    with col1:
        st.markdown("#### Rocky Terrain Samples")
        rocky_files = st.file_uploader(
            "Upload Rocky images (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="rocky_train_upload",
            help="Images of boulder fields, jagged rocks, rocky outcrops.",
        )
        if rocky_files:
            st.caption(f"{len(rocky_files)} rocky images staged.")

    with col2:
        st.markdown("#### Smooth Terrain Samples")
        smooth_files = st.file_uploader(
            "Upload Smooth images (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="smooth_train_upload",
            help="Images of smooth plains, wind-blown dunes, low-friction sand.",
        )
        if smooth_files:
            st.caption(f"{len(smooth_files)} smooth images staged.")

    st.markdown("---")
    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
    with col_cfg1:
        holdout_ratio = st.selectbox(
            "Holdout Validation Split",
            options=[0.20, 0.25, 0.30, 0.35],
            index=1,
            format_func=lambda x: f"{int(x * 100)}% Validation Holdout",
        )
    with col_cfg2:
        candidate_options = ["Auto-select Best Model"] + list(get_candidate_models().keys())
        selected_arch = st.selectbox(
            "Model Selection Strategy",
            options=candidate_options,
            index=0,
            help="Automatically select based on holdout F1-score or pin a specific ML architecture.",
        )
    with col_cfg3:
        filter_duplicates = st.checkbox(
            "Enforce Data Leakage Guard",
            value=True,
            help="Filters near-duplicate frames and flags cross-class label contradictions via perceptual dHash.",
        )

    if st.button("Extract SqueezeNet Features and Train Classifier", type="primary", use_container_width=True):
        if not rocky_files or not smooth_files:
            st.error("Please upload at least 3 images for Rocky and at least 3 images for Smooth terrain.")
        elif len(rocky_files) < 3 or len(smooth_files) < 3:
            st.error(f"Need at least 3 samples per class. Got {len(rocky_files)} rocky and {len(smooth_files)} smooth.")
        else:
            with st.spinner("Decoding images, running perceptual hash verification, and extracting SqueezeNet embeddings..."):
                raw_images: List[Image.Image] = []
                labels: List[str] = []
                filenames: List[str] = []
                hashes: List[int] = []

                # Process Rocky
                for f in rocky_files:
                    try:
                        im = Image.open(f)
                        raw_images.append(im)
                        labels.append("rocky")
                        filenames.append(f.name)
                        hashes.append(compute_perceptual_hash(im))
                    except Exception as err:
                        st.warning(f"Could not read {f.name}: {err}")

                # Process Smooth
                for f in smooth_files:
                    try:
                        im = Image.open(f)
                        raw_images.append(im)
                        labels.append("smooth")
                        filenames.append(f.name)
                        hashes.append(compute_perceptual_hash(im))
                    except Exception as err:
                        st.warning(f"Could not read {f.name}: {err}")

                # Duplicate / Contamination audit
                dup_pairs = detect_duplicates(hashes, filenames, labels, threshold=3)
                excluded_indices = set()
                if filter_duplicates and dup_pairs:
                    for d in dup_pairs:
                        excluded_indices.add(d["index_b"])

                final_images = [im for idx, im in enumerate(raw_images) if idx not in excluded_indices]
                final_labels = [lb for idx, lb in enumerate(labels) if idx not in excluded_indices]
                final_filenames = [fn for idx, fn in enumerate(filenames) if idx not in excluded_indices]

                # Extract SqueezeNet bottleneck features
                X_feats = feature_extractor.extract_batch(final_images)
                y_arr = np.array(final_labels)

                # Train and benchmark
                chosen_model = None if selected_arch == "Auto-select Best Model" else selected_arch
                try:
                    bundle, best_metrics, bench_results = train_and_evaluate_models(
                        X=X_feats,
                        y=y_arr,
                        test_size=holdout_ratio,
                        selected_model_name=chosen_model,
                    )

                    st.session_state["active_model_bundle"] = bundle
                    st.session_state["benchmark_results"] = bench_results
                    st.session_state["best_metrics"] = best_metrics
                    st.session_state["audit_duplicates"] = dup_pairs
                    st.session_state["excluded_count"] = len(excluded_indices)

                    st.success(
                        f"Training complete. Selected {bundle['model_name']} "
                        f"trained on {len(final_labels)} unique images "
                        f"({final_labels.count('rocky')} rocky, {final_labels.count('smooth')} smooth)."
                    )
                except Exception as ex:
                    st.error(f"Training failed: {ex}")

    # Display training summary if available
    if "best_metrics" in st.session_state:
        bm = st.session_state["best_metrics"]
        bundle = st.session_state["active_model_bundle"]

        st.markdown("### Model Performance Overview")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Selected Model", bundle.get("model_name", "Best").split("(")[0].strip())
        m2.metric("Test Accuracy", f"{bm['accuracy']:.1%}")
        m3.metric("Precision (Macro)", f"{bm['precision']:.1%}")
        m4.metric("Recall (Macro)", f"{bm['recall']:.1%}")
        m5.metric("F1-Score (Macro)", f"{bm['f1']:.1%}")

        if st.session_state.get("audit_duplicates"):
            dups = st.session_state["audit_duplicates"]
            st.markdown("#### Data Leakage and Duplicate Audit")
            conflict_count = sum(1 for d in dups if d.get("conflict"))
            if conflict_count > 0:
                st.error(
                    f"Warning: Found {conflict_count} cross-class duplicate images. "
                    "Near-identical images have conflicting labels (Rocky vs Smooth)."
                )
            dup_df = pd.DataFrame([
                {
                    "Image A": d["file_a"],
                    "Label A": d.get("label_a", "N/A"),
                    "Image B": d["file_b"],
                    "Label B": d.get("label_b", "N/A"),
                    "dHash Distance": d["distance"],
                    "Conflicting Label": "Yes (Conflict)" if d.get("conflict") else "No (Redundant Frame)",
                }
                for d in dups
            ])
            st.dataframe(dup_df, use_container_width=True)
            if st.session_state.get("excluded_count", 0) > 0:
                st.caption(f"Filtered out {st.session_state['excluded_count']} redundant duplicate images from training.")

        # Download Model Bundle
        st.markdown("---")
        st.markdown("#### Export Model Bundle")
        st.caption(
            "Download this self-contained `.joblib` bundle. "
            "You can restore it in any session or Streamlit instance without retraining."
        )
        model_bytes = io.BytesIO()
        joblib.dump(bundle, model_bytes)
        st.download_button(
            label="Download Trained Model (.joblib)",
            data=model_bytes.getvalue(),
            file_name=f"mars_surface_classifier_{bundle.get('model_name', 'model').lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )


# =========================================================
# TAB 2: CLASSIFY IMAGES
# =========================================================
with tabs[1]:
    st.subheader("2. Classify Unseen Mars Images")
    st.markdown(
        "Upload raw Mars imagery (JPG/PNG). The model extracts deep visual features with SqueezeNet "
        "and classifies each terrain as **Rocky** or **Smooth** with calibrated confidence scores."
    )

    with st.expander("Load an existing .joblib model bundle (Optional)", expanded=False):
        uploaded_model_file = st.file_uploader(
            "Upload trained model bundle (.joblib)",
            type=["joblib", "pkl"],
            key="model_bundle_upload",
        )
        if uploaded_model_file:
            try:
                loaded = joblib.load(io.BytesIO(uploaded_model_file.getvalue()))
                if isinstance(loaded, dict) and "model" in loaded:
                    st.session_state["active_model_bundle"] = loaded
                    st.success(f"Loaded bundle: {loaded.get('model_name', 'Custom Model')} ({loaded.get('feature_source', 'SqueezeNet')})")
                else:
                    st.error("Invalid model bundle format. Expected dictionary containing trained 'model'.")
            except Exception as e:
                st.error(f"Failed to load model bundle: {e}")

    active_bundle = st.session_state.get("active_model_bundle")
    if not active_bundle:
        st.warning("No trained model is currently active. Please train a model in Tab 01 or load a `.joblib` bundle above.")
    else:
        st.info(
            f"Active Model: {active_bundle.get('model_name', 'Trained Classifier')} | "
            f"Trained on {active_bundle.get('n_training_samples', 'N/A')} samples."
        )

        pred_images_files = st.file_uploader(
            "Upload new Mars images to classify (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="predict_images_upload",
        )

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            uncertainty_threshold = st.slider(
                "Uncertainty Alert Threshold",
                min_value=0.50,
                max_value=0.90,
                value=0.65,
                step=0.05,
                format="%.0f%%",
                help="Classifications with confidence below this threshold will be flagged as uncertain.",
            )

        if pred_images_files and st.button("Classify Mars Images", type="primary", use_container_width=True):
            with st.spinner("Extracting SqueezeNet features and predicting terrain..."):
                loaded_imgs = []
                loaded_names = []
                for pf in pred_images_files:
                    try:
                        im = Image.open(pf)
                        loaded_imgs.append(im)
                        loaded_names.append(pf.name)
                    except Exception as err:
                        st.warning(f"Skipping {pf.name}: {err}")

                if loaded_imgs:
                    results = predict_images(
                        bundle=active_bundle,
                        images=loaded_imgs,
                        filenames=loaded_names,
                        extractor=feature_extractor,
                        uncertainty_threshold=uncertainty_threshold,
                    )
                    st.session_state["prediction_results"] = results
                    st.session_state["prediction_imgs"] = loaded_imgs
                    st.session_state["prediction_filenames"] = loaded_names
                    st.success(f"Classified {len(results)} images successfully.")

        if "prediction_results" in st.session_state:
            results = st.session_state["prediction_results"]
            res_df = pd.DataFrame(results)

            st.markdown("### Classification Summary")
            c_rocky = sum(1 for r in results if r["Prediction"] == "ROCKY")
            c_smooth = sum(1 for r in results if r["Prediction"] == "SMOOTH")
            c_uncertain = sum(1 for r in results if r["Is Uncertain"])

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Classified", len(results))
            p2.metric("Predicted Rocky (Hazard)", c_rocky)
            p3.metric("Predicted Smooth (Safe)", c_smooth)
            p4.metric("Uncertain / Low Confidence", c_uncertain)

            # Tabular results
            st.markdown("#### Detailed Results Table")
            display_df = res_df.copy()
            display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.1%}")
            display_df["Rocky Score"] = display_df["Rocky Score"].map(lambda x: f"{x:.1%}")
            display_df["Smooth Score"] = display_df["Smooth Score"].map(lambda x: f"{x:.1%}")
            display_cols = ["Filename", "Prediction", "Confidence", "Rocky Score", "Smooth Score", "Uncertainty Warning"]
            st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

            # CSV Download
            csv_data = res_df.drop(columns=["Is Uncertain"], errors="ignore").to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_data,
                file_name="mars_surface_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Gallery with thumbnails
            st.markdown("#### Individual Image Predictions and Scores")
            imgs = st.session_state.get("prediction_imgs", [])
            cols_per_row = 4
            for idx in range(0, len(results), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for c_idx, col in enumerate(row_cols):
                    item_idx = idx + c_idx
                    if item_idx < len(results):
                        item = results[item_idx]
                        im = imgs[item_idx]
                        with col:
                            st.image(im, use_container_width=True)
                            pred_badge = item["Prediction"]
                            status_html = (
                                f"<span class='status-badge status-warn'>{item['Uncertainty Warning']}</span>"
                                if item["Is Uncertain"]
                                else "<span class='status-badge status-safe'>High Confidence</span>"
                            )
                            st.markdown(
                                f"""
                                <div class="prediction-card">
                                    <div style="font-size: 0.82rem; font-weight: 700; word-break: break-all;">{item['Filename']}</div>
                                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0; color: #A94F37;">{pred_badge}</div>
                                    <div style="font-size: 0.85rem;">Confidence: <strong>{item['Confidence']:.1%}</strong></div>
                                    <div style="font-size: 0.78rem; color: #57544C;">Rocky: {item['Rocky Score']:.1%} | Smooth: {item['Smooth Score']:.1%}</div>
                                    {status_html}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


# =========================================================
# TAB 3: BENCHMARK & EVALUATION
# =========================================================
with tabs[2]:
    st.subheader("3. Model Benchmark and Validation Performance")
    st.markdown(
        "Compare candidate transfer learning heads on the holdout validation set. "
        "Every model is strictly evaluated on unseen validation samples."
    )

    bench = st.session_state.get("benchmark_results")
    if not bench:
        st.info("Train a model in Tab 01 to generate multi-model comparative validation benchmarks.")
    else:
        # Comparison Table
        bench_rows = []
        for name, data in bench.items():
            bench_rows.append({
                "Model Architecture": name,
                "Accuracy": data["accuracy"],
                "Precision (Macro)": data["precision"],
                "Recall (Macro)": data["recall"],
                "F1-Score (Macro)": data["f1"],
                "Train Samples": data["n_train"],
                "Holdout Samples": data["n_test"],
            })
        bench_df = pd.DataFrame(bench_rows)
        styled_df = bench_df.copy()
        for col in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]:
            styled_df[col] = styled_df[col].map(lambda x: f"{x:.1%}")

        st.markdown("#### Candidate Model Benchmark")
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Selected Model Details
        bm = st.session_state.get("best_metrics", {})
        if bm:
            st.markdown(f"#### Detailed Diagnostics for {bm.get('name', 'Active Model')}")
            col_cm, col_cr = st.columns([1, 1.2])

            with col_cm:
                st.markdown("**Confusion Matrix (Holdout Set)**")
                cm_arr = np.array(bm["cm"])
                cm_df = pd.DataFrame(
                    cm_arr,
                    index=["Actual Rocky", "Actual Smooth"],
                    columns=["Predicted Rocky", "Predicted Smooth"],
                )
                st.dataframe(cm_df, use_container_width=True)

                tn = cm_arr[0, 0]
                fp = cm_arr[0, 1]
                fn = cm_arr[1, 0]
                tp = cm_arr[1, 1]
                st.caption(
                    f"Rocky accuracy: {tn}/{tn+fp} ({tn/(tn+fp):.1%}) | "
                    f"Smooth accuracy: {tp}/{tp+fn} ({tp/(tp+fn):.1%})"
                )

            with col_cr:
                st.markdown("**Per-Class Classification Report**")
                rep_dict = bm.get("report", {})
                if rep_dict:
                    report_rows = []
                    for cls in LABELS:
                        if cls in rep_dict:
                            report_rows.append({
                                "Class": cls.upper(),
                                "Precision": f"{rep_dict[cls]['precision']:.1%}",
                                "Recall": f"{rep_dict[cls]['recall']:.1%}",
                                "F1-Score": f"{rep_dict[cls]['f1-score']:.1%}",
                                "Support": int(rep_dict[cls]['support']),
                            })
                    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


# =========================================================
# TAB 4: METHOD & INTEGRITY
# =========================================================
with tabs[3]:
    st.subheader("4. Method, Data Integrity and Deployment Architecture")
    st.markdown(
        """
        ### Transfer Learning Pipeline
        1. **Pretrained Backbone**: Utilizes torchvision's official SqueezeNet 1.1 pretrained weights.
        2. **Visual Feature Extraction**: Rather than classifying into ImageNet's 1,000 biological and object categories, 
           the system taps the bottleneck feature map (`features` block) and applies **Adaptive Average Pooling** down to a 512-dimensional continuous visual vector.
        3. **Data Contamination Safeguards**: Planetary imagery taken from rovers or orbiters often exhibits frame bursts (near-duplicate consecutive exposures). 
           Difference hashing (dHash) audits every image to detect near-duplicates and block cross-class contamination before training.
        4. **Multi-Model Benchmark**: Evaluates four distinct learning heads:
           - **Random Forest (300 trees)**: Non-parametric ensemble resistant to outliers.
           - **SVM (RBF Kernel)**: Non-linear boundary with probability calibration.
           - **SVM (Linear Kernel)**: High-dimensional linear margin.
           - **Logistic Regression (Transfer Head)**: Standard soft-margin convex classifier.
        5. **Deployment Stateless Persistence**: Streamlit Community Cloud ephemeral worker containers do not guarantee persistent local filesystem storage across reboots. 
           This app encapsulates models in standalone `.joblib` bundles that can be downloaded and reloaded at will.
        """
    )
    st.info(
        "Integrity Note: Accuracy figures presented reflect the holdout validation split. "
        "Performance on novel planetary surfaces under varying sun angles, dust deposition, and rover lens distortion may vary."
    )
