"""
Mars Surface Atlas — Core Feature Extractor & ML Utilities
- Pretrained SqueezeNet v1.1 deep feature extractor (512-dim bottleneck visual representations).
- Perceptual hashing and duplicate/near-duplicate detection to prevent data leakage.
- Multi-model benchmarking (Random Forest, SVM RBF, Linear SVM, Logistic Regression / Transfer Head).
- Model bundling, serialization, and deserialization.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Target labels
LABELS: List[str] = ["rocky", "smooth"]

# Standard ImageNet normalization used by PyTorch pretrained vision models
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class SqueezeNetFeatureExtractor:
    """
    Extracts 512-dimensional visual features from SqueezeNet 1.1's convolutional
    bottleneck followed by Adaptive Average Pooling.
    Does NOT output 1,000-class ImageNet classification logits.
    """

    def __init__(self) -> None:
        weights = models.SqueezeNet1_1_Weights.DEFAULT
        model = models.squeezenet1_1(weights=weights)
        self.features = model.features.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def extract_tensor(self, batch_tensor: torch.Tensor) -> np.ndarray:
        feats = self.features(batch_tensor)
        pooled = self.pool(feats)
        flattened = torch.flatten(pooled, 1)
        return flattened.cpu().numpy().astype(np.float32)

    def extract_image(self, img: Image.Image) -> np.ndarray:
        img_rgb = ImageOps.exif_transpose(img).convert("RGB")
        tensor = _TRANSFORM(img_rgb).unsqueeze(0)
        return self.extract_tensor(tensor)[0]

    def extract_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = [_TRANSFORM(ImageOps.exif_transpose(im).convert("RGB")) for im in chunk]
            batch_t = torch.stack(tensors)
            emb = self.extract_tensor(batch_t)
            embeddings.append(emb)
        return np.vstack(embeddings) if embeddings else np.empty((0, 512), dtype=np.float32)


def compute_perceptual_hash(img: Image.Image, hash_size: int = 8) -> int:
    """
    Computes difference-hash (dHash) for fast near-duplicate image detection.
    """
    img_gray = ImageOps.exif_transpose(img).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(img_gray, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for i, bit in enumerate(diff.flatten()):
        if bit:
            val |= 1 << i
    return val


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def detect_duplicates(
    hashes: List[int],
    filenames: List[str],
    labels: Optional[List[str]] = None,
    threshold: int = 4,
) -> List[Dict[str, Any]]:
    """
    Identifies near-duplicate pairs (Hamming distance <= threshold).
    Cross-class duplicates are flagrant data contamination.
    """
    duplicates = []
    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= threshold:
                dup_info = {
                    "index_a": i,
                    "index_b": j,
                    "file_a": filenames[i],
                    "file_b": filenames[j],
                    "distance": dist,
                }
                if labels is not None:
                    dup_info["label_a"] = labels[i]
                    dup_info["label_b"] = labels[j]
                    dup_info["conflict"] = labels[i] != labels[j]
                duplicates.append(dup_info)
    return duplicates


def get_candidate_models() -> Dict[str, Any]:
    """
    Model alternatives for transfer learning on SqueezeNet deep features:
    1. Random Forest (Robust non-linear ensemble, invariant to monotonic scaling)
    2. Support Vector Machine with RBF kernel (Standard deep embedding non-linear margin with calibrated probabilities)
    3. Support Vector Machine with Linear kernel (Linear margin in feature space with calibrated probabilities)
    4. Logistic Regression / Softmax Transfer Head (Calibrated probabilistic baseline)
    """
    return {
        "Random Forest (300 trees)": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42, class_weight="balanced"), cv=3)),
        ]),
        "SVM (Linear Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(SVC(kernel="linear", C=0.5, random_state=42, class_weight="balanced"), cv=3)),
        ]),
        "Logistic Regression (Transfer Head)": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
    }


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    selected_model_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Performs stratified train/test split.
    Evaluates all candidate models on the holdout validation set.
    Selects the best performing model (or the user-specified model).
    Retrains the winning model architecture on 100% of the training data for the downloadable artifact.
    """
    if len(X) < 6:
        raise ValueError("At least 6 images (at least 3 per class) are required to train and evaluate.")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or min(counts) < 2:
        raise ValueError("Both 'rocky' and 'smooth' classes must have at least 2 samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    candidates = get_candidate_models()
    benchmark_results: Dict[str, Dict[str, Any]] = {}

    best_name = None
    best_f1 = -1.0
    best_metrics = {}

    for name, model_inst in candidates.items():
        # Fit on training fold
        model_inst.fit(X_train, y_train)
        y_pred = model_inst.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        rep = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)

        res = {
            "name": name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "cm": cm.tolist(),
            "report": rep,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        benchmark_results[name] = res

        score = (f1 * 100.0) + acc
        if score > best_f1:
            best_f1 = score
            best_name = name
            best_metrics = res

    if selected_model_name and selected_model_name in benchmark_results:
        best_name = selected_model_name
        best_metrics = benchmark_results[selected_model_name]

    assert best_name is not None

    # Refit winning model architecture on ALL training data for final production bundle
    final_model = get_candidate_models()[best_name]
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "classes": LABELS,
        "n_features": X.shape[1],
        "feature_source": "PyTorch Pretrained SqueezeNet 1.1 (AdaptiveAvgPool 512-D)",
        "n_training_samples": len(y),
        "class_counts": {k: int(v) for k, v in zip(classes, counts)},
    }

    return bundle, best_metrics, benchmark_results


def predict_images(
    bundle: Dict[str, Any],
    images: List[Image.Image],
    filenames: List[str],
    extractor: SqueezeNetFeatureExtractor,
    uncertainty_threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Extracts features for new images and produces predictions with confidence and uncertainty flags.
    """
    model = bundle["model"]
    embeddings = extractor.extract_batch(images)

    preds = model.predict(embeddings)
    probs = model.predict_proba(embeddings)
    classes = list(model.classes_)

    results = []
    for fn, pred, prob in zip(filenames, preds, probs):
        score_dict = {cls_name: float(p) for cls_name, p in zip(classes, prob)}
        rocky_score = score_dict.get("rocky", 0.0)
        smooth_score = score_dict.get("smooth", 0.0)
        confidence = float(max(prob))
        uncertain = confidence < uncertainty_threshold

        results.append({
            "Filename": fn,
            "Prediction": str(pred).upper(),
            "Confidence": confidence,
            "Rocky Score": rocky_score,
            "Smooth Score": smooth_score,
            "Uncertainty Warning": "⚠️ Low Confidence (< 65%)" if uncertain else "Confident",
            "Is Uncertain": uncertain,
        })
    return results
