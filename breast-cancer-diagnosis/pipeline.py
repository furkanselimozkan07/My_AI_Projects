"""Breast cancer diagnosis: compare classifiers on the Wisconsin diagnostic dataset.

The dataset ships with scikit-learn, so nothing has to be downloaded.
Run `python pipeline.py` to train, evaluate and write plots/metrics to `results/`.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
    confusion_matrix, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# SVC(probability=True) is deprecated in newest scikit-learn but still works.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

RESULTS = Path(__file__).parent / "results"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target; target 1 = malignant, 0 = benign."""
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    y = (raw.target == 0).astype(int)  # sklearn encodes malignant as 0; flip it
    y.name = "malignant"
    return X, y


def build_models(seed: int = 42) -> dict[str, BaseEstimator]:
    """Models to compare. Scale-sensitive ones get a StandardScaler in a Pipeline."""
    def scaled(clf: BaseEstimator) -> Pipeline:
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=1000, random_state=seed)),
        "k-NN": scaled(KNeighborsClassifier(n_neighbors=7)),
        "SVM (RBF)": scaled(SVC(probability=True, random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }


def cross_validate_models(models: dict[str, BaseEstimator], X, y,
                          folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Stratified k-fold CV; returns mean metrics per model sorted by ROC AUC."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X, y, cv=cv,
                           scoring=["accuracy", "recall", "precision", "roc_auc"])
        rows.append({"model": name,
                     **{m: float(np.mean(s[f"test_{m}"])) for m in
                        ("accuracy", "recall", "precision", "roc_auc")}})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def choose_threshold(y_true, proba, min_recall: float = 0.98) -> float:
    """Highest probability threshold that still reaches `min_recall` on malignant cases.

    In screening, missing a malignant tumour (false negative) is far worse than a
    false alarm, so we trade precision for recall.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    for t in np.unique(proba)[::-1]:
        if recall_score(y_true, (proba >= t).astype(int)) >= min_recall:
            return float(t)
    return float(proba.min())


def evaluate_holdout(model: BaseEstimator, X, y, seed: int = 42, min_recall: float = 0.98):
    """Fit on 80%, tune the threshold on a validation split, report on the 20% test set."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)
    model.fit(X_tr, y_tr)
    thr = choose_threshold(y_val, model.predict_proba(X_val)[:, 1], min_recall)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    return {
        "threshold": thr,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, target_names=["benign", "malignant"], output_dict=True),
        "y_test": y_test, "proba": proba, "pred": pred, "X_test": X_test,
    }


def top_features(model: BaseEstimator, X_test, y_test, n: int = 10, seed: int = 42) -> pd.Series:
    """Permutation importance (ROC AUC drop) -- model-agnostic."""
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=seed)
    return pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False).head(n)


def run(out: Path = RESULTS, seed: int = 42) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    models = build_models(seed)
    cv_table = cross_validate_models(models, X, y, seed=seed)
    cv_table.round(4).to_csv(out / "cv_results.csv", index=False)

    best_name = cv_table.loc[0, "model"]
    res = evaluate_holdout(models[best_name], X, y, seed)
    imp = top_features(models[best_name], res["X_test"], res["y_test"], seed=seed)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(res["y_test"], res["pred"],
                                            display_labels=["benign", "malignant"], ax=ax)
    ax.set_title(f"{best_name} (threshold {res['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(res["y_test"], res["proba"], name=best_name, ax=ax)
    fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    imp[::-1].plot.barh(ax=ax); ax.set_xlabel("Permutation importance (ROC AUC drop)")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=120); plt.close(fig)

    summary = {"best_model": best_name,
               **{k: res[k] for k in ("threshold", "roc_auc", "recall", "confusion_matrix")},
               "top_features": imp.round(4).to_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(cv_table.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    run(seed=ap.parse_args().seed)
