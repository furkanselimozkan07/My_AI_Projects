import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p  # noqa: E402


def test_load_data_shape_and_labels():
    X, y = p.load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 212  # 212 malignant cases in the dataset


def test_choose_threshold_meets_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.6, 0.3, 0.5, 0.2, 0.1, 0.05])
    assert p.choose_threshold(y, proba, min_recall=0.75) == 0.6
    assert p.choose_threshold(y, proba, min_recall=1.0) == 0.3


def test_cross_validation_table():
    X, y = p.load_data()
    models = {k: v for k, v in p.build_models().items() if k in ("Logistic Regression", "k-NN")}
    t = p.cross_validate_models(models, X, y, folds=3)
    assert list(t.columns) == ["model", "accuracy", "recall", "precision", "roc_auc"]
    assert (t["roc_auc"] > 0.95).all()
    assert t["roc_auc"].is_monotonic_decreasing


def test_holdout_hits_recall_target():
    X, y = p.load_data()
    res = p.evaluate_holdout(p.build_models()["Logistic Regression"], X, y)
    assert res["roc_auc"] > 0.97
    assert res["recall"] >= 0.9
    assert np.array(res["confusion_matrix"]).sum() == 114


def test_run_writes_outputs(tmp_path):
    summary = p.run(out=tmp_path)
    for f in ("cv_results.csv", "summary.json", "confusion_matrix.png", "roc_curve.png", "feature_importance.png"):
        assert (tmp_path / f).exists()
    assert summary["roc_auc"] > 0.95
