import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402

HERE = Path(train.__file__).parent


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(HERE / "ttrain.csv")
    return train.add_features(df), df["Survived"]


def test_extract_title_merges_rare():
    s = pd.Series(["A, Mr. B", "C, Mlle. D", "E, Dr. F", "G, Mme. H", "I, Mrs. J"])
    assert train.extract_title(s).tolist() == ["Mr", "Miss", "Mr", "Mrs", "Mrs"]


def test_add_features_family(data):
    X, _ = data
    assert (X["FamilySize"] >= 1).all()
    assert ((X["FamilySize"] == 1) == (X["IsAlone"] == 1)).all()


def test_preprocessor_no_nans_and_stable_columns(data):
    X, _ = data
    p = train.Preprocessor().fit(X.iloc[:100])
    out = p.transform(X.iloc[100:])
    assert not out.isna().any().any()
    assert list(out.columns) == p.columns_


def test_preprocessor_unseen_title_uses_default():
    X = pd.DataFrame({"Pclass": [1, 3], "Sex": ["male", "female"], "Age": [30.0, 20.0],
                      "SibSp": [0, 0], "Parch": [0, 0], "Fare": [10.0, 5.0],
                      "Embarked": ["S", "C"], "Title": ["Mr", "Miss"],
                      "FamilySize": [1, 1], "IsAlone": [1, 1]})
    p = train.Preprocessor().fit(X)
    new = X.iloc[:1].assign(Title="Rare", Age=np.nan, Fare=np.nan, Embarked=np.nan)
    out = p.transform(new)
    assert out["Age"].iloc[0] == pytest.approx(25.0)
    assert out.shape[1] == len(p.columns_)


def test_compare_models_beats_baseline(data):
    X, y = data
    table = train.compare_models(X, y, folds=3)
    assert len(table) == 5
    assert table["cv_mean"].max() > 0.78  # majority-class baseline is ~0.62


def test_end_to_end_predict_shape(data):
    X, y = data
    pipe = train.make_pipeline(train.build_models()["Logistic Regression"]).fit(X, y)
    test = train.add_features(pd.read_csv(HERE / "ttest.csv"))
    pred = pipe.predict(test)
    assert len(pred) == 418 and set(pred) <= {0, 1}
