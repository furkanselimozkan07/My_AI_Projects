"""Reproducible Titanic pipeline (script version of the SELIM notebook).

Differences from the notebook, on purpose:
* imputation statistics are learned on training folds only (no leakage from the test set),
* models are compared with stratified 5-fold cross-validation instead of one split,
* adds FamilySize / IsAlone features and writes feature importances.

Usage:  python train.py            # CV comparison + results/ + TitanicPredict_script.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
RESULTS = HERE / "results"

TITLE_MAP = {
    **dict.fromkeys(["Rev", "Dr", "Col", "Major", "Sir", "Don", "Capt", "Jonkheer"], "Mr"),
    **dict.fromkeys(["Mlle", "Ms"], "Miss"),
    **dict.fromkeys(["Lady", "Mme", "Countess", "Dona"], "Mrs"),
}
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title",
            "FamilySize", "IsAlone"]


def extract_title(names: pd.Series) -> pd.Series:
    """'Braund, Mr. Owen Harris' -> 'Mr', with rare titles merged as in the notebook."""
    title = names.str.extract(r" ([A-Za-z]+)\.", expand=False)
    return title.replace(TITLE_MAP)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (no statistics learned, so safe before splitting)."""
    out = df.copy()
    out["Title"] = extract_title(out["Name"])
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    return out[FEATURES]


class Preprocessor(BaseEstimator, TransformerMixin):
    """Fills Age by title mean, Fare/Embarked by mode, then one-hot encodes.

    All statistics come from `fit`, so inside cross-validation nothing leaks.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "Preprocessor":
        self.age_by_title_ = X.groupby("Title")["Age"].mean().to_dict()
        self.age_default_ = float(X["Age"].mean())
        self.fare_ = float(X["Fare"].median())
        self.embarked_ = X["Embarked"].mode()[0]
        self.columns_ = list(self._encode(X).columns)
        return self

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        fill = X["Title"].map(self.age_by_title_).fillna(self.age_default_)
        X["Age"] = X["Age"].fillna(fill)
        X["Fare"] = X["Fare"].fillna(self.fare_)
        X["Embarked"] = X["Embarked"].astype(object).fillna(self.embarked_)
        X["Sex"] = (X["Sex"] == "female").astype(int)
        X = pd.get_dummies(X, columns=["Embarked", "Title"], dtype=int)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # reindex guarantees identical columns even if a category is absent in a fold
        return self._encode(X).reindex(columns=self.columns_, fill_value=0)


def build_models(seed: int = 42) -> dict[str, BaseEstimator]:
    return {
        "Gaussian NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }


def make_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline([("prep", Preprocessor()), ("model", model)])


def compare_models(X: pd.DataFrame, y: pd.Series, folds: int = 5, seed: int = 42) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in build_models(seed).items():
        scores = cross_val_score(make_pipeline(model), X, y, cv=cv, scoring="accuracy")
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)


def feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.Series:
    """Permutation importance on raw columns (the pipeline handles the encoding)."""
    r = permutation_importance(pipe, X, y, n_repeats=10, random_state=seed, scoring="accuracy")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    train = pd.read_csv(HERE / "ttrain.csv")
    test = pd.read_csv(HERE / "ttest.csv")
    X, y = add_features(train), train["Survived"]

    table = compare_models(X, y, args.folds, args.seed)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = table.loc[0, "model"]
    pipe = make_pipeline(build_models(args.seed)[best]).fit(X, y)
    imp = feature_importance(pipe, X, y, args.seed)
    print("\nPermutation importance (best model):")
    print(imp.round(4).to_string())

    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "cv_results.csv", index=False)
    imp.rename("importance").to_csv(RESULTS / "feature_importance.csv")
    (RESULTS / "summary.json").write_text(json.dumps(
        {"best_model": best, "cv_accuracy": round(float(table.loc[0, "cv_mean"]), 4),
         "folds": args.folds, "seed": args.seed}, indent=2))

    pred = pipe.predict(add_features(test))
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred.astype(int)}) \
        .to_csv(HERE / "TitanicPredict_script.csv", index=False)
    print(f"\nBest: {best}. Wrote results/ and TitanicPredict_script.csv")


if __name__ == "__main__":
    main()
