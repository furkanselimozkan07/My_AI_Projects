# Breast cancer diagnosis (Wisconsin diagnostic dataset)

Classify breast tumours as **malignant** or **benign** from 30 numeric features computed from fine-needle-aspirate images (569 samples, 212 malignant). The data ships with scikit-learn (`load_breast_cancer`), so nothing is downloaded or committed.

## What it does

1. Compares five classifiers (Logistic Regression, k-NN, RBF SVM, Random Forest, Gradient Boosting) with 5-fold stratified cross-validation. Scale-sensitive models use a `StandardScaler` inside a `Pipeline`, so there is no leakage.
2. Takes the best model by ROC AUC and evaluates it on a held-out test set.
3. Picks the decision threshold on a separate validation split so recall on malignant cases is at least 98 %. In screening a false negative is much worse than a false alarm.
4. Explains the model with permutation importance.

## Run

```bash
pip install -r requirements.txt
python pipeline.py        # writes results/
pytest                    # 5 tests
```

## Results (seed 42)

| model | accuracy | recall | precision | ROC AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.974 | 0.944 | 0.986 | 0.995 |
| SVM (RBF) | 0.977 | 0.958 | 0.981 | 0.995 |
| Gradient Boosting | 0.951 | 0.911 | 0.957 | 0.993 |
| Random Forest | 0.954 | 0.934 | 0.946 | 0.990 |
| k-NN | 0.965 | 0.920 | 0.986 | 0.986 |

Held-out test (114 patients) with Logistic Regression and the recall-tuned threshold (0.11): ROC AUC 0.995, **recall 97.6 %** (41 of 42 malignant found, 1 missed), at the cost of 7 false alarms.

Plots and raw numbers are in [`results/`](results/): `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `cv_results.csv`, `summary.json`.

## Notes

- Educational project, not a medical device.
- The test set is small, so single-patient differences move recall by ~2 points.
