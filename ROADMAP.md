# Roadmap

## Vision
A collection of small, reproducible, self-contained ML mini-projects. Each folder has code, a README with real results, tests where sensible, and only public datasets loaded by code.

## Backlog
- [x] Breast cancer diagnosis: model comparison, recall-tuned threshold, explainability
- [ ] Titanic: reproducible `train.py` script from the notebook, with CV and feature importance
- [ ] Wine quality regression vs. classification (UCI, downloaded by code)
- [ ] Hyperparameter search (GridSearch/Optuna) added to breast-cancer-diagnosis
- [ ] Probability calibration curves for breast-cancer-diagnosis
- [ ] MNIST/Fashion-MNIST small CNN with PyTorch (CPU-friendly)
- [ ] House prices regression with feature pipelines
- [ ] Customer churn classification with class-imbalance handling (SMOTE vs. class weights)
- [ ] Time-series forecasting mini-project (Istanbul air-quality or electricity data)
- [ ] Text classification: Turkish news categories with TF-IDF
- [ ] Shared GitHub Actions workflow running tests of all projects
- [ ] Clustering: customer segmentation with k-means/DBSCAN

## Changelog
- 2026-11-10: Added `breast-cancer-diagnosis` (5-model CV comparison, threshold tuning, permutation importance, tests).
