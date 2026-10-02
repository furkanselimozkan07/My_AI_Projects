# My AI Projects

Machine learning projects I build outside of coursework. Each project lives in its own folder.

## Projects

### [Titanic survival classification](Titanic/)

Team project from a Hack Day event, built as a live Kaggle competition: predict which passengers survived the Titanic from the passenger list.

- **EDA and feature engineering:** filled missing ages using the title extracted from each passenger's name (Mr, Mrs, Miss, ...), plus other cleanup and visualizations with Seaborn and Matplotlib.
- **Models compared:** Gaussian Naive Bayes, Bernoulli Naive Bayes, Random Forest and Gradient Boosting (scikit-learn). Gradient Boosting performed best and was retrained on the full training set.
- **Output:** `TitanicPredict.csv`, the Kaggle submission file.

| File | Content |
| --- | --- |
| `Titanic_Machine_Learning_Classification-SELIM.ipynb` | My notebook |
| `Titanic_Machine_Learning_Classification-Omer-Can.ipynb` | My teammate's notebook |
| `Hack Day Titanic Project.pptx` | Team presentation |
| `ttrain.csv`, `ttest.csv` | Kaggle training and test data |

**Stack:** Python, pandas, scikit-learn, Seaborn, Matplotlib

More projects: see [nvidia-deep-learning](https://github.com/furkanselimozkan07/nvidia-deep-learning) and [ai-bootcamp](https://github.com/furkanselimozkan07/ai-bootcamp).
