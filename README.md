# On-the-clinical-acceptance-of-EEG-seizure-prediction-methodologies
Thesis project: "On the clinical acceptance of EEG seizure prediction methodologies". Explainability of seizure prediction models.

This is the code used for the Master thesis "On the Clinical Acceptance of EEG Seizure Prediction Methodologies". It provides one full Machine Learning pipeline for seizure prediction (ensemble of 31 Support Vector Machines (SVMs)), including surrogate analysis. It also provides explainability methods adapted to EEG seizure prediction.

You can not execute these codes as it is necessary the preprocessed data from EEG recordings. As the used dataset belongs to EPILEPSIAE, we can not make it publicly available online due to ethical concerns.

## Code Folders
- Seizure Prediction Pipeline
- Explanations

## Results Folders
- Plots_timeAnalysis folder - classifier time plots
- Plots_featuresAnalysis folder - classifier time plots considering distinct features sets

## Seizure Prediction Pipeline
- [main_training.py] - execute it to train the model and to get the best grid-search parameters (preictal period, k number of features, SVM C value).
- [main testing.py] - execute it to test the model in new seizures, get the performance (seizure sensitivity, FPR/h, and surrogate analysis), and get the selected features.

- [aux_fun.py] - code with utility functions.
- [import_data.py] - code to import data.
- [regularization.py] - code to perform the regularization step using the Firing Power method.
- [training.py] - code to execute the grid search and to train the model.
- [testing.py] - code to test and evaluate the model.
- [save_results.py] - code to save the resulst for each patient in xlsx files.
- [plot_results] - code to get performance plots and selected features. 
- [evaluation] - code to evaluate the tested models (seizure sensitivity, FPR/h, and surrogate analysis).

## Explanations 
- [main.py] - execute it to get the classifier time plots for an ensemble of SVMs
- [plot_explainability.py] - code to develop the classifier time plots and the beeswarm summary SHAP values

