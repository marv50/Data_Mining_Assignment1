"""
classification.py

This script provides utilities for training and evaluating classification models
(Logistic Regression and K-Nearest Neighbors) on processed datasets. It supports:
- Cross-validated experiments with repeated random splits
- Coefficient analysis using L1-penalized (Lasso) logistic regression

Dependencies:
- pandas
- numpy
- matplotlib
- scikit-learn

Author: [Your Name]
Date: [Today's Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.basic.interpolation import classify_columns

# --- Utility Functions ---

def create_pipeline(categorical_cols, numeric_cols, model_type="logreg", lasso=False):
    """
    Creates a preprocessing and modeling pipeline.

    Args:
        categorical_cols (list): List of categorical feature names.
        numeric_cols (list): List of numeric feature names.
        model_type (str): 'logreg' for logistic regression, 'knn' for KNN classifier.
        lasso (bool): Whether to use L1 penalty (only applicable for logistic regression).

    Returns:
        sklearn.pipeline.Pipeline: A configured pipeline ready for training.
    """
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    if model_type == "logreg":
        penalty = 'l1' if lasso else 'l2'
        model = LogisticRegression(
            multi_class='auto', max_iter=1000, solver='saga',
            penalty=penalty, warm_start=True
        )
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipeline

def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """
    Trains a model pipeline and evaluates its accuracy.

    Args:
        pipeline (Pipeline): Preprocessing and modeling pipeline.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.

    Returns:
        tuple: (accuracy score, model coefficients if available)
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    model = pipeline.named_steps['model']
    coef = model.coef_ if hasattr(model, "coef_") else None

    return acc, coef

def run_experiment(dataset_name, selected_columns, lasso=False, n_runs=50, model_type="logreg"):
    """
    Runs repeated training/testing experiments to evaluate model performance.

    Args:
        dataset_name (str): Dataset file prefix (excluding path and extension).
        selected_columns (list): Features to use.
        lasso (bool): Whether to use L1 penalty (logistic regression only).
        n_runs (int): Number of random train/test splits.
        model_type (str): 'logreg' or 'knn'.

    Returns:
        tuple: (mean accuracy, standard deviation of accuracy)
    """
    filepath = f"data/basic/basic_{dataset_name}.csv"
    df = pd.read_csv(filepath)

    y = df['ml_course'].astype('category')
    if len(y.cat.categories) != 2:
        raise ValueError("Target variable must have exactly 2 classes for binomial logistic regression.")

    X = df[selected_columns]
    categorical_cols, numeric_cols = classify_columns(X)

    accuracies = []

    for random_state in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )

        pipeline = create_pipeline(categorical_cols, numeric_cols, model_type=model_type, lasso=lasso)
        acc, _ = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    return np.mean(accuracies), np.std(accuracies)

def run_single_experiment(dataset_name, selected_columns):
    """
    Runs a single train/test split experiment using L1 logistic regression,
    and summarizes feature importance.

    Args:
        dataset_name (str): Dataset file prefix (excluding path and extension).
        selected_columns (list): Features to use.

    Returns:
        dict: Mapping of feature names to summarized coefficient magnitudes.
    """
    filepath = f"data/basic/basic_{dataset_name}.csv"
    df = pd.read_csv(filepath)

    y = df['ml_course'].astype('category')
    if len(y.cat.categories) != 2:
        raise ValueError("Target variable must have exactly 2 classes for binomial logistic regression.")

    X = df[selected_columns]
    categorical_cols, numeric_cols = classify_columns(X)

    # Single train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    lasso_pipeline = create_pipeline(categorical_cols, numeric_cols, model_type="logreg", lasso=True)
    lasso_pipeline.fit(X_train, y_train)

    model = lasso_pipeline.named_steps['model']
    coef = model.coef_[0]

    preprocessor = lasso_pipeline.named_steps['preprocess']
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)

    feature_names = cat_feature_names.tolist() + numeric_cols
    summarized_coefs = {}

    idx = 0
    for feature in categorical_cols:
        matching_features = [i for i, name in enumerate(cat_feature_names) if name.startswith(feature + "_")]
        matching_coefs = coef[matching_features]
        summarized_coefs[feature] = np.max(np.abs(matching_coefs))  # max absolute value
        idx += len(matching_features)

    for feature in numeric_cols:
        summarized_coefs[feature] = abs(coef[idx])
        idx += 1

    print(f"\nSummarized Coefficients for {dataset_name} (Lasso):")
    for fname, c in summarized_coefs.items():
        print(f"{fname}: {c:.4f}")

    return summarized_coefs
