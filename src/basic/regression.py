"""
regression.py

This script provides utilities to run regression experiments using linear regression 
and K-Nearest Neighbors (KNN) regression models. It includes functions for:
- Creating a machine learning pipeline with preprocessing
- Training and evaluating models
- Running repeated experiments to assess model performance
- Plotting distributions of evaluation metrics

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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.basic.interpolation import classify_columns  # Assuming you have this helper function

# --- Utility Functions ---

def create_regression_pipeline(categorical_cols, numeric_cols, model_type='linear', n_neighbors=5):
    """
    Creates a regression pipeline with preprocessing steps.

    Args:
        categorical_cols (list): List of categorical feature names.
        numeric_cols (list): List of numeric feature names.
        model_type (str): 'linear' for LinearRegression, 'knn' for KNeighborsRegressor.
        n_neighbors (int): Number of neighbors for KNN (if model_type='knn').

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn pipeline with preprocessing and model.
    """
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model)
    ])
    return pipeline

def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """
    Trains a pipeline and evaluates it on a test set.

    Args:
        pipeline (Pipeline): The regression pipeline.
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.

    Returns:
        tuple: (mean_squared_error, mean_absolute_error) on the test set.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae

def run_experiment(dataset_name, selected_columns, model_type='linear', n_neighbors=5, n_runs=50):
    """
    Runs multiple training/evaluation cycles for a given dataset and model type.

    Args:
        dataset_name (str): Name of the dataset file (excluding prefix/suffix).
        selected_columns (list): Columns to be used as features.
        model_type (str): 'linear' for LinearRegression, 'knn' for KNeighborsRegressor.
        n_neighbors (int): Number of neighbors for KNN (if model_type='knn').
        n_runs (int): Number of repeated random train/test splits.

    Returns:
        tuple: Arrays of MSEs and MAEs for all runs.
    """
    filepath = f"data/basic/basic_{dataset_name}.csv"
    df = pd.read_csv(filepath)

    X = df[selected_columns]
    y = df['stress_level']  # Target variable

    categorical_cols, numeric_cols = classify_columns(X)

    mses = []
    maes = []

    for random_state in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        pipeline = create_regression_pipeline(
            categorical_cols, numeric_cols, model_type=model_type, n_neighbors=n_neighbors
        )
        mse, mae = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

        mses.append(mse)
        maes.append(mae)

    mses = np.array(mses)
    maes = np.array(maes)
    
    return mses, maes

def plot_results(mses, maes, dataset_name, model_type='linear'):
    """
    Plots histograms of MSE and MAE distributions over multiple experiment runs.

    Args:
        mses (np.ndarray): Array of MSE values.
        maes (np.ndarray): Array of MAE values.
        dataset_name (str): Dataset name for title labeling.
        model_type (str): Model type for title labeling ('linear' or 'knn').
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].hist(mses, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title(f'MSE Distribution for {dataset_name} ({model_type})')
    axs[0].set_xlabel('MSE')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(maes, bins=20, color='salmon', edgecolor='black')
    axs[1].set_title(f'MAE Distribution for {dataset_name} ({model_type})')
    axs[1].set_xlabel('MAE')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
