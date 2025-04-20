"""
scrpt_combined_regression.py

This script compares Linear Regression and K-Nearest Neighbors (KNN) Regression
on different datasets using selected features. Evaluation metrics include 
Mean Squared Error (MSE) and Mean Absolute Error (MAE) across multiple random splits.

Dependencies:
- numpy
- matplotlib
- src.basic.regression (must define `run_experiment`)

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from src.basic.regression import run_experiment

# --- Configuration ---

datasets = ["mode_knn", "mode_median", "knn_median", "knn_knn"]
selected_columns = [
    "program", "sports_hours", "ml_course", "used_chatgpt",
    "gender", "ir_course", "bedtime_angle", "stats_course"
]

n_runs = 50
n_neighbors = 5  # For KNN regression

# --- Storage for Results ---

results_mean = {'MSE': {}, 'MAE': {}}
results_stderr = {'MSE': {}, 'MAE': {}}

# --- Run Experiments: Linear Regression ---

for dataset in datasets:
    mse_values, mae_values = run_experiment(dataset, selected_columns, n_runs=n_runs)
    results_mean['MSE'][dataset] = np.mean(mse_values)
    results_stderr['MSE'][dataset] = np.std(mse_values) / np.sqrt(n_runs)
    results_mean['MAE'][dataset] = np.mean(mae_values)
    results_stderr['MAE'][dataset] = np.std(mae_values) / np.sqrt(n_runs)

# Store Linear Regression results
linear_mse_mean = [results_mean['MSE'][d] for d in datasets]
linear_mse_stderr = [results_stderr['MSE'][d] for d in datasets]
linear_mae_mean = [results_mean['MAE'][d] for d in datasets]
linear_mae_stderr = [results_stderr['MAE'][d] for d in datasets]

# --- Run Experiments: KNN Regression ---

for dataset in datasets:
    mse_values, mae_values = run_experiment(dataset, selected_columns, model_type='knn', n_neighbors=n_neighbors, n_runs=n_runs)
    results_mean['MSE'][dataset] = np.mean(mse_values)
    results_stderr['MSE'][dataset] = np.std(mse_values) / np.sqrt(n_runs)
    results_mean['MAE'][dataset] = np.mean(mae_values)
    results_stderr['MAE'][dataset] = np.std(mae_values) / np.sqrt(n_runs)

# Store KNN Regression results
knn_mse_mean = [results_mean['MSE'][d] for d in datasets]
knn_mse_stderr = [results_stderr['MSE'][d] for d in datasets]
knn_mae_mean = [results_mean['MAE'][d] for d in datasets]
knn_mae_stderr = [results_stderr['MAE'][d] for d in datasets]

# --- Plotting Results ---

labels = datasets
x = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Plot MSE
ax1.errorbar(
    x=x - 0.15, y=linear_mse_mean, yerr=linear_mse_stderr,
    fmt='o', label='Linear Regression MSE', capsize=8,
    markersize=12, color='blue'
)
ax1.errorbar(
    x=x + 0.15, y=knn_mse_mean, yerr=knn_mse_stderr,
    fmt='o', label='KNN Regression MSE', capsize=8,
    markersize=12, color='red'
)
ax1.set_ylabel('MSE', fontsize=24)
ax1.set_xlabel('Dataset', fontsize=24)
ax1.set_title('MSE: Linear vs KNN Regression', fontsize=28)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=22)
ax1.tick_params(axis='y', labelsize=22)
ax1.tick_params(axis='x', labelsize=22)
ax1.legend(fontsize=20)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot MAE
ax2.errorbar(
    x=x - 0.15, y=linear_mae_mean, yerr=linear_mae_stderr,
    fmt='o', label='Linear Regression MAE', capsize=8,
    markersize=12, color='lightblue'
)
ax2.errorbar(
    x=x + 0.15, y=knn_mae_mean, yerr=knn_mae_stderr,
    fmt='o', label='KNN Regression MAE', capsize=8,
    markersize=12, color='pink'
)
ax2.set_ylabel('MAE', fontsize=24)
ax2.set_xlabel('Dataset', fontsize=24)
ax2.set_title('MAE: Linear vs KNN Regression', fontsize=28)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=22)
ax2.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='x', labelsize=22)
ax2.legend(fontsize=20)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Ensure figure output directory exists
os.makedirs("fig", exist_ok=True)

plt.savefig("fig/comparison_mse_mae.pdf")
plt.show()
