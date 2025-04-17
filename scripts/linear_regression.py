from src.regression import *


datasets = ["mode_knn", "mode_median", "knn_median", "knn_knn"]
selected_columns = ["program", "sports_hours", "ml_course", "used_chatgpt", "gender", "ir_course", "bedtime_angle", "stats_course"]
                    
n_runs = 50  # Number of random runs

# Store results for MSE and MAE
results_mean = {'MSE': {}, 'MAE': {}}
results_stderr = {'MSE': {}, 'MAE': {}}  # <-- storing standard error

for dataset in datasets:
    mse_values, mae_values = run_experiment(dataset, selected_columns, n_runs=n_runs)

    # Store mean MSE and MAE for each dataset
    results_mean['MSE'][dataset] = np.mean(mse_values)
    results_stderr['MSE'][dataset] = np.std(mse_values) / np.sqrt(n_runs)

    results_mean['MAE'][dataset] = np.mean(mae_values)
    results_stderr['MAE'][dataset] = np.std(mae_values) / np.sqrt(n_runs)

# --- Print Results ---
print("Average MSE and MAE (Mean ± Standard Error) over multiple runs:")
for dataset in datasets:
    print(f"{dataset}: MSE = {results_mean['MSE'][dataset]:.4f} ± {results_stderr['MSE'][dataset]:.4f}, "
          f"MAE = {results_mean['MAE'][dataset]:.4f} ± {results_stderr['MAE'][dataset]:.4f}")

# --- Plotting Results ---
# Plotting MSE and MAE results for comparison across datasets
labels = datasets
x = np.arange(len(labels))  # label locations
fig, ax = plt.subplots(figsize=(14, 10))

# Plot MSE with error bars
ax.errorbar(
    x=x - 0.1,
    y=[results_mean['MSE'][d] for d in datasets],
    yerr=[results_stderr['MSE'][d] for d in datasets],
    fmt='o', label='MSE', capsize=8, markersize=12, color='blue'
)

# Plot MAE with error bars
ax.errorbar(
    x=x + 0.1,
    y=[results_mean['MAE'][d] for d in datasets],
    yerr=[results_stderr['MAE'][d] for d in datasets],
    fmt='o', label='MAE', capsize=8, markersize=12, color='red'
)

ax.set_ylabel('Error Metric', fontsize=24)
ax.set_xlabel('Dataset', fontsize=24)
ax.set_title('MSE and MAE (Mean ± Standard Error) by Dataset', fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.tick_params(axis='x', labelsize=22)
ax.legend(fontsize=20)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("fig/linear_regression_results.pdf")
plt.show()
