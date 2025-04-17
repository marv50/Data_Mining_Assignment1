from src.classification import *

datasets = ["mode_knn", "mode_median", "knn_median", "knn_knn"]
selected_columns = ["program", "sports_hours", "stress_level", "used_chatgpt", "gender", "ir_course", "bedtime_angle", "stats_course"]

n_runs = 50  # Number of random runs

# Store results
results_mean = {'Lasso': {}, 'Ridge': {}}
results_stderr = {'Lasso': {}, 'Ridge': {}}  # <-- storing standard error

for dataset in datasets:
    mean_lasso, std_lasso = run_experiment(dataset, selected_columns, lasso=True, n_runs=n_runs)
    mean_ridge, std_ridge = run_experiment(dataset, selected_columns, lasso=False, n_runs=n_runs)

    results_mean['Lasso'][dataset] = mean_lasso
    results_stderr['Lasso'][dataset] = std_lasso / np.sqrt(n_runs)  # <-- standard error

    results_mean['Ridge'][dataset] = mean_ridge
    results_stderr['Ridge'][dataset] = std_ridge / np.sqrt(n_runs)  # <-- standard error
    run_single_experiment(dataset, selected_columns)
    

# --- Print Results -


print("Average Accuracy (Mean ± Standard Error) over multiple runs:")
for dataset in datasets:
    print(f"{dataset}: Lasso = {results_mean['Lasso'][dataset]:.4f} ± {results_stderr['Lasso'][dataset]:.4f}, Ridge = {results_mean['Ridge'][dataset]:.4f} ± {results_stderr['Ridge'][dataset]:.4f}")

# --- Plotting Results ---

# --- Plotting Results with Larger Fonts ---

labels = datasets
x = np.arange(len(labels))  # label locations
fig, ax = plt.subplots(figsize=(14, 10))

# Scatter points for Ridge
ax.errorbar(
    x=x - 0.1,
    y=[results_mean['Ridge'][d] for d in datasets],
    yerr=[results_stderr['Ridge'][d] for d in datasets],
    fmt='o', label='Ridge (L2)', capsize=8, markersize=12, color='blue'
)

# Scatter points for Lasso
ax.errorbar(
    x=x + 0.1,
    y=[results_mean['Lasso'][d] for d in datasets],
    yerr=[results_stderr['Lasso'][d] for d in datasets],
    fmt='o', label='Lasso (L1)', capsize=8, markersize=12, color='red'
)

ax.set_ylabel('Accuracy', fontsize=24)
ax.set_xlabel('Dataset', fontsize=24)
ax.set_title('Accuracy (Mean ± Standard Error) by Dataset and Regularizer', fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.tick_params(axis='x', labelsize=22)
ax.legend(fontsize=20)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("fig/logistic_results.pdf")
plt.show()
