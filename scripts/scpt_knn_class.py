from src.basic.classification import *

datasets = ["mode_knn", "mode_median", "knn_median", "knn_knn"]
full_feature_set = ["program", "sports_hours", "stress_level", "used_chatgpt", "gender", "ir_course", "bedtime_angle", "stats_course", "est_age_days", "room_estimate"]
small_feature_set = ["program", "ir_course", "stress_level", "bedtime_angle"]

n_runs = 50

# Store results
results_mean_knn = {'Full': {}, 'Small': {}}
results_stderr_knn = {'Full': {}, 'Small': {}}

for dataset in datasets:
    # Full feature set
    mean_full, std_full = run_experiment(dataset, full_feature_set, n_runs=n_runs, model_type="knn")
    results_mean_knn['Full'][dataset] = mean_full
    results_stderr_knn['Full'][dataset] = std_full / np.sqrt(n_runs)

    # Small feature set
    mean_small, std_small = run_experiment(dataset, small_feature_set, n_runs=n_runs, model_type="knn")
    results_mean_knn['Small'][dataset] = mean_small
    results_stderr_knn['Small'][dataset] = std_small / np.sqrt(n_runs)

# --- Print Results ---

print("KNN Average Accuracy (Mean ± Standard Error) over multiple runs:")
for dataset in datasets:
    print(f"{dataset}: Full = {results_mean_knn['Full'][dataset]:.4f} ± {results_stderr_knn['Full'][dataset]:.4f}, "
          f"Small = {results_mean_knn['Small'][dataset]:.4f} ± {results_stderr_knn['Small'][dataset]:.4f}")

# --- Plotting Results ---

labels = datasets
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(14, 10))

# Full feature set
ax.errorbar(
    x=x - 0.1,
    y=[results_mean_knn['Full'][d] for d in datasets],
    yerr=[results_stderr_knn['Full'][d] for d in datasets],
    fmt='o', label='KNN (Full Features)', capsize=8, markersize=12, color='blue'
)

# Small feature set
ax.errorbar(
    x=x + 0.1,
    y=[results_mean_knn['Small'][d] for d in datasets],
    yerr=[results_stderr_knn['Small'][d] for d in datasets],
    fmt='o', label='KNN (Selected 4 Features)', capsize=8, markersize=12, color='green'
)

ax.set_ylabel('Accuracy', fontsize=24)
ax.set_xlabel('Dataset', fontsize=24)
ax.set_title('KNN Accuracy (Full vs Selected Features)', fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.tick_params(axis='x', labelsize=22)
ax.legend(fontsize=20)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("fig/knn_features_comparison.pdf")
plt.show()
