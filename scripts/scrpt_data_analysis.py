import os

from src.basic.csv_utils_basic import *
from src.basic.data_exploration_basic import *
from src.basic.plotting_basic import *
from src.basic.data_cleaning_basic import *

def analyze_data(data_frame, name_extension, iqr_thresh=1.5):
    explore_dataset(data_frame)

    summary_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'summary' + name_extension + '.csv')
    summary_noclean = summarize_dataframe(data_frame, iqr_thresh=iqr_thresh)
    summary_noclean.to_csv(summary_path, index=False)
    create_summary_table_visualization(summary_noclean, file_name='summary' + name_extension + '.png')

    plot_distributions(data_frame, file_name="distributions" + name_extension + ".png")
    plot_correlation_heatmap(data_frame, file_name="correlation_heatmap" + name_extension + ".png")

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'ODI-2025 - adjusted.csv')
    data = read_csv(read_file_path)

    categorical_cols, numeric_cols = classify_columns(data)
    convert_binary_floats_to_categoricals(data, categorical_cols)

    analyze_data(data, name_extension='_noclean', iqr_thresh=1.5)

    data_cleaned = remove_duplicates(data)
    data_cleaned = remove_iqr_outliers(data, threshold=1.5)
    data_cleaned = convert_bedtime_to_sleeptime_10am(data_cleaned)

    analyze_data(data_cleaned, name_extension='_clean', iqr_thresh=float('inf'))

    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_cleaned.csv')
    dataframe_to_csv(data_cleaned, write_file_path)

    summarize_categories(data_cleaned, "category_summary_cleaned.csv")

if __name__ == "__main__":
   main()