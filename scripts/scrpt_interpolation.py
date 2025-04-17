import os

from src.basic.csv_utils_basic import *
from src.basic.data_exploration_basic import *
from src.basic.plotting_basic import *
from src.basic.data_cleaning_basic import *
from src.basic.feature_engineering_basic import *
from src.basic.interpolation_basic import *

from scripts.scrpt_utils import analyze_data

def interpolation(data_frame):
    combinations = [
        ("mode", "median"),
        ("mode", "knn"),
        ("knn", "median"),
        ("knn", "knn"),
    ]

    summary = summarize_dataframe(data_frame, iqr_thresh=float('inf'))
    create_summary_table_visualization(summary)

    for cat_method, num_method in combinations:
        data_interpolated = data_frame.copy()
        categorical_cols, numeric_cols = classify_columns(data_interpolated)

        data_interpolated = interpolate_sin_cos(data_interpolated)

        if cat_method == "mode":
            data_interpolated = impute_categoricals_mode(data_interpolated, categorical_cols)
        elif cat_method == "knn":
            data_interpolated = impute_categoricals_knn(data_interpolated, categorical_cols, k=3)

        if num_method == "median":
            data_interpolated = impute_numericals_median(data_interpolated, numeric_cols)
        elif num_method == "knn":
            data_interpolated = impute_numericals_knn(data_interpolated, numeric_cols, k=3)

        file_extension = f"_{cat_method}_{num_method}"

        plot_positional_times(data_interpolated, file_name="positional_times" + file_extension + ".png")

        analyze_data(data_interpolated, name_extension=file_extension, iqr_thresh=float('inf'))

        write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', "basic" + file_extension + ".csv")
        dataframe_to_csv(data_interpolated, write_file_path)

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_engineered.csv')
    data_feature_engineered = read_csv(read_file_path)
    interpolation(data_frame=data_feature_engineered)


if __name__ == "__main__":
   main()