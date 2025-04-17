import os

from src.basic.csv_utils_basic import *
from src.basic.data_exploration_basic import *
from src.basic.plotting_basic import *
from src.basic.data_cleaning_basic import *
from src.basic.feature_engineering_basic import *
from src.basic.interpolation_basic import *

from sklearn.preprocessing import OrdinalEncoder
from scripts.scrpt_utils import analyze_data

def feature_engineering(data_frame):
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_cleaned.csv')
    data_cleaned = read_csv(read_file_path)

    categorical_cols, numeric_cols = classify_columns(data_cleaned)

    data_feature_engineered = data_cleaned.copy()
    
    #data_feature_engineered = normalize_stress(data_feature_engineered)
    data_feature_engineered = convert_bedtime_to_sleeptime_10am(data_feature_engineered)
    data_feature_engineered = convert_to_positional_times(data_feature_engineered)
    data_feature_engineered = remove_time_outliers(data_feature_engineered, threshold_degrees=120) # Cleaning step after converting to positional times

    analyze_data(data_feature_engineered, name_extension='_engineered_inclbooleans', iqr_thresh=float('inf'))

    data_feature_engineered = convert_binary_floats_to_categoricals(data_feature_engineered, categorical_cols)
    data_feature_engineered = convert_objects_to_categoricals(data_feature_engineered, categorical_cols)

    plot_positional_times(data_feature_engineered, file_name="positional_times_engineered.png")

    analyze_data(data_feature_engineered, name_extension='_engineered', iqr_thresh=float('inf'))
    summarize_categories(data_feature_engineered, "category_summary_engineered.csv")
    
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_engineered.csv')
    dataframe_to_csv(data_feature_engineered, write_file_path)

    return data_feature_engineered

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
    data_feature_engineered = feature_engineering(data_frame=None)
    interpolation(data_frame=data_feature_engineered)


if __name__ == "__main__":
   main()