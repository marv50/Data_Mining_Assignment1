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
    
    data_feature_engineered = convert_bedtime_to_sleeptime_10am(data_feature_engineered)
    data_feature_engineered = convert_to_positional_times(data_feature_engineered)
    data_feature_engineered = remove_time_outliers(data_feature_engineered, threshold_degrees=120) # Cleaning step after converting to positional times

    analyze_data(data_feature_engineered, name_extension='_engineered_inclbooleans', iqr_thresh=float('inf'))

    data_feature_engineered = convert_objects_to_categoricals(data_feature_engineered, categorical_cols)
    data_feature_engineered = convert_binary_floats_to_categoricals(data_feature_engineered, categorical_cols)

    plot_positional_times(data_feature_engineered, file_name="positional_times_engineered.png")

    analyze_data(data_feature_engineered, name_extension='_engineered', iqr_thresh=float('inf'))
    summarize_categories(data_feature_engineered, "category_summary_engineered.csv")
    
    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_engineered.csv')
    dataframe_to_csv(data_feature_engineered, write_file_path)

    return data_feature_engineered

def main():
    feature_engineering(data_frame=None)


if __name__ == "__main__":
   main()