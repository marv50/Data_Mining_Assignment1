import os

from src.basic.csv_utils_basic import *
from src.basic.data_exploration_basic import *
from src.basic.plotting_basic import *
from src.basic.data_cleaning_basic import *
from src.basic.feature_engineering_basic import *
from scripts.scrpt_utils import analyze_data

def main():
    """
    Read and analyze a dataset, clean it, and save the cleaned data to a CSV file.
    The script performs the following tasks:
    1. Reads a CSV file containing the dataset.
    2. Analyzes the dataset by exploring its structure and generating summary statistics.
    3. Cleans the dataset by removing duplicates and outliers using the IQR method.
    4. Saves the cleaned dataset to a new CSV file.
    5. Summarizes the categorical features and saves the summary to a CSV file.
    """
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_manualadj.csv')
    data = read_csv(read_file_path)

    analyze_data(data, name_extension='_noclean', iqr_thresh=1.5)

    data_cleaned = data.copy()
    
    data_cleaned = remove_duplicates(data_cleaned)
    data_cleaned = remove_iqr_outliers(data_cleaned, threshold=1.5)

    analyze_data(data_cleaned, name_extension='_clean', iqr_thresh=float('inf'))

    write_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_cleaned.csv')
    dataframe_to_csv(data_cleaned, write_file_path)

    summarize_categories(data_cleaned, "category_summary_cleaned.csv")

if __name__ == "__main__":
   main()