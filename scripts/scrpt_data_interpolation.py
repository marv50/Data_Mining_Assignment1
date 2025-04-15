import os

from src.basic.csv_utils_basic import *
from src.basic.interpolation_basic import *
from src.basic.data_exploration_basic import *

def main():
    read_file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'basic_cleaned.csv')
    data = read_csv(read_file_path)

    categorical_cols, numeric_cols = classify_columns(data)
    
    combinations = [
        ("mode", "median"),
        ("mode", "knn"),
        ("knn", "median"),
        ("knn", "knn"),
    ]

    for cat_method, num_method in combinations:
        data_copy = data.copy()
        if cat_method == "mode":
            data_copy = impute_categoricals_mode(data_copy, categorical_cols)
        elif cat_method == "knn":
            data_copy = impute_categoricals_knn(data_copy, categorical_cols, k=3)
            
        if num_method == "median":
            data_copy = impute_numericals_median(data_copy, numeric_cols)
        elif num_method == "knn":
            data_copy = impute_numericals_knn(data_copy, numeric_cols, k=3)

        filename = f"cat-{cat_method}_num-{num_method}.csv"
        file_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', filename)
        data_copy.to_csv(file_path, index=False)
    

if __name__ == "__main__":
   main()