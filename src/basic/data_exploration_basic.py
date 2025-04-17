import os
import numpy as np
import numpy as np
import pandas as pd


def classify_columns(data_frame, binary_threshold=2):
    """
    Classifies columns in a DataFrame into categorical and numeric types based on unique values.
    Categorical columns are those with a limited number of unique values or binary values.
    Numeric columns are those with a larger number of unique values or continuous data types.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to classify columns from.
        binary_threshold (int): The threshold for classifying binary columns.

    Returns:
        tuple: A tuple containing two lists:
            - Categorical columns
            - Numeric columns
    """
    categorical_cols = []
    numeric_cols = []

    for col in data_frame.columns:
        unique_vals = data_frame[col].dropna().unique()
        num_unique = len(unique_vals)

        if pd.api.types.is_numeric_dtype(data_frame[col]):
            if num_unique <= binary_threshold and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                categorical_cols.append(col)
            elif pd.api.types.is_integer_dtype(data_frame[col]) and num_unique <= 5:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return categorical_cols, numeric_cols

def explore_dataset(dataframe):
    """
    Prints basic information about the DataFrame, including data types, summary statistics,
    and missing values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to explore.
    """
    print("~~~~~~~~~~~~~~~ Dataset Info ~~~~~~~~~~~~~~~")
    print(dataframe.info())
    print("\n~~~~~~~~~~~~~~~ Summary Statistics ~~~~~~~~~~~~~~~")
    print(dataframe.describe(include='all'))
    print("\n~~~~~~~~~~~~~~~ Missing Values ~~~~~~~~~~~~~~~")
    print(dataframe.isnull().sum())

def summarize_dataframe(df, iqr_thresh=float('inf'), iqr_degree_thresh = 120):
    """
    Summarizes the DataFrame by providing basic statistics for each column.
    The summary includes data type, non-null count, missing count, unique values,
    min, max, mean, outlier count (IQR), and example values.
    The function also handles periodic float columns (e.g., angles) by removing outliers
    based on the specified threshold.
    NOTE: Only works for non-periodic float columns. Needs to be updated for periodic float columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to summarize.
        iqr_thresh (float): The IQR threshold for outlier detection.
        iqr_degree_thresh (float): The threshold in degrees for periodic float columns.
    
    Returns:
        pd.DataFrame: A summary DataFrame containing the statistics for each column.
    """
    summary = []

    for col in df.columns:
        col_data = df[col]
        col_dtype = col_data.dtype
        non_null_count = col_data.notnull().sum()
        missing_count = col_data.isnull().sum() + col_data.isna().sum()
        unique_count = col_data.nunique()

        example_values = col_data.dropna().unique()[:2] if non_null_count > 0 else []
        example_values = [str(val) for val in example_values]  # Convert to string for better readability

        min_val, max_val, mean_val, outlier_count = None, None, None, None

        if pd.api.types.is_numeric_dtype(col_dtype) and not set(col_data.dropna().unique()).issubset({0, 1}):
            min_val = np.round(col_data.min(), 5)
            max_val = np.round(col_data.max(), 5)
            mean_val = np.round(col_data.mean(), 5)

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_thresh * IQR
            upper_bound = Q3 + iqr_thresh * IQR

            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        summary.append({
            'Column': col,
            'Data Type': col_dtype,
            'Non-Null Count': non_null_count,
            'Missing Count': missing_count,
            'Unique Values': unique_count,
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Outlier Count (IQR)': outlier_count,
            'Example Values': example_values
        })

    summary_df = pd.DataFrame(summary)
    return summary_df

def summarize_categories(data_frame, file_name="summarized_categories.csv"):
    """
    Summarizes the categorical features in the DataFrame by providing value counts for each categorical column.
    The summary includes the value counts for each category in the categorical columns.
    The function saves the summary to a CSV file.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to summarize.
        file_name (str): The name of the file to save the summary. Default is "summarized_categories.csv".
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'data', 'basic', file_name)
    print("\n~~~~~~~~~~~~~~~ Categorical Feature Distributions ~~~~~~~~~~~~~~~")
    summary = []
    for col in data_frame.select_dtypes(include='object').columns:
        print(f"\n{col}:\n", data_frame[col].value_counts(dropna=False))
        value_counts = data_frame[col].value_counts(dropna=False).reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Column'] = col
        summary.append(value_counts)
    
    if file_path:
        pd.concat(summary).to_csv(file_path, index=False)