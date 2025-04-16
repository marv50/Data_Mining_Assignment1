import os
import numpy as np
import numpy as np
import pandas as pd


def classify_columns(df, binary_threshold=2):
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        num_unique = len(unique_vals)

        if pd.api.types.is_numeric_dtype(df[col]):
            if num_unique <= binary_threshold and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                categorical_cols.append(col)
            elif pd.api.types.is_integer_dtype(df[col]) and num_unique <= 5:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return categorical_cols, numeric_cols

def explore_dataset(dataframe):
    print("~~~~~~~~~~~~~~~ Dataset Info ~~~~~~~~~~~~~~~")
    print(dataframe.info())
    print("\n~~~~~~~~~~~~~~~ Summary Statistics ~~~~~~~~~~~~~~~")
    print(dataframe.describe(include='all'))
    print("\n~~~~~~~~~~~~~~~ Missing Values ~~~~~~~~~~~~~~~")
    print(dataframe.isnull().sum())

def summarize_dataframe(df, iqr_thresh=float('inf')):
    summary = []

    for col in df.columns:
        col_data = df[col]
        col_dtype = col_data.dtype
        non_null_count = col_data.notnull().sum()
        missing_count = col_data.isnull().sum() + col_data.isna().sum()
        unique_count = col_data.nunique()

        example_values = col_data.dropna().unique()[:3] if non_null_count > 0 else []
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

def summarize_categories(df, file_name="summarized_categories.csv"):
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'data', 'basic', file_name)
    print("\n~~~~~~~~~~~~~~~ Categorical Feature Distributions ~~~~~~~~~~~~~~~")
    summary = []
    for col in df.select_dtypes(include='object').columns:
        print(f"\n{col}:\n", df[col].value_counts(dropna=False))
        value_counts = df[col].value_counts(dropna=False).reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Column'] = col
        summary.append(value_counts)
    
    if file_path:
        pd.concat(summary).to_csv(file_path, index=False)