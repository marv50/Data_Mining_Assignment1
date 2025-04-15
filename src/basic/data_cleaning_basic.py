import numpy as np
import numpy as np
import pandas as pd


def remove_duplicates(data_frame):
    return data_frame.drop_duplicates()

def remove_iqr_outliers(data_frame, threshold=1.5):
    float_columns = data_frame.select_dtypes(include=['float64']).columns
    filtered_data_frame = data_frame[float_columns]

    Q1 = filtered_data_frame.quantile(0.25)
    Q3 = filtered_data_frame.quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    data_frame_no_outliers = data_frame[~((filtered_data_frame < lower_bound) | (filtered_data_frame > upper_bound)).any(axis=1)]

    return data_frame_no_outliers


def convert_binary_floats_to_categoricals(df, categorical_cols):
    for col in categorical_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('category')
    return df