import numpy as np
import pandas as pd

def remove_duplicates(data_frame):
    return data_frame.drop_duplicates()

def remove_iqr_outliers(data_frame, threshold=1.5):
    df = data_frame.copy()
    float_columns = df.select_dtypes(include=['float64']).columns

    # Optional: remove binary float columns (i.e., 0/1)
    float_columns = [col for col in float_columns if df[col].nunique() > 2]

    Q1 = df[float_columns].quantile(0.25)
    Q3 = df[float_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Replace only the outlier cells with NaN
    for col in float_columns:
        outliers = (df[col] < lower_bound[col]) | (df[col] > upper_bound[col])
        df.loc[outliers, col] = np.nan

    return df

def remove_time_outliers(data_frame, threshold_degrees=120):
    df = data_frame.copy()

    if 'bedtime_sin' in df.columns and 'bedtime_cos' in df.columns:
        angles = np.arctan2(df['bedtime_sin'], df['bedtime_cos'])
        mean_angle = np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles)))
        angle_diff = np.abs(angles - mean_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        threshold_radians = 2 * np.pi * (threshold_degrees / 360)
        outlier_mask = angle_diff > threshold_radians

        df.loc[outlier_mask, ['bedtime_sin', 'bedtime_cos']] = np.nan

    return df