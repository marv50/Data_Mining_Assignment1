import numpy as np
import pandas as pd

def remove_duplicates(data_frame):
    """
    Removes duplicate rows from the DataFrame.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame from which to remove duplicates.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    return data_frame.drop_duplicates()

def remove_iqr_outliers(data_frame, threshold=1.5):
    """
    Removes outliers from the DataFrame based on the IQR method.
    Outliers are defined as values that are below Q1 - threshold * IQR or above Q3 + threshold * IQR.
    Only works for non-periodic float columns.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame from which to remove outliers.
        threshold (float): The multiplier for the IQR to define outliers.

            Default is 1.5, which is the standard IQR method.
    
    Returns:
        pd.DataFrame: The DataFrame with outliers replaced by NaN.
    """
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
    """
    Removes outliers from the DataFrame based on the periodic nature of time data.
    Outliers are defined as angles that are outside the specified threshold in degrees.

    Only works for periodic float columns (e.g., 'bedtime_sin', 'bedtime_cos').
    The angles are calculated using the arctangent of the sine and cosine values.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame from which to remove outliers.
        threshold_degrees (float): The threshold in degrees to define outliers.

            Default is 120 degrees.
    
    Returns:
        pd.DataFrame: The DataFrame with outliers replaced by NaN.
    """
    df = data_frame.copy()

    if 'bedtime_sin' in df.columns and 'bedtime_cos' in df.columns:
        angles = np.arctan2(df['bedtime_sin'], df['bedtime_cos'])
        mean_angle = np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles)))
        angle_diff = np.abs(angles - mean_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        threshold_radians = 2 * np.pi * (threshold_degrees / 360)
        outlier_mask = angle_diff > threshold_radians

        df.loc[outlier_mask, ['bedtime_sin', 'bedtime_cos', 'bedtime_anglenorm', 'bedtime_angle']] = np.nan

    return df