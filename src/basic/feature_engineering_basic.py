import numpy as np
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


def calculate_sleep_duration(bedtime):
    """
    Calculates the sleep duration in hours until 10 AM the next day.
    If bedtime is NaN, returns NaN.
    If bedtime is not in the correct format, returns NaN.

    If bedtime is before 10 AM, calculates the duration until 10 AM the next day.
    If bedtime is after 10 AM, calculates the duration until 10 AM the day after next.

    Parameters:
        bedtime (str): Bedtime in the format 'HH:MM:SS'.
    
    Returns:
        float: Sleep duration in hours until 10 AM the next day.
    """
    bedtime = pd.to_datetime(bedtime, format='%H:%M:%S', errors='coerce')
    print(bedtime)
    if pd.isna(bedtime):
        return np.nan
    if bedtime.hour >= 0 and bedtime.hour <= 10:
        next_day_10am = bedtime.replace(hour=10, minute=0)
    else:
        next_day_10am = bedtime.replace(hour=10, minute=0) + pd.Timedelta(days=1)
    sleep_duration = (next_day_10am - bedtime).total_seconds() / 3600
    return sleep_duration

def convert_bedtime_to_sleeptime_10am(data_frame):
    """
    Converts the 'bedtime' column in the DataFrame to a new column 'sleep_to_10am'.
    The new column contains the sleep duration in hours until 10 AM the next day.

    If 'bedtime' is NaN, 'sleep_to_10am' will also be NaN.
    If 'bedtime' is not in the correct format, 'sleep_to_10am' will also be NaN.

    If 'bedtime' is before 10 AM, 'sleep_to_10am' will be the duration until 10 AM the next day.
    If 'bedtime' is after 10 AM, 'sleep_to_10am' will be the duration until 10 AM the day after next.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the 'bedtime' column.

    Returns:
        pd.DataFrame: The DataFrame with the new 'sleep_to_10am' column.
    """
    if 'bedtime' in data_frame.columns:
        data_frame['sleep_to_10am'] = data_frame['bedtime'].apply(calculate_sleep_duration)
    return data_frame


def convert_binary_floats_to_categoricals(data_frame, categorical_cols):
    """
    Converts binary float columns (0.0 and 1.0) to categorical columns in the DataFrame.
    The function checks if the unique values in the column are a subset of {0.0, 1.0} and converts them to categorical.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the columns to be converted.
        categorical_cols (list): List of categorical columns to check.
    
    Returns:
        pd.DataFrame: The DataFrame with the binary float columns converted to categorical.
    """
    for col in categorical_cols:
        unique_vals = set(data_frame[col].dropna().unique())
        if unique_vals.issubset({0.0, 1.0}):
            data_frame[col] = data_frame[col].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('category')
    return data_frame

def convert_objects_to_categoricals(data_frame, categorical_cols):
    """
    Converts Objects to Categoricals
    This function converts object columns to categorical columns in the DataFrame.
    NOTE: Currently only for gender and program to avoid changing other object attributes

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the columns to be converted.
        categorical_cols (list): List of categorical columns to check.
    
    Returns:
        pd.DataFrame: The DataFrame with the binary float columns converted to categorical.
    """
    data_frame['program'] = data_frame['program'].astype('category')
    data_frame['gender'] = data_frame['gender'].astype('category')
    # for col in categorical_cols:
    #     if data_frame[col].dtype == 'object':
    #         data_frame[col] = data_frame[col].astype('category')
    return data_frame


def time_to_seconds(time_str):
    """
    Converts a time string in the format 'HH:MM:SS' to seconds.
    If the time string is NaN or not in the correct format, returns NaN.

    Parameters:
        time_str (str): Time string in the format 'HH:MM:SS'.

    Returns:
        float: Time in seconds.
        float: NaN if the time string is NaN or not in the correct format.
    """
    if pd.isna(time_str):
        return np.nan
    try:
        time_obj = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
        if pd.isna(time_obj):
            return np.nan
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    except ValueError:
        return np.nan
    
def circular_encode_time(time_str):
    """
    Converts a time string in the format 'HH:MM:SS' to sine and cosine values.
    The sine and cosine values represent the circular nature of time.
    The angle is normalized to the range [0, 2π].
    If the time string is NaN or not in the correct format, returns NaN.

    Parameters:
        time_str (str): Time string in the format 'HH:MM:SS'.
    
    Returns:
        tuple: Sine value, cosine value, normalized angle, and angle in radians.
    """
    if pd.isnull(time_str): 
        return (np.nan, np.nan, np.nan, np.nan)
    seconds = time_to_seconds(time_str)
    angle = 2 * np.pi * seconds / 86400 
    angle_norm = np.arctan2(np.sin(angle), np.cos(angle))

    return np.sin(angle), np.cos(angle), angle_norm, angle

def convert_to_positional_times(data_frame):
    """
    Converts the 'bedtime' column in the DataFrame to sine and cosine values.
    The sine and cosine values represent the circular nature of time.
    The angle is normalized to the range [0, 2π].
    If the 'bedtime' column is NaN or not in the correct format, returns NaN.
    The function also creates new columns 'bedtime_sin', 'bedtime_cos', 'bedtime_anglenorm', and 'bedtime_angle'.
    
    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the 'bedtime' column.
    
    Returns:
        pd.DataFrame: The DataFrame with the new columns 'bedtime_sin', 'bedtime_cos', 'bedtime_anglenorm', and 'bedtime_angle'.
    """
    data_frame['bedtime_sin'], data_frame['bedtime_cos'], data_frame['bedtime_anglenorm'], data_frame['bedtime_angle'] = zip(*data_frame['bedtime'].apply(circular_encode_time))
    
    return data_frame

def normalize_stress(data_frame):
    """
    Normalizes the 'stress_level' column in the DataFrame to a range of [0, 1].
    The normalization is done using min-max scaling.
    If the 'stress_level' column is NaN or not in the correct format, returns NaN.
    NOTE: Not currently used

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the 'stress_level' column.

    Returns:
        pd.DataFrame: The DataFrame with the normalized 'stress_level' column.
    """
    if 'stress_level' in data_frame.columns:
        data_frame['stress_level'] = (data_frame['stress_level'] - data_frame['stress_level'].min()) / (data_frame['stress_level'].max() - data_frame['stress_level'].min())
    return data_frame