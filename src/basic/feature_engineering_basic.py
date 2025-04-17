import numpy as np
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


def calculate_sleep_duration(bedtime):
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
    if 'bedtime' in data_frame.columns:
        data_frame['sleep_duration_to_10am'] = data_frame['bedtime'].apply(calculate_sleep_duration)
    return data_frame


def convert_binary_floats_to_categoricals(df, categorical_cols):
    for col in categorical_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('category')
    return df

def convert_categories_to_ordinals(data_frame, categorical_cols):
    data_frame['program'] = OrdinalEncoder().fit_transform(data_frame[['program']])
    data_frame['gender'] = OrdinalEncoder().fit_transform(data_frame[['gender']])
    for col in categorical_cols:
        if data_frame[col].dtype.name == 'category':
            data_frame[col] = OrdinalEncoder().fit_transform(data_frame[[col]])
    return data_frame


"""
Converts Objects to Categoricals
This function converts object columns to categorical columns in the DataFrame.
NOTE: Currently only for gender and program to avoid changing other object attributes
"""
def convert_objects_to_categoricals(data_frame, categorical_cols):
    data_frame['program'] = data_frame['program'].astype('category')
    data_frame['gender'] = data_frame['gender'].astype('category')
    # for col in categorical_cols:
    #     if data_frame[col].dtype == 'object':
    #         data_frame[col] = data_frame[col].astype('category')
    return data_frame


def time_to_seconds(time_str):
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
    if pd.isnull(time_str): return (np.nan, np.nan, np.nan, np.nan)
    seconds = time_to_seconds(time_str)
    angle = 2 * np.pi * seconds / 86400 
    angle_norm = np.arctan2(np.sin(angle), np.cos(angle))

    return np.sin(angle), np.cos(angle), angle_norm, angle

def convert_to_positional_times(data_frame):
    data_frame['bedtime_sin'], data_frame['bedtime_cos'], data_frame['bedtime_anglenorm'], data_frame['bedtime_angle'] = zip(*data_frame['bedtime'].apply(circular_encode_time))
    
    return data_frame

def normalize_stress(data_frame):
    if 'stress_level' in data_frame.columns:
        data_frame['stress_level'] = (data_frame['stress_level'] - data_frame['stress_level'].min()) / (data_frame['stress_level'].max() - data_frame['stress_level'].min())
    return data_frame