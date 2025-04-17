import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

def impute_categoricals_mode(data_frame, cat_cols):
    """
    Imputes missing values in categorical columns using the mode (most frequent value).
    This method replaces NaN values with the most common value in each column.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing categorical columns.
        cat_cols (list): List of categorical column names.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values in categorical columns imputed.
    """
    df_cat = data_frame[cat_cols].copy()
    for col in df_cat:
        df_cat[col] = df_cat[col].fillna(df_cat[col].mode().iloc[0])
    data_frame[cat_cols] = df_cat
    return data_frame

def impute_categoricals_knn(data_frame, cat_cols, k=5):
    """
    Impute missing values in categorical columns using KNN imputation.
    This method replaces NaN values with the most similar values based on K nearest neighbors.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing categorical columns.
        cat_cols (list): List of categorical column names.
        k (int): Number of neighbors to use for imputation.

    Returns:
        pd.DataFrame: The DataFrame with missing values in categorical columns imputed.
    """
    df_cat = data_frame[cat_cols].copy()

    for col in df_cat.columns:
        if df_cat[col].dtype == 'object' or isinstance(df_cat[col].dtype, pd.CategoricalDtype):
            df_cat[col] = df_cat[col].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(df_cat[col]) or pd.api.types.is_timedelta64_dtype(df_cat[col]):
            df_cat[col] = df_cat[col].astype(str)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded = encoder.fit_transform(df_cat)
    imputed = KNNImputer(n_neighbors=k).fit_transform(encoded)
    df_cat[:] = encoder.inverse_transform(imputed)
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype('category')
    data_frame[cat_cols] = df_cat
    return data_frame

def impute_numericals_median(data_frame, num_cols):
    """
    Imputes missing values in numeric columns using the median.
    This method replaces NaN values with the median value of each column.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing numeric columns.
        num_cols (list): List of numeric column names.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values in numeric columns imputed.
    """
    for col in num_cols:
        data_frame[col] = data_frame[col].fillna(data_frame[col].median())
    return data_frame

def impute_numericals_knn(data_frame, num_cols, k=5):
    """
    Imputes missing values in numeric columns using KNN imputation.
    This method replaces NaN values with the most similar values based on K nearest neighbors.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing numeric columns.
        num_cols (list): List of numeric column names.
        k (int): Number of neighbors to use for imputation.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values in numeric columns imputed.
    """
    imputer = KNNImputer(n_neighbors=k)
    data_frame[num_cols] = imputer.fit_transform(data_frame[num_cols])
    return data_frame

def interpolate_sin_cos(data_frame):
    """
    Interpolates the sine and cosine values of the 'bedtime' column in the DataFrame.
    This method is used to fill in missing values in the sine and cosine representations of time data.

    It calculates the angles from the sine and cosine values, performs linear interpolation,
    and then updates the sine and cosine values based on the interpolated angles.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame containing the 'bedtime_sin' and 'bedtime_cos' columns.
    
    Returns:
        pd.DataFrame: The DataFrame with interpolated sine and cosine values.
    """
    if 'bedtime_sin' in data_frame.columns and 'bedtime_cos' in data_frame.columns:
        angles = np.arctan2(data_frame['bedtime_sin'], data_frame['bedtime_cos'])
        
        anglesnorm_interpolated = pd.Series(angles).interpolate(method='linear')
        
        angles_interpolated = anglesnorm_interpolated % (2 * np.pi)
        
        data_frame['bedtime_sin'] = np.sin(angles_interpolated)
        data_frame['bedtime_cos'] = np.cos(angles_interpolated)
        data_frame['bedtime_anglenorm'] = anglesnorm_interpolated
        data_frame['bedtime_angle'] = angles_interpolated
    
    return data_frame