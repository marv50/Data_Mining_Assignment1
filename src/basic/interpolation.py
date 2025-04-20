"""
interpolation.py

This script handles the preprocessing of the ODI-2025 dataset. It:
- Classifies columns into categorical and numerical types
- Fixes binary floats into proper categorical types
- Replaces outliers with NaN values
- Imputes missing values for both categorical and numerical columns using different methods
- Saves multiple versions of the preprocessed dataset for further experiments

Dependencies:
- pandas
- numpy
- scikit-learn

Author: [Your Name]
Date: [Today's Date]
"""

import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

# Ensure output directory exists
os.makedirs("data/basic", exist_ok=True)

# Load dataset
df = pd.read_csv('data/basic/ODI-2025 - adjusted.csv')

# --------- Classification ---------

def classify_columns(df, binary_threshold=2):
    """
    Classifies columns into categorical and numeric based on data type and unique values.

    Args:
        df (pd.DataFrame): Input dataframe.
        binary_threshold (int): Maximum number of unique values to consider as binary categorical.

    Returns:
        tuple: (list of categorical columns, list of numeric columns)
    """
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

# --------- Binary Fix ---------

def convert_binary_floats_to_categoricals(df, categorical_cols):
    """
    Converts columns with binary float values (0.0, 1.0) into categorical type.

    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_cols (list): List of categorical column names.

    Returns:
        pd.DataFrame: Dataframe with updated binary categorical columns.
    """
    for col in categorical_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('category')
    return df

# --------- Outlier Replacement ---------

def replace_outliers_with_nan(df, numeric_cols):
    """
    Replaces outliers in numeric columns with NaN using the IQR method.

    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_cols (list): List of numeric column names.

    Returns:
        pd.DataFrame: Dataframe with outliers replaced by NaN.
    """
    for col in numeric_cols:
        if df[col].dtype.kind in "iufc":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].apply(lambda x: np.nan if pd.notna(x) and (x < lower or x > upper) else x)
    return df

# --------- Categorical Imputation ---------

def impute_categoricals(df, cat_cols, method='mode', k=5):
    """
    Imputes missing values in categorical columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        cat_cols (list): List of categorical column names.
        method (str): 'mode' or 'knn' for imputation strategy.
        k (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: Dataframe with imputed categorical columns.
    """
    df_cat = df[cat_cols].copy()

    if method == 'mode':
        for col in df_cat:
            df_cat[col] = df_cat[col].fillna(df_cat[col].mode().iloc[0])
    elif method == 'knn':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded = encoder.fit_transform(df_cat)
        imputed = KNNImputer(n_neighbors=k).fit_transform(encoded)
        df_cat[:] = encoder.inverse_transform(imputed)
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype('category')
    else:
        raise ValueError("Imputation method must be either 'mode' or 'knn'.")

    df[cat_cols] = df_cat
    return df

# --------- Numerical Imputation ---------

def impute_numericals(df, num_cols, method='median', k=5):
    """
    Imputes missing values in numeric columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        num_cols (list): List of numeric column names.
        method (str): 'median' or 'knn' for imputation strategy.
        k (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: Dataframe with imputed numeric columns.
    """
    if method == 'median':
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=k)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        raise ValueError("Imputation method must be 'median' or 'knn'.")

    return df


