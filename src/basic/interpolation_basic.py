import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

def impute_categoricals(df, cat_cols, method='mode', k=5):
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

def impute_numericals(df, num_cols, method='median', k=5):
    if method == 'median':
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=k)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        raise ValueError("Imputation method must be 'median' or 'knn'.")

    return df