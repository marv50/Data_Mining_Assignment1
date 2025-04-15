import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

def impute_categoricals_mode(df, cat_cols):
    df_cat = df[cat_cols].copy()
    for col in df_cat:
        df_cat[col] = df_cat[col].fillna(df_cat[col].mode().iloc[0])
    df[cat_cols] = df_cat
    return df

def impute_categoricals_knn(df, cat_cols, k=5):
    df_cat = df[cat_cols].copy()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded = encoder.fit_transform(df_cat)
    imputed = KNNImputer(n_neighbors=k).fit_transform(encoded)
    df_cat[:] = encoder.inverse_transform(imputed)
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype('category')
    df[cat_cols] = df_cat
    return df

def impute_numericals_median(df, num_cols):
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def impute_numericals_knn(df, num_cols, k=5):
    imputer = KNNImputer(n_neighbors=k)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df
