import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import os

# Ensure output directory exists
os.makedirs("data/basic", exist_ok=True)

# Load dataset
df = pd.read_csv('data/basic/ODI-2025 - adjusted.csv')

# --------- Classification ---------
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

# --------- Binary Fix ---------
def convert_binary_floats_to_categoricals(df, categorical_cols):
    for col in categorical_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0.0, 1.0}):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('category')
    return df

# --------- Outlier Replacement ---------
def replace_outliers_with_nan(df, numeric_cols):
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
    if method == 'median':
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=k)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        raise ValueError("Imputation method must be 'median' or 'knn'.")

    return df

# --------- Full Pipeline ---------
# Step 1: Classify
categorical_cols, numeric_cols = classify_columns(df)

# Step 2: Fix binary floats
df = convert_binary_floats_to_categoricals(df, categorical_cols)

# Step 2.5: Replace outliers with NaN
df = replace_outliers_with_nan(df, numeric_cols)

# Step 3: Impute combinations and save
combinations = [
    ("mode", "median"),
    ("mode", "knn"),
    ("knn", "median"),
    ("knn", "knn"),
]

for cat_method, num_method in combinations:
    df_copy = df.copy()
    df_copy = impute_categoricals(df_copy, categorical_cols, method=cat_method, k=3)
    df_copy = impute_numericals(df_copy, numeric_cols, method=num_method, k=3)

    filename = f"cat-{cat_method}_num-{num_method}.csv"
    df_copy.to_csv(os.path.join("data/basic", filename), index=False)
    print(f"✅ Saved: {filename}")
