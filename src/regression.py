import pandas as pd
import numpy as np
from interpolation import classify_columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("data/basic/cat-mode_num-median.csv")

# Split features and target
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# Identify categorical and numerical columns
categorical_cols, numeric_cols = classify_columns(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Linear Regression Pipeline ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

regression_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# Train and evaluate
regression_pipeline.fit(X_train, y_train)
y_pred = regression_pipeline.predict(X_test)

# --- Evaluation ---
print("📉 Linear Regression Results:")
print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
