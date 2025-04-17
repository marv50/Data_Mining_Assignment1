import pandas as pd
import numpy as np
from interpolation import classify_columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load one of the processed datasets
df = pd.read_csv("data/basic/cat-mode_num-median.csv")  # Change to whichever combination you prefer

# Split features and target
X = df.drop(columns=['ml_course'])
y = df['ml_course']

# Identify categorical and numerical columns 
categorical_cols, numeric_cols = classify_columns(X)

# Encode target variable (if it's not already)
y = y.astype('category')

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- KNN Classifier Pipeline ---
knn_preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

knn_pipeline = Pipeline([
    ("preprocess", knn_preprocessor),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Train and evaluate KNN
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
print("📊 KNN Classifier Report:")
print(classification_report(y_test, y_pred_knn))

# --- Logistic Regression Pipeline ---
logreg_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

logreg_pipeline = Pipeline([
    ("preprocess", logreg_preprocessor),
    ("logreg", LogisticRegression(multi_class='multinomial', max_iter=1000))
])

# Train and evaluate Logistic Regression
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)
print("📊 Logistic Regression Report:")
print(classification_report(y_test, y_pred_logreg))
