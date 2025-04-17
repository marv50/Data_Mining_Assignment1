import pandas as pd
import numpy as np
from interpolation import classify_columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# List of dataset filenames (without ".csv")
datasets = [
    "mode_knn",
    "mode_median",
    "knn_median",
    "knn_knn", 
]

# Initialize result storage
knn_results = []
logreg_results = []

for name in datasets:
    # Load dataset
    filepath = f"data/basic/basic_{name}.csv"
    df = pd.read_csv(filepath)
    
    selected_columns = ["ml_course", "sports_hours", "stress_level", "used_chatgpt", "ir_course"]

    # Split features and target
    X = df[selected_columns]
    y = df['program']
    
    # Identify categorical and numerical columns
    categorical_cols, numeric_cols = classify_columns(X)
    
    # Encode target variable (if necessary)
    y = y.astype('category')
    
    # Train/test split
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

    # Save KNN results
    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_prec, knn_rec, knn_f1, _ = precision_recall_fscore_support(y_test, y_pred_knn, average='weighted', zero_division=0)
    knn_results.append([name, knn_acc, knn_prec, knn_rec, knn_f1])
    
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

    # Save Logistic Regression results
    logreg_acc = accuracy_score(y_test, y_pred_logreg)
    logreg_prec, logreg_rec, logreg_f1, _ = precision_recall_fscore_support(y_test, y_pred_logreg, average='weighted', zero_division=0)
    logreg_results.append([name, logreg_acc, logreg_prec, logreg_rec, logreg_f1])

# --- Create result tables ---
knn_df = pd.DataFrame(knn_results, columns=["Dataset", "Accuracy", "Precision", "Recall", "F1 Score"])
logreg_df = pd.DataFrame(logreg_results, columns=["Dataset", "Accuracy", "Precision", "Recall", "F1 Score"])

# Display tables
print("\n📋 KNN Classifier Results:")
print(knn_df)

print("\n📋 Logistic Regression Results:")
print(logreg_df)
