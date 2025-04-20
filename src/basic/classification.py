import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.basic.interpolation import classify_columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Utility Functions ---

def create_pipeline(categorical_cols, numeric_cols, model_type="logreg", lasso=False):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])
    
    if model_type == "logreg":
        penalty = 'l1' if lasso else 'l2'
        model = LogisticRegression(
            multi_class='auto', max_iter=1000, solver='saga',
            penalty=penalty, warm_start=True
        )
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipeline

def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # For logistic regression, return coefficients
    model = pipeline.named_steps['model']
    coef = model.coef_ if hasattr(model, "coef_") else None

    return acc, coef

def run_experiment(dataset_name, selected_columns, lasso=False, n_runs=50, model_type="logreg"):
    filepath = f"data/basic/basic_{dataset_name}.csv"
    df = pd.read_csv(filepath)

    y = df['ml_course'].astype('category')
    if len(y.cat.categories) != 2:
        raise ValueError("Target variable must have exactly 2 classes for binomial logistic regression.")

    X = df[selected_columns]
        
    categorical_cols, numeric_cols = classify_columns(X)

    accuracies = []

    for random_state in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )

        pipeline = create_pipeline(categorical_cols, numeric_cols, model_type=model_type, lasso=lasso)
        acc, coef = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    
    return np.mean(accuracies), np.std(accuracies)


def run_single_experiment(dataset_name, selected_columns):
    filepath = f"data/basic/basic_{dataset_name}.csv"
    df = pd.read_csv(filepath)

    y = df['ml_course'].astype('category')
    if len(y.cat.categories) != 2:
        raise ValueError("Target variable must have exactly 2 classes for binomial logistic regression.")

    X = df[selected_columns]

    categorical_cols, numeric_cols = classify_columns(X)

    # Split once
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Create and train Lasso (L1) logistic regression pipeline
    lasso_pipeline = create_pipeline(categorical_cols, numeric_cols, model_type="logreg", lasso=True)
    lasso_pipeline.fit(X_train, y_train)

    model = lasso_pipeline.named_steps['model']
    coef = model.coef_[0]

    preprocessor = lasso_pipeline.named_steps['preprocess']
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)

    # Full list of feature names (after transformation)
    feature_names = cat_feature_names.tolist() + numeric_cols

    # Summarize coefficients per original feature
    summarized_coefs = {}

    idx = 0
    for feature in categorical_cols:
        matching_features = [i for i, name in enumerate(cat_feature_names) if name.startswith(feature + "_")]
        matching_coefs = coef[matching_features]
        summarized_coefs[feature] = np.max(np.abs(matching_coefs))  # max absolute value
        idx += len(matching_features)

    for feature in numeric_cols:
        summarized_coefs[feature] = abs(coef[idx])
        idx += 1

    print(f"\nSummarized Coefficients for {dataset_name} (Lasso):")
    for fname, c in summarized_coefs.items():
        print(f"{fname}: {c:.4f}")

    return summarized_coefs
