import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.interpolation import classify_columns  # Assuming you have this helper function

# --- Utility Functions ---

def create_linreg_pipeline(categorical_cols, numeric_cols):
    # Set up preprocessor with OneHotEncoding for categorical columns and StandardScaler for numerical ones
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])
    
    # Linear regression model
    model = LinearRegression()
    
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("linreg", model)
    ])
    return pipeline

def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    # Fit the pipeline and evaluate on the test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae

def run_experiment(dataset_name, selected_columns, n_runs=50):
    filepath = f"data/basic/basic_{dataset_name}.csv"
    df = pd.read_csv(filepath)

    # X features (including ml_course) and y (stress_level as target)
    X = df[selected_columns]
    y = df['stress_level']  # Target variable

    # Classify the columns into categorical and numerical ones
    categorical_cols, numeric_cols = classify_columns(X)

    mses = []
    maes = []

    for random_state in range(n_runs):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # Create and evaluate the linear regression pipeline
        linreg_pipeline = create_linreg_pipeline(categorical_cols, numeric_cols)
        mse, mae = evaluate_model(linreg_pipeline, X_train, X_test, y_train, y_test)

        mses.append(mse)
        maes.append(mae)

    mses = np.array(mses)
    maes = np.array(maes)
    
    return mses, maes

def plot_results(mses, maes, dataset_name):
    # Plot the MSE and MAE results
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].hist(mses, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title(f'MSE Distribution for {dataset_name}')
    axs[0].set_xlabel('MSE')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(maes, bins=20, color='salmon', edgecolor='black')
    axs[1].set_title(f'MAE Distribution for {dataset_name}')
    axs[1].set_xlabel('MAE')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()



