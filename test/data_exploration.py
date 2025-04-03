import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/adv/dataset_mood_smartphone.csv')


def count_missing_values(variable):
    """
    Count the number of missing values in each column of the DataFrame.
    """
    missing_values = variable.isnull().sum()
    print("Missing values in each column:")
    print(missing_values[missing_values > 0])
    return missing_values[missing_values > 0]


def count_outliers(variable, method='IQR'):
    """
    Count the number of outliers in the given variable using either the IQR method or the STD method (2 standard deviations).

    Parameters:
    - variable: The data column to analyze.
    - method: The method to use for outlier detection ('IQR' or 'STD').

    Returns:
    - The number of outliers.
    """
    if method == 'IQR':
        # IQR method
        Q1 = variable.quantile(0.25)
        Q3 = variable.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'STD':
        # Standard deviation method
        mean = variable.mean()
        std = variable.std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
    else:
        raise ValueError("Invalid method. Choose 'IQR' or 'STD'.")

    # Identify outliers
    outliers = variable[(variable < lower_bound) | (variable > upper_bound)]
    print(f"Number of outliers using {method} method:", len(outliers))
    return len(outliers)


def generate_distribution(variable):
    """
    Generate a distribution plot for the given variable.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(variable, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of ' + variable.name)
    plt.xlabel(variable.name)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def generate_boxplot(variable):
    """
    Generate a box plot for the given variable.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(variable, vert=False)
    plt.title('Box Plot of ' + variable.name)
    plt.xlabel(variable.name)
    plt.grid(axis='x', alpha=0.75)
    plt.show()


mood = df.loc[df['variable'] == 'mood', 'value']
print(len(mood))
generate_distribution(mood)
count_missing_values(mood)
count_outliers(mood)
