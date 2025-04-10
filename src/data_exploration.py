import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the figure output directory exists
os.makedirs("fig", exist_ok=True)

# Function to count unique values
def count_values(column):
    return column.value_counts()

# Function to count missing values
def count_missing_values(column):
    return column.isnull().sum()

# Function to count outliers
def count_outliers(column, method='IQR'):
    if not pd.api.types.is_numeric_dtype(column):
        return 0  # Categorical data doesn't have outliers

    if method == 'IQR':
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'STD':
        mean = column.mean()
        std = column.std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
    else:
        raise ValueError("Invalid method. Choose 'IQR' or 'STD'.")

    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)

# Function to summarize variables
def summarize_variables(data):
    summary_data = []
    for col_name in data.columns:
        col = data[col_name]
        summary_data.append({
            "Variable": col_name,
            "Num Entries": col.shape[0],
            "Num Missing Values": count_missing_values(col),
            "Num Outliers (IQR)": count_outliers(col, method='IQR')
        })
    return pd.DataFrame(summary_data)

# Function to create a clean summary table visualization
def create_summary_table_visualization(summary_table, filename='summary_table.png'):
    plt.figure(figsize=(12, len(summary_table) * 0.5 + 2))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = plt.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(summary_table.columns),
        cellColours=[['#e6f7ff'] * len(summary_table.columns)] * len(summary_table)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Summary Statistics for Variables", pad=20, fontsize=16)
    plt.savefig("fig/" + filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename

# Function to create a dashboard-style visualization
def create_summary_dashboard(summary_table, filename='summary_dashboard.png'):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Variable Summary Dashboard', fontsize=20, y=0.98)

    summary_sorted = summary_table.sort_values('Num Entries', ascending=False)

    axes[0].bar(summary_sorted['Variable'], summary_sorted['Num Entries'], color='steelblue')
    axes[0].set_title('Number of Entries by Variable')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=90)

    axes[1].bar(summary_sorted['Variable'], summary_sorted['Num Missing Values'], color='tomato')
    axes[1].set_title('Missing Values by Variable')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=90)

    axes[2].bar(summary_sorted['Variable'], summary_sorted['Num Outliers (IQR)'], color='mediumseagreen')
    axes[2].set_title('Outliers (IQR Method) by Variable')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("fig/" + filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename

# Function to plot histogram of a column
def plot_histogram(column, filename='histogram.png'):
    plt.figure(figsize=(10, 6))
    sns.histplot(column.dropna(), bins=30)
    plt.title('Histogram of ' + column.name)#
    plt.xlabel(column.name)
    plt.ylabel('Frequency')
    plt.savefig("fig/" + filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename

# Function to plot boxplot of a column
def plot_boxplot(column, filename='boxplot.png'):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=column.dropna())
    plt.title('Boxplot of ' + column.name)
    plt.xlabel(column.name)
    plt.savefig("fig/" + filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename

# Main execution code
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('data/basic/ODI-2025 - adjusted.csv')

    df = df.drop(df.index[-1])

    age = df['sports_hours']
    plot_histogram(age, filename='age_histogram.png')

   
    
