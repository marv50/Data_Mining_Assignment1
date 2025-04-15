import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def plot_distributions(dataframe, file_name=None):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    dataframe[numeric_cols].hist(bins=30, figsize=(15, 10))

    plt.suptitle("Distributions of Numerical Features")
    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', file_name)
        plt.savefig(file_path, dpi=300)
    plt.show()

def plot_correlation_heatmap(dataframe, file_name=None):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    corr = dataframe[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', file_name)
        plt.savefig(file_path, dpi=300)
    plt.show()


def create_summary_table_visualization(dataframe, file_name=None):
    fig, ax = plt.subplots(figsize=(20, len(dataframe) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    table_data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))

    plt.title("Summary Statistics for Variables", pad=20, fontsize=16)
    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', file_name)
        plt.savefig(file_path, dpi=300)

    plt.show()
