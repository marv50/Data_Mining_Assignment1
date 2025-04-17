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
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', 'collective_distr', file_name)
        plt.savefig(file_path, dpi=300)
        
    plt.show()

def plot_single_distribution(dataframe, column_name, title=None, xlabel=None, ylabel=None, file_name=None):
    plt.figure(figsize=(6, 4))
    sns.histplot(dataframe[column_name], bins=30, kde=True, alpha=0.6)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', 'distributions', file_name)
        plt.savefig(file_path, dpi=300)

    #plt.show()

def plot_correlation_heatmap(dataframe, file_name=None):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if 'bedtime_sin' in numeric_cols:
        numeric_cols = numeric_cols.drop('bedtime_sin')
    if 'bedtime_cos' in numeric_cols:
        numeric_cols = numeric_cols.drop('bedtime_cos')
    if 'bedtime_anglenorm' in numeric_cols:
        numeric_cols = numeric_cols.drop('bedtime_anglenorm')
    
    corr = dataframe[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    #plt.title("Correlation Matrix")

    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', 'correlations', file_name)
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

    #plt.title("Summary Statistics for Variables", pad=20, fontsize=16)
    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', 'summaries', file_name)
        plt.savefig(file_path, dpi=300)

    plt.show()


def plot_positional_times(dataframe, file_name=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(dataframe['bedtime_sin'], dataframe['bedtime_cos'], alpha=0.6)
    #plt.title('Positional Bedtime')
    plt.ylabel('cos(angle)')
    plt.xlabel('sin(angle)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)

    plt.text(0, 1.1, '0/24', ha='center', va='center', fontsize=10)  # Top
    plt.text(1.1, 0, '6', ha='center', va='center', fontsize=10)     # Right
    plt.text(0, -1.1, '12', ha='center', va='center', fontsize=10)   # Bottom
    plt.text(-1.1, 0, '18', ha='center', va='center', fontsize=10)   # Left

    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", 'fig', 'bedtime_distr', file_name)
        plt.savefig(file_path, dpi=300)

    plt.show()
