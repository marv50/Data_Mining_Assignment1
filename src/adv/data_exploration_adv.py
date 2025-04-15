import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to count unique values
def count_values(variable):
    return variable.value_counts()

# Function to count missing values
def count_missing_values(variable):
    total_missing = variable.isnull().sum()
    return total_missing

# Function to count outliers
def count_outliers(variable, method='IQR'):
    if method == 'IQR':
        Q1 = variable.quantile(0.25)
        Q3 = variable.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'STD':
        mean = variable.mean()
        std = variable.std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
    else:
        raise ValueError("Invalid method. Choose 'IQR' or 'STD'.")
    
    outliers = variable[(variable < lower_bound) | (variable > upper_bound)]
    return len(outliers)

# Function to summarize variables
def summarize_variables(data):
    summary_data = []
    for var_name, values in data.items():
        summary_data.append({
            "Variable": var_name,
            "Num Entries": len(values),
            "Num Missing Values": count_missing_values(values),
            "Num Outliers (IQR)": count_outliers(values, method='IQR')
        })
    return pd.DataFrame(summary_data)

# Function to create a clean summary table visualization
def create_summary_table_visualization(summary_table, filename='summary_table.png'):
    # Create a figure with appropriate size
    plt.figure(figsize=(12, len(summary_table) * 0.5 + 2))
    
    # Create a table plot
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Create the table with the data
    table = plt.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(summary_table.columns),
        cellColours=[['#e6f7ff'] * len(summary_table.columns)] * len(summary_table)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add a title
    plt.title("Summary Statistics for Variables", pad=20, fontsize=16)
    
    # Save the table
    plt.savefig("fig/" + filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

# Function to create a dashboard-style visualization
def create_summary_dashboard(summary_table, filename='summary_dashboard.png'):
    # Set the style
    plt.style.use('ggplot')
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Variable Summary Dashboard', fontsize=20, y=0.98)
    
    # Sort variables by number of entries
    summary_by_entries = summary_table.sort_values('Num Entries', ascending=False)
    
    # Plot 1: Number of entries by variable
    axes[0].bar(summary_by_entries['Variable'], summary_by_entries['Num Entries'], color='steelblue')
    axes[0].set_title('Number of Entries by Variable')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=90)
    
    # Plot 2: Missing values by variable
    axes[1].bar(summary_by_entries['Variable'], summary_by_entries['Num Missing Values'], color='tomato')
    axes[1].set_title('Missing Values by Variable')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=90)
    
    # Plot 3: Outliers by variable
    axes[2].bar(summary_by_entries['Variable'], summary_by_entries['Num Outliers (IQR)'], color='mediumseagreen')
    axes[2].set_title('Outliers (IQR Method) by Variable')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=90)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the dashboard
    plt.savefig("fig/" + filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

# Main execution code
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('data/adv/dataset_mood_smartphone.csv')
    
    # List of variables to extract
    variables = [
        "mood", "circumplex.arousal", "circumplex.valence", "activity", "screen", "call", "sms",
        "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game",
        "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown",
        "appCat.utilities", "appCat.weather"
    ]
    
    # Extract data for each variable
    data = {}
    for var in variables:
        values = df.loc[df['variable'] == var, 'value']
        if not values.empty:  # Only process non-empty series
            values.name = var  # Set name for the series
            data[var] = values
    
    # Generate summary table
    summary_table = summarize_variables(data)
    print(summary_table)
    
    # Create and save the visualizations
    create_summary_table_visualization(summary_table)
    create_summary_dashboard(summary_table)