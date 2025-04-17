# Script for standardizing all numerical datain the dataset to mean 0 and standard deviation 1
# This is done using the StandardScaler from sklearn
# The script takes a CSV file as input, standardizes the numerical columns, and saves the result to a new CSV file
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

def standardize_csv(input_file, output_file):
    """
    Standardizes numerical columns in a CSV file to mean 0 and standard deviation 1.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the standardized CSV file.
    """
    # Load the dataset
    data = pd.read_csv(input_file)
    
    # Identify numerical columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Standardize the numerical columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # Save the standardized dataset to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Standardized dataset saved to {output_file}")

if __name__ == "__main__":
    input_csv = 'data/basic/ODI-2025 - adjusted2.csv'
    output_csv = "data/basic/ODI-2025 - adjusted3.csv"  
    standardize_csv(input_csv, output_csv)
