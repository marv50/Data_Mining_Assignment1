import pandas as pd

def read_csv(file_path):
    try:
        print(file_path)
        data = pd.read_csv(file_path)

        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data['birthday'] = pd.to_datetime(data['birthday'], errors='coerce')
        data['bedtime'] = pd.to_datetime(data['bedtime'], format="%H:%M:%S", errors='coerce').dt.time
        data['est_age_days'] = pd.to_numeric(data['est_age_days'], errors='coerce')

        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def dataframe_to_csv(dataframe, file_path):
    try:
        dataframe.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")