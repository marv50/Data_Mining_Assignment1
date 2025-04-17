import os

from src.basic.csv_utils_basic import *
from src.basic.data_exploration_basic import *
from src.basic.plotting_basic import *
from src.basic.data_cleaning_basic import *
from src.basic.feature_engineering_basic import *


def analyze_data(data_frame, name_extension, iqr_thresh=1.5):
    explore_dataset(data_frame)

    summary_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'basic', 'summary' + name_extension + '.csv')
    summary = summarize_dataframe(data_frame, iqr_thresh=iqr_thresh)
    summary.to_csv(summary_path, index=False)
    create_summary_table_visualization(summary, file_name='summary' + name_extension + '.png')

    plot_distributions(data_frame, file_name="distributions" + name_extension + ".png")
    plot_correlation_heatmap(data_frame, file_name="correlation_heatmap" + name_extension + ".png")

