from scripts.data_preparation import load_and_prepare_data
from scripts.plotting import create_figure

# File paths
height_file_path = 'data/height.csv'
area_file_path = 'data/area.csv'

# Load and prepare height data
height_df, height_columns = load_and_prepare_data(height_file_path, 'timestamp', 'height_')
height_group1 = height_columns[:20]
height_group2 = height_columns[20:40]

# Plot height growth
create_figure(height_df, height_group1, 'REX Height Growth', 'tab20', 'rex_height_growth.png')
create_figure(height_df, height_group2, 'ROUXAI Height Growth', 'tab20b', 'rouxai_height_growth.png')

# Load and prepare area data
area_df, area_columns = load_and_prepare_data(area_file_path, 'timestamp', 'area_')
area_group1 = area_columns[:19]
area_group2 = area_columns[19:]

# Plot area growth
create_figure(area_df, area_group1, 'REX Area Growth', 'tab20', 'rex_area_growth.png')
create_figure(area_df, area_group2, 'ROUXAI Area Growth', 'tab20b', 'rouxai_area_growth.png')
