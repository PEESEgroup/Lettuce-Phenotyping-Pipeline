from data_preprocessing import load_and_preprocess_data, define_sample_columns
from plot_growth_rate import plot_growth_rate_by_stage

# Load and preprocess data
file_path = 'data/height.csv'
data = load_and_preprocess_data(file_path)
rex_samples, rouxai_samples = define_sample_columns()

# Plot growth rate for REX
plot_growth_rate_by_stage(data, rex_samples, 'REX', 'height_growth_rate_plot', color='green')

# Plot growth rate for ROUXAI
plot_growth_rate_by_stage(data, rouxai_samples, 'ROUXAI', 'height_growth_rate_plot', color='blue')
