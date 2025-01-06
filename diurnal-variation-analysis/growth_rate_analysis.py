import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Calculate growth rate for a given stage
def calculate_growth_rate(data, cultivar_samples):
    df_samples = data[cultivar_samples]
    df_mean = df_samples.mean(axis=1).to_frame(name='height')
    df_mean.dropna(subset=['height'], inplace=True)

    # Resample data to 10-minute intervals using mean
    df_resampled = df_mean.resample('10min').mean()

    # Calculate growth rate
    df_resampled['height_diff'] = df_resampled['height'].diff()
    df_resampled['time_diff'] = df_resampled.index.to_series().diff().dt.total_seconds() / 3600  # in hours
    df_resampled['growth_rate'] = df_resampled['height_diff'] / df_resampled['time_diff']

    # Remove NaN and infinite values
    df_resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_resampled.dropna(subset=['growth_rate'], inplace=True)

    # Remove outliers using the IQR method
    if len(df_resampled['growth_rate']) > 2:
        Q1 = df_resampled['growth_rate'].quantile(0.25)
        Q3 = df_resampled['growth_rate'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_resampled = df_resampled[(df_resampled['growth_rate'] >= lower_bound) & (df_resampled['growth_rate'] <= upper_bound)]

    # Set Negative Growth Rates to Zero
    df_resampled['growth_rate'] = df_resampled['growth_rate'].apply(lambda x: x if x > 0 else 0)

    return df_resampled

# Smooth growth rate using Savitzky-Golay filter
def smooth_growth_rate(hourly_stats):
    window_length = min(7, len(hourly_stats))
    if window_length % 2 == 0:  # Ensure window_length is odd
        window_length -= 1
    if window_length >= 3:
        hourly_stats['mean_smooth'] = savgol_filter(hourly_stats['mean'], window_length=window_length, polyorder=2)
    else:
        hourly_stats['mean_smooth'] = hourly_stats['mean']
    return hourly_stats
