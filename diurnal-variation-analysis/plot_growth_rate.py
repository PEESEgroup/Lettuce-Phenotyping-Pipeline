import matplotlib.pyplot as plt
import pandas as pd

# Plot growth rate by stage
def plot_growth_rate_by_stage(data, cultivar_samples, cultivar_name, save_path_prefix, color):
    stages = ['Seedling', 'Cupping', 'Rosette', 'Heading']

    for stage in stages:
        data_stage = data[data['stage'] == stage]
        if data_stage.empty:
            print(f"No data available for {cultivar_name} during the {stage} stage.")
            continue

        # Calculate growth rate for the stage
        df_resampled = calculate_growth_rate(data_stage, cultivar_samples)
        if df_resampled.empty:
            print(f"No growth rate data available for {cultivar_name} during the {stage} stage.")
            continue

        # Extract hour of day
        df_resampled['hour'] = df_resampled.index.hour

        # Calculate hourly statistics
        hourly_stats = df_resampled.groupby('hour')['growth_rate'].agg(['mean', 'std'])
        if hourly_stats.empty:
            print(f"No hourly growth rate data available for {cultivar_name} during the {stage} stage.")
            continue

        # Smooth the mean growth rate
        hourly_stats = smooth_growth_rate(hourly_stats)
        lower_bound = hourly_stats['mean_smooth'] - hourly_stats['std'].clip(lower=0)
        upper_bound = hourly_stats['mean_smooth'] + hourly_stats['std']

        # Plot growth rate
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(hourly_stats.index, hourly_stats['mean_smooth'], color=color, linewidth=2, marker='o', label='Growth Rate')
        ax.fill_between(hourly_stats.index, lower_bound, upper_bound, alpha=0.3, color=color)
        ax.set_title(f'{cultivar_name} Lettuce - {stage} Stage', fontsize=19, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=17, fontweight='bold')
        ax.set_ylabel('Height (cm/hour)', fontsize=17, fontweight='bold')
        ax.set_xticks(range(0, 24))
        ax.set_xlim(0, 23)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.legend(loc='best', fontsize=15)
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{save_path_prefix}_{cultivar_name}_{stage}_stage.png', format='png', dpi=600)
        plt.show()
        plt.close(fig)
