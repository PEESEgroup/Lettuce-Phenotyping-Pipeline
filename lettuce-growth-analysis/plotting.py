matplotlib.pyplot as plt
import numpy as np

# Set global styling
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

def process_and_plot(ax, df, columns_group, group_name, cmap_name='tab20'):
    cmap = plt.get_cmap(cmap_name, len(columns_group))
    colors = [cmap(i) for i in range(len(columns_group))]
    all_means = []

    for idx, sample_column in enumerate(columns_group):
        color = colors[idx]

        lower_bound = df[sample_column].quantile(0.01)
        upper_bound = df[sample_column].quantile(0.99)
        filtered_df = df[(df[sample_column] >= lower_bound) & (df[sample_column] <= upper_bound)]

        daily_mean = filtered_df[sample_column].resample('D').mean()
        daily_max = filtered_df[sample_column].resample('D').max()

        rolling_mean = daily_mean.rolling(window=7, min_periods=1).mean()
        rolling_max = daily_max.rolling(window=7, min_periods=1).mean()
        rolling_max_clipped = np.minimum(rolling_max, rolling_mean * 1.5)

        error_up = rolling_max_clipped - rolling_mean
        all_means.extend(rolling_mean.values)

        day_number = np.arange(1, len(rolling_mean) + 1)

        ax.plot(day_number, rolling_mean.values, label=f'{sample_column} Mean', color=color, linewidth=2)
        ax.errorbar(
            day_number,
            rolling_mean.values,
            yerr=[np.zeros_like(error_up.values), error_up.values],
            fmt='o',
            ecolor=color,
            elinewidth=2,
            capsize=4,
            alpha=1,
            color=color
        )

    min_mean = np.min(all_means)
    max_mean = np.max(all_means)
    y_range = max_mean - min_mean
    padding = y_range * 0.1
    ax.set_ylim(min_mean - padding, max_mean + padding)

    ax.set_xlabel('Day', fontsize=17, fontweight='bold')
    ax.set_ylabel('Value', fontsize=17, fontweight='bold')
    ax.set_title(f'{group_name}', fontsize=19, fontweight='bold')
    ax.tick_params(axis='x', which='major', labelsize=15, rotation=45, pad=10)
    ax.tick_params(axis='y', which='major', labelsize=15)
    return ax

def create_figure(df, columns_group, group_name, cmap_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    process_and_plot(ax, df, columns_group, group_name, cmap_name)
    plt.tight_layout()
    fig.savefig(save_path, format='png', dpi=600)
    plt.show()
