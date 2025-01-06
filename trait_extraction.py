import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.patches import Rectangle

def visualize_height_timeseries(selected_timestamps):
    '''
    Read extracted height data from CSV file and plot the timeseries
    selected_timestamps: list of timestamps to be plotted
    '''

    df = pd.read_csv('Data/output_data_height.csv')
    df['Timeframe'] = pd.to_datetime(df['Timeframe'], format='%Y-%m-%d %H_%M_%S', errors='coerce')


    Ts = pd.to_datetime(selected_timestamps, format='%Y-%m-%d %H:%M:%S')
    df_copy = df.set_index("Timeframe")

    plot_data = []

    for i in range(len(Ts)-1):
        tmp = df_copy[Ts[i]: Ts[i+1]].iloc[:-1]
        tmp = tmp.values
        plot_data.append(tmp)

    indices_to_remove = [1, 5, 9, 11, 13, 16, 17, 19, 23]

    #    Specified indices Removed from daily_data
    daily_data_filtered = [plot_data[i] for i in range(len(plot_data)) if i not in indices_to_remove]

    plot_data = daily_data_filtered

    fig, ax = plt.subplots(figsize=(10, 6))
    means = np.array([np.mean(i, axis=0) for i in plot_data])
    num_samples = plot_data[0].shape[1]

    # Color map for different lettuce samples
    colors = plt.cm.viridis(np.linspace(0, 1, num_samples))

    # Positions of the box plots to avoid overlapping
    box_positions = np.arange(1, len(plot_data) + 1)
    width = 0.1  # Width of the box plots
    spacing = 0.1  # Spacing between box plots of different lettuce_idx

    for lettuce_idx in range(num_samples):
        # Smoothed values
        smoothed = savgol_filter(means[:, lettuce_idx], window_length=9, polyorder=2)  # Adjust window_length as needed

        # Adjusted positions for this lettuce_idx
        adjusted_positions = [pos + (lettuce_idx - num_samples / 2) * (width + spacing) for pos in box_positions]

        # Plot
        ax.plot(adjusted_positions, smoothed, color=colors[lettuce_idx], linewidth=2, label=f'Lettuce Sample {lettuce_idx + 1}')
        ax.scatter(adjusted_positions, smoothed, color=colors[lettuce_idx])

        # Box plot for each lettuce_idx
        box_data = [plot_data[i][:, lettuce_idx] for i in range(len(plot_data))]
        box_plot = ax.boxplot(box_data, positions=adjusted_positions, widths=width, notch=True, patch_artist=True)

        # Style box plots
        for box in box_plot['boxes']:
            box.set_facecolor(colors[lettuce_idx])
            box.set_edgecolor('darkgreen')
            box.set_alpha(0.6)
            
    day_mapping = {
        1: "Day 1",
        3: "Day 2",
        6: "Day 3",
        8: "Day 4",
        9: "Day 5",
        10.5: "Day 6",
        12: "Day 10",
        14: "Day 11",
        16: "Day 12"
    }

    # Set labels and legend
    ax.set_ylabel('Height (cm)')
    ax.legend(loc='upper left', fontsize='small')
    ax.set_xticks(list(day_mapping.keys()))
    ax.set_xticklabels(list(day_mapping.values()), rotation=45)

    plt.xticks(range(1, len(plot_data) + 1))
    plt.tight_layout()
    plt.show()


def visualize_area_timeseries(selected_timestamps):
    '''
    Read extracted area and visualize the timeseries
    selected_timestamps: list of timestamps to be plotted
    '''
    df2 = pd.read_csv('Data/output_data_area.csv')
    df2['Timeframe'] = pd.to_datetime(df2['Timeframe'], format='%Y-%m-%d %H_%M_%S', errors='coerce')

    Ts = pd.to_datetime(selected_timestamps, format='%Y-%m-%d %H:%M:%S')
    df_copy2 = df2.set_index("Timeframe")

    plot_data2 = []

    for i in range(len(Ts)-1):
        tmp = df_copy2[Ts[i]: Ts[i+1]].iloc[:-1]
        tmp = tmp.values
        plot_data2.append(tmp)

    indices_to_remove = [1, 5, 9, 11, 13, 16, 17, 19, 23]

    # Specified indices removed from daily_data
    daily_data_filtered2 = [plot_data2[i] for i in range(len(plot_data2)) if i not in indices_to_remove]


    plot_data2 = daily_data_filtered2

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for better visibility

    means = np.array([np.mean(i, axis=0) for i in plot_data2])
    num_samples = means.shape[1]  # Number of lettuce samples

    # Color map for different lettuce samples
    colors = plt.cm.viridis(np.linspace(0, 1, num_samples))

    # Positions for each group of box plots to avoid overlapping
    box_positions = np.arange(1, len(plot_data2) + 1)
    width = 0.1  # Width of each box plot
    spacing = 0.05  # Spacing between box plots of different lettuce_idx

    for lettuce_idx in range(num_samples):
        # Smoothed values for each lettuce_idx
        smoothed = savgol_filter(means[:, lettuce_idx], window_length=9, polyorder=2)

        # Adjusted positions for this lettuce_idx
        adjusted_positions = [pos + (lettuce_idx - num_samples / 2) * (width + spacing) for pos in box_positions]

        # Plotting the smoothed line and scatter for each lettuce sample
        ax.plot(adjusted_positions, smoothed, color=colors[lettuce_idx], linewidth=2, label=f'Lettuce Sample {lettuce_idx + 1}')
        ax.scatter(adjusted_positions, smoothed, color=colors[lettuce_idx])

        # Box plot for each lettuce_idx
        box_data = [plot_data2[i][:, lettuce_idx] for i in range(len(plot_data2))]
        box_plot = ax.boxplot(box_data, positions=adjusted_positions, widths=width, notch=True, patch_artist=True)

        # Style the box plots
        for box in box_plot['boxes']:
            box.set_facecolor(colors[lettuce_idx])
            box.set_edgecolor('darkgreen')
            box.set_alpha(0.6)

    # x-ticks and labels based on the day_mapping
    day_mapping = {
        1: "Day 1",
        3: "Day 2",
        6: "Day 3",
        8: "Day 4",
        9: "Day 5",
        10.5: "Day 6",
        12: "Day 10",
        14: "Day 11",
        16: "Day 12"
    }
    ax.set_xticks(list(day_mapping.keys()))
    ax.set_xticklabels(list(day_mapping.values()), rotation=45)

    ax.set_ylabel('Area (cm²)')
    ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()