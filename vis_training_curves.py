import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_align_data(folder_path, file_pattern):
    """
    Load all CSV files matching a pattern and align them based on epoch numbers.
    If files have fewer lines (early stopping), the remaining values are padded 
    with the last recorded value.
    """
    all_data = []

    # Collect all matching CSV files
    max_length = 0  # Track the longest file length
    for file_name in os.listdir(folder_path):
        if file_name.startswith(file_pattern) and file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, header=None)[0]
            max_length = max(max_length, len(data))  # Find the longest file

            all_data.append(data)

    # Align data by padding shorter files with their last value
    padded_data = [
        data.reindex(range(max_length), method='ffill') for data in all_data
    ]
    
    # Combine all data into a single DataFrame (columns = different files)
    aligned_data = pd.DataFrame(padded_data).T
    return aligned_data

def plot_means_and_medians(datasets, names, zoom_window=0.0):
    """
    Plot mean and median curves for multiple datasets, print their final values,
    and generate two plots: normal and zoomed-in along the y-axis.
    
    Args:
    - datasets: List of DataFrames containing aligned data sequences.
    - names: List of names corresponding to each dataset.
    """
    final_values = []  # Collect final values to determine zoomed plot range

    # Create the first figure (normal plot)
    plt.figure(figsize=(12, 6))

    # Loop through each dataset and plot mean and median curves
    for data, name in zip(datasets, names):
        mean_curve = data.mean(axis=1)
        median_curve = data.median(axis=1)

        # Plot both curves
        plt.plot(mean_curve, label=f'{name} (Mean)')
        plt.plot(median_curve, label=f'{name} (Median)')

        # Print final values and store them
        final_mean = mean_curve.iloc[-1]
        final_median = median_curve.iloc[-1]
        final_values.extend([final_mean, final_median])

        print(f"Final Mean Value for {name}: {final_mean}")
        print(f"Final Median Value for {name}: {final_median}")

    # Configure the first plot (normal view)
    plt.xlabel('Epoch')
    plt.ylabel('Bray-Curtis Dissimilarity')
    plt.title('cNODE.jl-Ocean Loss Curves')
    plt.legend()

    if zoom_window >= 0.0:
        plt.show(block=False)

        # Create the second figure (zoomed-in plot)
        max_final_value = np.mean(final_values)  # Find the maximum final value
        y_max = zoom_window * max_final_value  # Set y-axis upper limit

        plt.figure(figsize=(12, 6))

        # Re-plot all curves for the zoomed-in view
        for data, name in zip(datasets, names):
            mean_curve = data.mean(axis=1)
            median_curve = data.median(axis=1)
            
            plt.plot(mean_curve, label=f'{name} (Mean)')
            plt.plot(median_curve, label=f'{name} (Median)')

        # Configure the second plot (zoomed-in view)
        plt.ylim(0, y_max)
        plt.xlabel('Epoch')
        plt.ylabel('Bray-Curtis Dissimilarity')
        plt.title('cNODE.jl-Ocean Loss Curves (zoomed)')
        plt.legend()
    
    plt.show()


def main(folder_path):
    # Load and align datasets (modify as needed to load more datasets)
    train_data = load_and_align_data(folder_path, 'train')
    test_data = load_and_align_data(folder_path, 'test')

    # Call the plotting function with datasets and their names
    datasets = [train_data, test_data]
    names = ['Train', 'Test']

    plot_means_and_medians(datasets, names, zoom_window=2.0)

if __name__ == "__main__":
    main('./results/real/Ocean/loss_epochs/')
