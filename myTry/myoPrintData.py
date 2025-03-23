import csv
import matplotlib.pyplot as plt

def plot_emg_data(filename):
    # Read the CSV file
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the header row
        emg_data = list(reader)

    # Convert the data to a list of lists (each sublist is a time step)
    emg_data = [[float(value) for value in row] for row in emg_data]

    # Transpose the data to separate each EMG channel
    emg_channels = list(zip(*emg_data))

    # Create a figure with 8 subplots
    fig, axs = plt.subplots(8, 1, figsize=(10, 12))
    fig.suptitle(f"EMG Data from {filename}", fontsize=16)

    # Plot each EMG channel in a separate subplot
    for i in range(8):
        axs[i].plot(emg_channels[i], label=f"EMG {i+1}")
        axs[i].set_ylabel(f"EMG {i+1}")
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    # Set common x-axis label
    axs[-1].set_xlabel("Time Steps")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Example usage
filename = "gesture_1_data.csv"  # Replace with your CSV file name
plot_emg_data(filename)