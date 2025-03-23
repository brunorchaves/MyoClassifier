import multiprocessing
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, iirnotch  # For filtering and Hilbert transform
from pyomyo import Myo, emg_mode
import os  # To check if the CSV file exists

# Constants
WINDOW_SIZE = 100  # 100 ms window
OVERLAP = 50       # 50% overlap
SAMPLE_RATE = 200  # Assuming 200 Hz sample rate (adjust based on your data)

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

def worker(q):
    """
    Worker function to read EMG data from the Myo armband.
    """
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    
    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)
    
    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Orange logo and bar LEDs
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)
    
    """worker function"""
    while True:
        m.run()
    print("Worker Stopped")

def normalize_emg_data(emg_data):
    """
    Normalize raw EMG data to the range [-1, 1].
    
    Args:
        emg_data (list of lists): Raw EMG data, where each sublist contains 8 EMG values.
    
    Returns:
        list of lists: Normalized EMG data.
    """
    normalized_data = []
    for emg_sample in emg_data:
        normalized_sample = [value / 128.0 for value in emg_sample]
        normalized_data.append(normalized_sample)
    return normalized_data

# Function to design a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling frequency.
        order (int): Filter order.
    
    Returns:
        b, a (numpy arrays): Numerator and denominator polynomials of the filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure the cutoff frequencies are within the valid range (0, 1)
    if low <= 0 or high >= 1:
        raise ValueError("Cutoff frequencies must be within the range (0, Nyquist).")
    
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to the data.
    
    Args:
        data (numpy array): Input data to filter.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling frequency.
        order (int): Filter order.
    
    Returns:
        y (numpy array): Filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Function to design a notch filter
def notch_filter(f0, Q, fs):
    """
    Design a notch filter to remove a specific frequency.
    
    Args:
        f0 (float): Frequency to remove (e.g., 50 Hz for power line interference).
        Q (float): Quality factor.
        fs (float): Sampling frequency.
    
    Returns:
        b, a (numpy arrays): Numerator and denominator polynomials of the filter.
    """
    nyquist = 0.5 * fs
    freq = f0 / nyquist
    b, a = iirnotch(freq, Q)
    return b, a

# Function to apply a notch filter
def apply_notch_filter(data, f0, Q, fs):
    """
    Apply a notch filter to the data.
    
    Args:
        data (numpy array): Input data to filter.
        f0 (float): Frequency to remove.
        Q (float): Quality factor.
        fs (float): Sampling frequency.
    
    Returns:
        y (numpy array): Filtered data.
    """
    b, a = notch_filter(f0, Q, fs)
    y = filtfilt(b, a, data)
    return y

# Function to filter the normalized EMG data
def filter_emg_data(normalized_emg_data, fs=200):
    """
    Apply a 50 Hz notch filter and a 10-100 Hz bandpass filter to the normalized EMG data.
    
    Args:
        normalized_emg_data (list of lists): Normalized EMG data.
        fs (float): Sampling frequency.
    
    Returns:
        filtered_emg_data (list of lists): Filtered EMG data.
    """
    filtered_emg_data = []
    for channel in range(8):  # Loop through all 8 channels
        # Extract data for the current channel
        channel_data = np.array([sample[channel] for sample in normalized_emg_data])
        
        # Apply a 50 Hz notch filter
        notch_filtered_data = apply_notch_filter(channel_data, f0=50, Q=30, fs=fs)
        
        # Apply a 10-100 Hz bandpass filter
        bandpass_filtered_data = bandpass_filter(notch_filtered_data, lowcut=10, highcut=99, fs=fs)
        
        # Append the filtered data to the list
        filtered_emg_data.append(bandpass_filtered_data)
    
    # Transpose the list to match the original format (samples x channels)
    filtered_emg_data = np.array(filtered_emg_data).T.tolist()
    return filtered_emg_data

def calculate_smooth_envelope(emg_signal, window_size=10):
    """
    Calculate a smooth envelope of the EMG signal using the Hilbert transform and a moving average filter.
    Args:
        emg_signal (list or numpy array): Raw EMG signal.
        window_size (int): Size of the moving average window.
    Returns:
        numpy array: Smoothed envelope of the EMG signal.
    """
    # Step 1: Compute the envelope using the Hilbert transform
    analytic_signal = hilbert(emg_signal)  # Apply Hilbert transform
    envelope = np.abs(analytic_signal)     # Calculate the envelope

    # Step 2: Apply a moving average filter for additional smoothing
    smoothed_envelope = np.convolve(envelope, np.ones(window_size) / window_size, mode='same')
    return smoothed_envelope

def detect_gesture_intervals(envelope, threshold_multiplier=1.5):
    """
    Detect the start and end of gesture intervals based on the envelope and an adaptive threshold.
    
    Args:
        envelope (numpy array): Smoothed envelope of the EMG signal.
        threshold_multiplier (float): Multiplier for the mean envelope value to calculate the threshold.
    
    Returns:
        gesture_intervals (list of tuples): List of (start, end) indices for gesture intervals.
        threshold (float): Calculated threshold value.
    """
    # Calculate the threshold as a multiple of the mean envelope value
    threshold = threshold_multiplier * np.mean(envelope)
    
    gesture_intervals = []
    in_gesture = False
    start = 0

    for i, value in enumerate(envelope):
        if value > threshold and not in_gesture:
            in_gesture = True
            start = i
        elif value <= threshold and in_gesture:
            in_gesture = False
            gesture_intervals.append((start, i))
    
    return gesture_intervals, threshold

def plot_emg_signals_with_gestures(emg_data, gesture_intervals, threshold):
    """
    Plot the EMG signals for all 8 channels and highlight gesture intervals.
    Args:
        emg_data (list of lists): Raw EMG data.
        gesture_intervals (list of tuples): List of (start, end) indices for gesture intervals.
        threshold (float): Threshold value.
    """
    # Convert EMG data to a numpy array
    emg_data = np.array(emg_data)
    
    # Create a figure with 8 subplots
    fig, axs = plt.subplots(8, 1, figsize=(12, 16))
    fig.suptitle('EMG Signals with Gesture Intervals', fontsize=16)
    
    # Plot each channel's EMG signal and highlight gesture intervals
    for i in range(8):
        axs[i].plot(emg_data[:, i], label=f'Channel {i+1}')
        axs[i].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        
        # Highlight gesture intervals
        for start, end in gesture_intervals:
            axs[i].axvspan(start, end, color='yellow', alpha=0.3, label='Gesture Interval' if i == 0 else "")
        
        axs[i].set_ylabel('Amplitude')
        axs[i].legend(loc='upper right')
        axs[i].grid(True)
    
    # Set common x-axis label
    axs[-1].set_xlabel('Time Steps')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_envelopes_with_thresholds(envelopes, gesture_intervals, threshold):
    """
    Plot the smoothed envelopes for all 8 channels along with the threshold and highlight gesture intervals.
    Args:
        envelopes (list of numpy arrays): Smoothed envelopes for all 8 channels.
        gesture_intervals (list of tuples): List of (start, end) indices for gesture intervals.
        threshold (float): Threshold value.
    """
    # Create a figure with 8 subplots
    fig, axs = plt.subplots(8, 1, figsize=(12, 16))
    fig.suptitle('Smoothed Envelopes with Thresholds and Gesture Intervals', fontsize=16)
    
    # Plot each channel's envelope and the threshold
    for i in range(8):
        axs[i].plot(envelopes[i], label=f'Channel {i+1} Envelope')
        axs[i].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        
        # Highlight gesture intervals
        for start, end in gesture_intervals:
            axs[i].axvspan(start, end, color='yellow', alpha=0.3, label='Gesture Interval' if i == 0 else "")
        
        axs[i].set_ylabel('Amplitude')
        axs[i].legend(loc='upper right')
        axs[i].grid(True)
    
    # Set common x-axis label
    axs[-1].set_xlabel('Time Steps')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Feature extraction functions
def enhanced_wavelength(data):
    """
    Calculate the Enhanced Wavelength (EWL) feature.
    """
    L = len(data)
    p = 0.75 if 0.25 * L <= 0.8 * L else 0.25
    return (1 / L) * np.sum(np.abs(np.diff(data)) ** p)

def root_mean_square(data):
    return np.sqrt(np.mean(np.square(data)))

def modified_mean_absolute_value_2(data):
    L = len(data)
    wi = np.where((0.25 * L <= np.arange(L)) & (np.arange(L) <= 0.75 * L), 1 / (4 * L), 4 * (L - np.arange(L)) / L)
    return np.mean(wi * np.abs(data))

def difference_absolute_standard_deviation_value(data):
    return np.sqrt(np.mean(np.square(np.diff(data))))

def maximum_fractal_length(data):
    return np.log10(np.sum(np.square(np.diff(data))))

# Function to extract features from a window for all 8 channels
def extract_features(window):
    """
    Extract features from a window of EMG data for all 8 channels.
    
    Args:
        window (numpy array): A window of EMG data with shape (window_size, 8).
    
    Returns:
        list: A list of features for all 8 channels, followed by the label.
    """
    features = []
    for channel in range(8):  # Loop through all 8 channels
        channel_data = window[:, channel]  # Extract data for the current channel
        channel_features = [
            enhanced_wavelength(channel_data),
            root_mean_square(channel_data),
            modified_mean_absolute_value_2(channel_data),
            difference_absolute_standard_deviation_value(channel_data),
            maximum_fractal_length(channel_data)
        ]
        features.extend(channel_features)  # Add features for the current channel to the list
    return features

# Function to create windows and label them
def create_windows_and_labels(emg_data, labels, gesture_intervals):
    """
    Create windows from EMG data and label them based on whether they fall within a gesture region or rest region.
    
    Args:
        emg_data (numpy array): Normalized EMG data.
        labels (list): List of labels for the entire dataset.
        gesture_intervals (list of tuples): List of (start, end) indices for gesture intervals.
    
    Returns:
        windowed_data (list of numpy arrays): List of windows.
        window_labels (list): List of labels for each window.
    """
    windowed_data = []
    window_labels = []
    step = WINDOW_SIZE - OVERLAP

    for i in range(0, len(emg_data) - WINDOW_SIZE + 1, step):
        window = emg_data[i:i + WINDOW_SIZE]
        
        # Check if the window falls within any gesture interval
        is_gesture = False
        for start, end in gesture_intervals:
            if start <= i + WINDOW_SIZE // 2 <= end:  # Check if the middle of the window is in a gesture interval
                is_gesture = True
                break
        
        # Assign label: gesture_type if in a gesture region, otherwise 0 (rest)
        window_label = labels[i + WINDOW_SIZE // 2] if is_gesture else 0
        windowed_data.append(window)
        window_labels.append(window_label)
    
    return windowed_data, window_labels

def record_emg_data(gesture_type, duration=5):
    """
    Record EMG data for a specified duration and save it to a CSV file.
    
    Args:
        gesture_type (str): Type of gesture being recorded.
        duration (int): Duration of the recording in seconds.
    """
    emg_data = []
    start_time = time.time()
    
    print(f"Recording Gesture Type {gesture_type} for {duration} seconds...")
    window_counter = 0  # Counter to track the number of windows processed

    # Clear the queue to discard any old data
    while not q.empty():
        q.get()

    while time.time() - start_time < duration:
        if not q.empty():
            emg = list(q.get())
            emg_data.append(emg)
            print(emg)

            # Discard the first 2 windows
            if len(emg_data) >= WINDOW_SIZE:
                window_counter += 1
                if window_counter <= 2:
                    emg_data = emg_data[WINDOW_SIZE - OVERLAP:]  # Discard the first window
                    continue  # Skip processing for the first 2 windows
    
    # Normalize the EMG data
    normalized_emg_data = normalize_emg_data(emg_data)
    
    # Filter the normalized EMG data - 50 Hz notch and 10-100 Hz bandpass
    filtered_emg_data = filter_emg_data(normalized_emg_data, fs=SAMPLE_RATE)
    
    # Calculate the smooth envelope for all 8 channels
    envelopes = []
    for channel in range(8):  # Loop through all 8 channels
        channel_data = [sample[channel] for sample in filtered_emg_data]
        envelope = calculate_smooth_envelope(channel_data, window_size=20)  # Larger window for smoother envelope
        envelopes.append(envelope)
    
    # Evaluate the threshold (using the first channel's envelope)
    gesture_intervals, threshold = detect_gesture_intervals(envelopes[0], threshold_multiplier=1.5)
    
    # Plot the EMG signals with highlighted gesture intervals
    plot_emg_signals_with_gestures(filtered_emg_data, gesture_intervals, threshold)
    
    # Plot the smoothed envelopes with thresholds and highlighted gesture intervals
    plot_envelopes_with_thresholds(envelopes, gesture_intervals, threshold)

    # Create labels for the recorded data (gesture_type is the label for gesture regions)
    labels = [gesture_type] * len(filtered_emg_data)

    # Create windows and labels
    windowed_data, window_labels = create_windows_and_labels(np.array(filtered_emg_data), labels, gesture_intervals)

    # Extract features for each window
    feature_data = []
    for window, label in zip(windowed_data, window_labels):
        features = extract_features(window)  # Extract features for all 8 channels
        feature_data.append(features + [label])  # Append label to features

    # Define column names for the dataset
    columns = []
    for channel in range(1, 9):  # Loop through all 8 channels
        columns.extend([f'EWL_{channel}', f'RMS_{channel}', f'MMAV2_{channel}', f'DASDV_{channel}', f'MFL_{channel}'])
    columns.append('Label')  # Add the label column

    # Convert to DataFrame
    df = pd.DataFrame(feature_data, columns=columns)

    # Save or append to CSV
    csv_file = 'emg_features_all_gestures.csv'
    if os.path.exists(csv_file):
        # Append to existing CSV
        df.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"Features appended to '{csv_file}'")
    else:
        # Create new CSV
        df.to_csv(csv_file, index=False)
        print(f"Features saved to '{csv_file}'")

def main_menu():
    """
    Display a menu for the user to select a gesture type to record.
    """
    print("Select a gesture type to record:")
    print("1. Gesture Close Hand")
    print("2. Gesture Open Hand")
    print("3. Gesture Pointing")
    
    choice = input("Enter your choice (1-3): ")
    if choice in ['1', '2', '3']:
        record_emg_data(choice)
    else:
        print("Invalid choice. Please try again.")

# -------- Main Program Loop -----------
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    try:
        while True:
            main_menu()
            another = input("Do you want to record another gesture? (y/n): ")
            if another.lower() != 'y':
                break
    except KeyboardInterrupt:
        print("Quitting")
    finally:
        p.terminate()
        p.join()