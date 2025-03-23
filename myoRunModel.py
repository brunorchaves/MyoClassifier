import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from pyomyo import Myo, emg_mode
import multiprocessing
import time

# Constants
WINDOW_SIZE = 100  # 100 ms window
OVERLAP = 50       # 50% overlap
SAMPLE_RATE = 200  # Assuming 200 Hz sample rate (adjust based on your data)

# Load the trained model and scaler
MODEL_PATH = 'rnn_gesture_classifier.h5'
SCALER_PATH = 'scaler.pkl'

model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Function to normalize EMG data
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

# Function to extract features from a window
def extract_features(window):
    """
    Extract features from a window of EMG data for all 8 channels.
    
    Args:
        window (numpy array): A window of EMG data with shape (window_size, 8).
    
    Returns:
        list: A list of features for all 8 channels.
    """
    def enhanced_wavelength(data):
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

# Function to classify gestures in real-time
def classify_gestures(q):
    """
    Classify gestures in real-time using the trained RNN model.
    
    Args:
        q (multiprocessing.Queue): Queue to receive EMG data from the Myo armband.
    """
    emg_data = []
    window_counter = 0

    while True:
        if not q.empty():
            emg = list(q.get())
            emg_data.append(emg)

            # Discard the first 2 windows
            if len(emg_data) >= WINDOW_SIZE:
                window_counter += 1
                if window_counter <= 2:
                    emg_data = emg_data[WINDOW_SIZE - OVERLAP:]  # Discard the first window
                    continue  # Skip processing for the first 2 windows

                # Extract the current window
                window = np.array(emg_data[-WINDOW_SIZE:])

                # Normalize the window
                normalized_window = normalize_emg_data(window)

                # Extract features
                features = extract_features(np.array(normalized_window))

                # Scale the features
                features_scaled = scaler.transform([features])

                # Reshape for RNN input
                features_scaled = features_scaled.reshape((1, 1, features_scaled.shape[1]))

                # Predict the gesture
                prediction = model.predict(features_scaled)
                predicted_class = np.argmax(prediction)

                # Print the predicted gesture
                print(f"Predicted Gesture: {predicted_class}")

# Myo worker function
def myo_worker(q):
    """
    Worker function to read EMG data from the Myo armband.
    
    Args:
        q (multiprocessing.Queue): Queue to store EMG data.
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

    while True:
        m.run()

# Main function
def main():
    q = multiprocessing.Queue()

    # Start the Myo worker process
    myo_process = multiprocessing.Process(target=myo_worker, args=(q,))
    myo_process.start()

    try:
        # Start real-time gesture classification
        classify_gestures(q)
    except KeyboardInterrupt:
        print("Stopping real-time classification...")
    finally:
        myo_process.terminate()
        myo_process.join()

if __name__ == "__main__":
    main()