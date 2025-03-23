import multiprocessing
import time
import numpy as np
from scipy.signal import hilbert  # For Hilbert transform
from pyomyo import Myo, emg_mode
import joblib  # For loading the trained model

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

# Feature extraction functions
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

# Function to extract features from a window
def extract_features(window):
    """
    Extract 5 features from a window of EMG data.
    """
    features = []
    for channel in range(8):  # 8 channels in Myo Armband
        channel_data = window[:, channel]
        features.extend([
            enhanced_wavelength(channel_data),
            root_mean_square(channel_data),
            modified_mean_absolute_value_2(channel_data),
            difference_absolute_standard_deviation_value(channel_data),
            maximum_fractal_length(channel_data)
        ])
    return features[:5]  # Only use the first 5 features

# Function to run the trained SKNN model in real-time
def run_sknn_model(model_path):
    """
    Run the trained SKNN model in real-time using Myo Armband data.
    
    Args:
        model_path (str): Path to the trained SKNN model (.pkl file).
    """
    # Load the trained model
    model = joblib.load(model_path)
    print("Model loaded successfully!")

    # Initialize Myo Armband
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    try:
        emg_buffer = []  # Buffer to store EMG data
        print("Waiting for Myo Armband to start streaming data...")

        # Wait until the Myo Armband starts streaming data
        while q.empty():
            time.sleep(0.1)  # Wait for 100 ms before checking again

        print("Myo Armband is ready. Running real-time gesture classification... Press Ctrl+C to stop.")

        while True:
            if not q.empty():
                emg = q.get()
                emg_normalized = normalize_emg_data([emg])[0]  # Normalize the EMG data
                emg_buffer.append(emg_normalized)

                # When the buffer has enough data for a window
                if len(emg_buffer) >= WINDOW_SIZE:
                    window = np.array(emg_buffer[-WINDOW_SIZE:])  # Use the latest window
                    features = extract_features(window)  # Extract features
                    gesture = model.predict([features])  # Predict the gesture
                    print(f"Predicted Gesture: {gesture[0]}")

    except KeyboardInterrupt:
        print("Stopping real-time classification...")
    finally:
        p.terminate()
        p.join()

# Run the real-time classification
if __name__ == "__main__":
    # Path to the trained SKNN model
    model_path = 'sknn_model_4_classes.pkl'  # Replace with the actual path to your model

    # Run the SKNN model in real-time
    run_sknn_model(model_path)