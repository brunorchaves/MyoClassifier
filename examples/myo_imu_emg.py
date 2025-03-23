from pyomyo import Myo, emg_mode
import sys
import math
import pandas as pd

from tomlkit import boolean
from threading import Lock, Thread
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
import keyboard  # Install with: pip install keyboard
import multiprocessing
import numpy as np
import socket
import queue
import time
# Queue for sharing data between threads
data_queue = queue.Queue()
host, port = "127.0.0.1", 25001
data = []

# Create a unique filename with timestamp to avoid overwriting
timestamp = time.strftime("%Y%m%d-%H%M%S")
emg_filename = f"emg_data_{timestamp}.txt"
# Open the file to save EMG data
emg_file = open(emg_filename, "w")

TRANSMIT_MODE = False
CLASSIFY_MODE = False

# Function to compute the derivative
def compute_derivative(emg_data, prev_data):
    return [(emg_data[i] - prev_data[i]) for i in range(8)]
# TensorFlow classification thread

def classify_gestures():
     
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    samples = 5
    columns = samples
    rows = 8
    totalSamples = samples * 8
    dimensions = (rows, columns)
    arraySize = (samples * rows) + 1

    # Arrays to store EMG data for each channel
    channel_0 = []
    channel_1 = []
    channel_2 = []
    channel_3 = []
    channel_4 = []
    channel_5 = []
    channel_6 = []
    channel_7 = []

    labelArray = np.zeros(1)
    dimensions_f = (0, arraySize)
    gestureArray = np.empty(dimensions_f)
    quantSamples = 0
    initializedVaribles = True
    model = tf.keras.models.load_model("gesturePredictor_RNN.model")
    print("model loaded")


    prev_gesture = None
    samples = 5  # Change as per your requirement
    # Arrays to store EMG data for each channel (this can be organized better)
    channel_0 = []
    channel_1 = []
    channel_2 = []
    channel_3 = []
    channel_4 = []
    channel_5 = []
    channel_6 = []
    channel_7 = []
    class_names = ['Relaxed', 'Fist', 'Spock', 'Pointing']

    while True:
        try:
                quantSamples = 0
                channel_0 = []
                channel_1 = []
                channel_2 = []
                channel_3 = []
                channel_4 = []
                channel_5 = []
                channel_6 = []
                channel_7 = []
                while quantSamples < samples:

                    emg_data = data_queue.get()  # Adjust timeout as needed
                    if None not in emg_data[:7] :
                        emg_float = [float(element) for element in emg_data]
                        # print(emg_float)

                        channel_0.append(emg_float[0])
                        channel_1.append(emg_float[1])
                        channel_2.append(emg_float[2])
                        channel_3.append(emg_float[3])
                        channel_4.append(emg_float[4])
                        channel_5.append(emg_float[5])
                        channel_6.append(emg_float[6])
                        channel_7.append(emg_float[7])
                        quantSamples += 1

                # Concatenate all channels into one horizontal array
                arrayLine = np.concatenate((channel_0, channel_1, channel_2, channel_3, channel_4, channel_5, channel_6, channel_7), axis=None)
                Single_gesture = arrayLine.reshape(1, samples * 8)
            
                prediction = model.predict(Single_gesture)
                predicted_class = class_names[np.argmax(prediction[0])]
                print("Previsão:", predicted_class)
                
                # Check if the predicted gesture is different from the previous one
                if predicted_class != prev_gesture:
                    if predicted_class == 'Relaxed':
                        keyboard.press_and_release('1')
                    elif predicted_class == 'Fist':
                        keyboard.press_and_release('2')
                    elif predicted_class == 'Spock':
                        keyboard.press_and_release('3')
                    elif predicted_class == 'Pointing':
                        keyboard.press_and_release('4')

                    prev_gesture = predicted_class

        except queue.Empty:
            continue  # If no data is available, continue to the next iteration

class MyoDataPrinter:
    def __init__(self):
        self.emg_data = [None] * 8
        self.euler_angles = [0.0, 0.0, 0.0]
        self.prev_emg_data = [0] * 8  # Stores the previous EMG data for computing the derivative
        self.myo = Myo(mode=emg_mode.RAW)
        self.myo.add_imu_handler(self.on_imu)
        self.myo.add_emg_handler(self.on_emg)
        self.myo.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs
        # Open the file for writing derivative data
        self.derivative_file = open("emg_derivative.txt", "w")

         # Buffer for storing EMG data windows
        self.emg_window = np.zeros((10, 8))  # 10 samples window for 8 channels
        self.window_size = 10
        self.current_sample = 0
        self.myo.connect()

    def on_imu(self, quat, acc, gyro):
        self.euler_angles = euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

    def on_emg(self, emg, moving):
        self.emg_data = list(emg)
        # Filter out negative values
        self.emg_data = [max(0, value) for value in self.emg_data]  # Keep only positive values
        self.derivative_data = compute_derivative(self.emg_data, self.prev_emg_data)
        self.prev_emg_data = self.emg_data[:]
        self.store_derivatives()
        

    def get_emg(self):
        return self.emg_data

    def get_euler_angles(self):
        return self.euler_angles
    
    def compute_and_save_derivative(self):
        # Calculate the derivative of the EMG signal
        derivative_data = compute_derivative(self.emg_data, self.prev_emg_data)
        self.prev_emg_data = self.emg_data[:]  # Update previous data with the current data

        # Write the derivative data to the file
        # self.derivative_file.write(",".join(map(str, derivative_data)) + "\n")
    def store_derivatives(self):
        # Shift window and store the new derivative
        self.emg_window = np.roll(self.emg_window, -1, axis=0)
        self.emg_window[-1, :] = self.derivative_data

        self.current_sample += 1
        if self.current_sample >= self.window_size:
            self.classify_gesture()
            

    def classify_gesture(self):
        # Simple rule-based classification based on channel derivatives
        avg_deriv_channel_1 = np.mean(self.emg_window[:, 1])
        avg_deriv_channel_4 = np.mean(self.emg_window[:, 4])

        # Print the average derivatives for channels 1 and 4
        print("Derivative ch1: " + str(avg_deriv_channel_1))
        print("Derivative ch4: " + str(avg_deriv_channel_4))


        
        if avg_deriv_channel_4 < 2.0 and  avg_deriv_channel_1 < 2.0:
            print("Gesture unclear")
        else:
            if avg_deriv_channel_1 > avg_deriv_channel_4:
                print("Detected gesture: Fist")
                keyboard.press_and_release('1')

            elif avg_deriv_channel_4 > avg_deriv_channel_1:
                print("Detected gesture: Open hand")
                keyboard.press_and_release('3')


    def run(self):
        try:
            self.myo.run()

        except KeyboardInterrupt:
            self.myo.disconnect()
            quit()

def normalize_quaternion(q):
    w, x, y, z = q
    # Calcula a magnitude do quaternion
    norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    # Normaliza os componentes
    if norm == 0:
        return 1, 0, 0, 0  # Evitar divisão por zero, retorna um quaternion neutro
    return w / norm, x / norm, y / norm, z / norm

def euler_from_quaternion(w, x, y, z):
    w, x, y, z = normalize_quaternion((w, x, y, z))

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    roll_x = int((roll_x)*(180/math.pi))
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    pitch_y = int((pitch_y)*(180/math.pi))


    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    yaw_z = int((yaw_z)*(180/math.pi))

    return roll_x, pitch_y, yaw_z  # in radians

if __name__ == '__main__':
    myo_printer = MyoDataPrinter()
    yaw, pitch, roll = 1.0, 2.0, 3.0
    control_value = 1
    yaw_zero = 0
    pitch_zero = 0
    roll_zero = 0

    if(CLASSIFY_MODE is True):
        # Start the TensorFlow classification thread
        classification_thread = Thread(target=classify_gestures, daemon=True)
        classification_thread.start()

    # SOCK_STREAM means TCP socket
    if(TRANSMIT_MODE == True):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server and send the data
        if(TRANSMIT_MODE == True):
            sock.connect((host, port))
        else:
            print("Test mode on")

        while True:
            myo_printer.run()
            emg = myo_printer.get_emg()
            euler_angles = myo_printer.get_euler_angles()
            if None not in emg[:7] and None not in euler_angles:
                emg_float = [float(element) for element in emg]
                myo_printer.compute_and_save_derivative()

                roll, pitch, yaw = euler_angles
                if keyboard.is_pressed('space'):
                    yaw_zero = yaw
                    pitch_zero = pitch
                    roll_zero = roll

                yaw = yaw- yaw_zero
                pitch = pitch - pitch_zero
                roll =  roll - roll_zero

                data = f"{roll},{yaw*(-1)},{pitch}"
                print(data)

                # # Save EMG data to the file
                # emg_data_str = ",".join(map(str, emg_float)) + "\n"
                # emg_file.write(emg_data_str)
                
                # print(data)

                if(CLASSIFY_MODE is True):
                    # Push data into the queue for classification
                    data_queue.put(emg)
                

                if(TRANSMIT_MODE == True):
                    sock.sendall(data.encode("utf-8"))
                    response = sock.recv(1024).decode("utf-8")
                    print(response)

    except KeyboardInterrupt:
        print("Exiting loop.")
    finally:
        if(TRANSMIT_MODE == True):
            sock.close()
        else:
            print("Test mode on")
