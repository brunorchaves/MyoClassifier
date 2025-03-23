from collections import Counter, deque
import struct
import sys
import time
import pygame
from pygame.locals import *
import numpy as np
import keyboard  # For simulating key presses
from pyomyo import Myo, emg_mode
import math
import queue
import socket

# Queue for sharing data between threads
data_queue = queue.Queue()
host, port = "127.0.0.1", 25001
data = []
TRANSMIT_MODE = False

SUBSAMPLE = 3
K = 15

def normalize_quaternion(q):
    """Normalize a quaternion to unit length."""
    w, x, y, z = q
    norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    if norm == 0:
        return 1.0, 0.0, 0.0, 0.0  # Return a neutral quaternion to avoid division by zero
    return w / norm, x / norm, y / norm, z / norm

def euler_from_quaternion(w, x, y, z):
    """Convert a quaternion into Euler angles (roll, pitch, yaw)."""
    w, x, y, z = normalize_quaternion((w, x, y, z))

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    roll_x = int((roll_x) * (180 / math.pi))

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    pitch_y = int((pitch_y) * (180 / math.pi))

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    yaw_z = int((yaw_z) * (180 / math.pi))

    return roll_x, pitch_y, yaw_z

class Classifier:
    """A wrapper for nearest-neighbor classifier that stores training data in vals0, ..., vals9.dat."""

    def __init__(self, name="Classifier", color=(0, 200, 0)):
        self.name = name
        self.color = color
        for i in range(10):
            with open(f'data/vals{i}.dat', 'ab') as f:
                pass
        self.read_data()

    def store_data(self, cls, vals):
        """Store EMG data for a specific class."""
        with open(f'data/vals{cls}.dat', 'ab') as f:
            f.write(struct.pack('<8H', *vals))
        self.train(np.vstack([self.X, vals]), np.hstack([self.Y, [cls]]))

    def read_data(self):
        """Read training data from files."""
        X = []
        Y = []
        for i in range(10):
            X.append(np.fromfile(f'data/vals{i}.dat', dtype=np.uint16).reshape((-1, 8)))
            Y.append(i + np.zeros(X[-1].shape[0]))
        self.train(np.vstack(X), np.hstack(Y))

    def delete_data(self):
        """Delete all training data."""
        for i in range(10):
            with open(f'data/vals{i}.dat', 'wb') as f:
                pass
        self.read_data()

    def train(self, X, Y):
        """Train the classifier."""
        self.X = X
        self.Y = Y
        self.model = None

    def nearest(self, d):
        """Find the nearest neighbor."""
        dists = ((self.X - d)**2).sum(1)
        ind = dists.argmin()
        return self.Y[ind]

    def classify(self, d):
        """Classify the input data."""
        if self.X.shape[0] < K * SUBSAMPLE:
            return 0
        return self.nearest(d)

class MyoClassifier(Myo):
    """Adds higher-level pose classification and handling onto Myo."""

    def __init__(self, cls, tty=None, mode=emg_mode.PREPROCESSED, hist_len=25):
        Myo.__init__(self, tty, mode=mode)
        self.cls = cls
        self.hist_len = hist_len
        self.history = deque([0] * self.hist_len, self.hist_len)
        self.history_cnt = Counter(self.history)
        self.euler_angles = [0.0, 0.0, 0.0]
        self.add_imu_handler(self.on_imu)
        self.add_emg_handler(self.emg_handler)
        self.last_pose = None
        self.pose_handlers = []
        self.last_key_press_time = 0

    def on_imu(self, quat, acc, gyro):
        """Handle IMU data and update Euler angles."""
        self.euler_angles = euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

    def emg_handler(self, emg, moving):
        """Handle EMG data and classify poses."""
        y = self.cls.classify(emg)
        self.history_cnt[self.history[0]] -= 1
        self.history_cnt[y] += 1
        self.history.append(y)

        r, n = self.history_cnt.most_common(1)[0]
        if self.last_pose is None or (n > self.history_cnt[self.last_pose] + 5 and n > self.hist_len / 2):
            self.on_raw_pose(r)
            self.last_pose = r

    def get_euler_angles(self):
        """Return the current Euler angles."""
        return self.euler_angles

    def add_raw_pose_handler(self, h):
        """Add a handler for raw pose events."""
        self.pose_handlers.append(h)

    def on_raw_pose(self, pose):
        """Handle raw pose events."""
        for h in self.pose_handlers:
            h(pose)

        # Simulate key press based on the detected gesture
        current_time = time.time()
        if current_time - self.last_key_press_time >= 0.4:  # Throttle key presses
            if 0 <= pose <= 9:
                self.simulate_key_press(str(int(pose)))  # Convert pose to integer and then to string
            self.last_key_press_time = current_time

    def simulate_key_press(self, key):
        """Simulate a key press using the `keyboard` library."""
        keyboard.press(key)
        time.sleep(0.1)
        keyboard.release(key)

    def run_gui(self, hnd, scr, font, w, h):
        """Run the Pygame GUI."""
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):
                raise KeyboardInterrupt()
            elif ev.type == KEYDOWN:
                if K_0 <= ev.key <= K_9:
                    hnd.recording = ev.key - K_0
                elif ev.unicode == 'r':
                    hnd.cl.read_data()
                elif ev.unicode == 'e':
                    print("Pressed e, erasing local data")
                    self.cls.delete_data()
            elif ev.type == KEYUP:
                if K_0 <= ev.key <= K_9:
                    hnd.recording = -1

        # Plotting
        scr.fill((0, 0, 0), (0, 0, w, h))
        r = self.history_cnt.most_common(1)[0][0]

        for i in range(10):
            x = 0
            y = 0 + 30 * i
            clr = self.cls.color if i == r else (255, 255, 255)

            txt = font.render(f'{self.history_cnt[i]:5d}', True, (255, 255, 255))
            scr.blit(txt, (x + 20, y))

            txt = font.render(f'{i}', True, clr)
            scr.blit(txt, (x + 110, y))

            scr.fill((0, 0, 0), (x + 130, y + txt.get_height() / 2 - 10, len(self.history) * 20, 20))
            scr.fill(clr, (x + 130, y + txt.get_height() / 2 - 10, self.history_cnt[i] * 20, 20))

        pygame.display.flip()

class EMGHandler:
    """Handle EMG data and store it for training."""

    def __init__(self, m):
        self.recording = -1
        self.m = m
        self.emg = (0,) * 8

    def __call__(self, emg, moving):
        self.emg = emg
        if self.recording >= 0:
            self.m.cls.store_data(self.recording, emg)

if __name__ == '__main__':
    pygame.init()
    w, h = 800, 320
    scr = pygame.display.set_mode((w, h))
    font = pygame.font.Font(None, 30)
    yaw, pitch, roll = 1.0, 2.0, 3.0
    control_value = 1
    yaw_zero = 0
    pitch_zero = 0
    roll_zero = 0
    m = MyoClassifier(Classifier())
    hnd = EMGHandler(m)
    m.add_emg_handler(hnd)
    m.connect()

    m.add_raw_pose_handler(print)
    m.set_leds(m.cls.color, m.cls.color)
    pygame.display.set_caption(m.cls.name)

    try:
        if TRANSMIT_MODE:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
        else:
            print("Test mode on")

        while True:
            euler_angles = m.get_euler_angles()
            if None not in euler_angles:
                roll, pitch, yaw = euler_angles
                if keyboard.is_pressed('space'):
                    yaw_zero = yaw
                    pitch_zero = pitch
                    roll_zero = roll

                yaw -= yaw_zero
                pitch -= pitch_zero
                roll -= roll_zero

                data = f"{roll},{yaw * (-1)},{pitch}"
                print(data)

            m.run()
            m.run_gui(hnd, scr, font, w, h)

            if TRANSMIT_MODE:
                sock.sendall(data.encode("utf-8"))
                response = sock.recv(1024).decode("utf-8")
                print(response)

    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        pygame.quit()