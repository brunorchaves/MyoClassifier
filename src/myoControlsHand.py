# === Arquivo principal: main.py ===

from collections import Counter, deque
import struct
import sys
import time
import pygame
from pygame.locals import *
import numpy as np
import keyboard
from pyomyo import Myo, emg_mode
import math
import queue
import socket
import multiprocessing
from plot_emgs import plot_worker

# Queue for sharing data with the plotter
plot_queue = multiprocessing.Queue()

# Constantes
host, port = "127.0.0.1", 25001
TRANSMIT_MODE = True
PLOT_MODE = True
SUBSAMPLE = 3
K = 15

def normalize_quaternion(q):
    w, x, y, z = q
    norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    return (w / norm, x / norm, y / norm, z / norm) if norm else (1.0, 0.0, 0.0, 0.0)

def euler_from_quaternion(w, x, y, z):
    w, x, y, z = normalize_quaternion((w, x, y, z))
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = int(math.atan2(t0, t1) * 180 / math.pi)
    t2 = max(-1.0, min(+1.0, +2.0 * (w * y - z * x)))
    pitch_y = int(math.asin(t2) * 180 / math.pi)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = int(math.atan2(t3, t4) * 180 / math.pi)
    return roll_x, pitch_y, yaw_z

class Classifier:
    def __init__(self, name="Classifier", color=(0, 200, 0)):
        self.name = name
        self.color = color
        for i in range(10):
            open(f'data/vals{i}.dat', 'ab').close()
        self.read_data()

    def store_data(self, cls, vals):
        with open(f'data/vals{cls}.dat', 'ab') as f:
            f.write(struct.pack('<8H', *vals))
        self.train(np.vstack([self.X, vals]), np.hstack([self.Y, [cls]]))

    def read_data(self):
        X, Y = [], []
        for i in range(10):
            Xi = np.fromfile(f'data/vals{i}.dat', dtype=np.uint16).reshape((-1, 8))
            Yi = i + np.zeros(Xi.shape[0])
            X.append(Xi)
            Y.append(Yi)
        self.train(np.vstack(X), np.hstack(Y))

    def delete_data(self):
        for i in range(10):
            open(f'data/vals{i}.dat', 'wb').close()
        self.read_data()

    def train(self, X, Y):
        self.X, self.Y = X, Y

    def nearest(self, d):
        dists = ((self.X - d)**2).sum(1)
        return self.Y[dists.argmin()]

    def classify(self, d):
        return 0 if self.X.shape[0] < K * SUBSAMPLE else self.nearest(d)

class MyoClassifier(Myo):
    def __init__(self, cls, tty=None, mode=emg_mode.PREPROCESSED, hist_len=25):
        super().__init__(tty, mode=mode)
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
        self.euler_angles = euler_from_quaternion(*quat)

    def emg_handler(self, emg, moving):
        y = self.cls.classify(emg)
        self.history_cnt[self.history[0]] -= 1
        self.history_cnt[y] += 1
        self.history.append(y)

        r, n = self.history_cnt.most_common(1)[0]
        if self.last_pose is None or (n > self.history_cnt[self.last_pose] + 5 and n > self.hist_len / 2):
            self.on_raw_pose(r)
            self.last_pose = r

    def get_euler_angles(self):
        return self.euler_angles

    def add_raw_pose_handler(self, h):
        self.pose_handlers.append(h)

    def on_raw_pose(self, pose):
        for h in self.pose_handlers:
            h(pose)
        now = time.time()
        if now - self.last_key_press_time >= 0.4:
            if 0 <= pose <= 9:
                keyboard.press(str(int(pose)))
                time.sleep(0.1)
                keyboard.release(str(int(pose)))
            self.last_key_press_time = now

class EMGHandler:
    def __init__(self, m, plot_queue=None):
        self.recording = -1
        self.m = m
        self.emg = (0,) * 8
        self.plot_queue = plot_queue

    def __call__(self, emg, moving):
        self.emg = emg
        if self.recording >= 0:
            self.m.cls.store_data(self.recording, emg)
        if self.plot_queue:
            self.plot_queue.put(emg)

    def get_emg(self):
        return self.emg

if __name__ == '__main__':
    plot_process = None
    plot_queue = None

    if PLOT_MODE:
        plot_queue = multiprocessing.Queue()
        plot_process = multiprocessing.Process(target=plot_worker, args=(plot_queue,))
        plot_process.start()

    m = MyoClassifier(Classifier())
    hnd = EMGHandler(m, plot_queue=plot_queue if PLOT_MODE else None)
    m.add_emg_handler(hnd)
    m.connect()
    m.add_raw_pose_handler(print)
    m.set_leds(m.cls.color, m.cls.color)

    yaw_zero = pitch_zero = roll_zero = 0

    try:
        if TRANSMIT_MODE:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))

        while True:
            euler_angles = m.get_euler_angles()
            emg_data = hnd.get_emg()

            if None not in euler_angles:
                roll, pitch, yaw = euler_angles
                if keyboard.is_pressed('space'):
                    yaw_zero, pitch_zero, roll_zero = yaw, pitch, roll

                yaw -= yaw_zero
                pitch -= pitch_zero
                roll -= roll_zero

                data_str = f"{roll},{-yaw},{pitch}"
                print(data_str)
                print(emg_data)

            m.run()

            if TRANSMIT_MODE:
                sock.sendall(data_str.encode("utf-8"))
                response = sock.recv(1024).decode("utf-8")
                print(response)

    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        if plot_process:
            plot_process.terminate()
            plot_process.join()


# === Arquivo separado: emg_plot.py ===

import pygame
from pygame.locals import *
import multiprocessing

last_vals = None

def plot(scr, vals, w, h):
    global last_vals
    DRAW_LINES = True
    D = 5
    if last_vals is None:
        last_vals = vals
        return
    scr.scroll(-D)
    scr.fill((0, 0, 0), (w - D, 0, D, h))
    for i, (u, v) in enumerate(zip(last_vals, vals)):
        if DRAW_LINES:
            pygame.draw.line(scr, (0, 255, 0),
                             (w - D, int(h / 9 * (i + 1 - u))),
                             (w, int(h / 9 * (i + 1 - v))))
            pygame.draw.line(scr, (255, 255, 255),
                             (w - D, int(h / 9 * (i + 1))),
                             (w, int(h / 9 * (i + 1))))
    pygame.display.flip()
    last_vals = vals

def plot_worker(q):
    pygame.init()
    w, h = 800, 600
    scr = pygame.display.set_mode((w, h))
    pygame.display.set_caption("EMG Plot")
    try:
        while True:
            pygame.event.pump()
            while not q.empty():
                emg = list(q.get())
                normalized = [e / 500.0 for e in emg]
                plot(scr, normalized, w, h)
    except KeyboardInterrupt:
        pygame.quit()
