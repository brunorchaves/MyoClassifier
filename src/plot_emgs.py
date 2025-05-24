# plot_emgs.py

import pygame
from pygame.locals import *

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
