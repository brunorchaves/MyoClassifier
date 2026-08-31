"""Classificacao em tempo real com o modelo NN, no mesmo formato de janela
deslizante que myTry/myoRunModel.py usa para o kNN -- so troca a extracao de
features + model.predict por predict_window (sem features, direto no sinal
bruto).

Uso: python -m nn_classifier.live_infer --checkpoint nn_classifier/checkpoints/finetuned.pt
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np

from pyomyo import Myo, emg_mode

from nn_classifier.classifier_adapter import load_model, predict_window
from nn_classifier.data.label_map import CLASS_NAMES
from nn_classifier.data.windowing import OVERLAP, WINDOW_SIZE


def myo_worker(q):
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    m.add_emg_handler(lambda emg, moving: q.put(emg))
    m.set_leds([0, 180, 220], [0, 180, 220])
    m.vibrate(1)
    while True:
        m.run()


def classify_live(q, model, device):
    step = WINDOW_SIZE - OVERLAP
    buffer = []
    while True:
        emg = list(q.get())
        buffer.append(emg)
        if len(buffer) >= WINDOW_SIZE:
            window = np.array(buffer[-WINDOW_SIZE:], dtype=np.float32) / 128.0  # EMG bruto -> mesma escala do treino
            label = predict_window(model, window, device)
            print(f"Gesto previsto: {label} ({CLASS_NAMES.get(label, '?')})")
            buffer = buffer[step:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "finetuned.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.device)

    q = multiprocessing.Queue()
    worker = multiprocessing.Process(target=myo_worker, args=(q,))
    worker.start()
    try:
        classify_live(q, model, args.device)
    except KeyboardInterrupt:
        print("Parando...")
    finally:
        worker.terminate()
        worker.join()


if __name__ == "__main__":
    main()
