"""Le os dados proprios gravados com 'src/emgGestureTrainer.py'
(src/data/vals0.dat .. vals9.dat), no mesmo esquema de rotulos de
nn_classifier/data/label_map.py (PROJECT_CLASSES).

Cada vals{N}.dat e uma sequencia continua de amostras brutas de 8 canais
(uint16, gravadas por Classifier.store_data em src/emgGestureTrainer.py:51-53)
para a classe N. Aqui so fatiamos essa sequencia em janelas -- sem
reamostrar nem filtrar, para ficar no mesmo dominio (RAW) que o
MyoClassifier ao vivo ve em src/emgGestureTrainer.py.
"""

from pathlib import Path

import numpy as np

from nn_classifier.data.label_map import PROJECT_CLASSES
from nn_classifier.data.windowing import slice_windows

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "data"


def load_own_windows(data_dir: Path = DEFAULT_DATA_DIR):
    """Retorna dict {label: lista de janelas (WINDOW_SIZE, 8)}."""
    data_dir = Path(data_dir)
    windows_by_label = {}
    for label in PROJECT_CLASSES:
        path = data_dir / f"vals{label}.dat"
        if not path.exists():
            windows_by_label[label] = []
            continue
        raw = np.fromfile(path, dtype=np.uint16)
        if raw.size % 8 != 0:
            raw = raw[: raw.size - (raw.size % 8)]
        signal = raw.reshape((-1, 8)).astype(np.float32) / 128.0  # mesma normalizacao do resto do projeto
        windows_by_label[label] = slice_windows(signal)
    return windows_by_label
