"""Convencao de janelamento compartilhada por todo o pipeline (mesma usada em
myTry/myoFeatures.py e myTry/myoRunModel.py), para que dados do EPN612 e dados
proprios cheguem ao modelo no mesmo formato.
"""

import numpy as np

WINDOW_SIZE = 100
OVERLAP = 50
STEP = WINDOW_SIZE - OVERLAP
SAMPLE_RATE = 200


def slice_windows(signal: np.ndarray):
    """signal: array (n_amostras, 8) -> lista de janelas (WINDOW_SIZE, 8).

    Janelas mais curtas que WINDOW_SIZE sao descartadas (nao ha o que fazer
    com elas alem de padding, que introduziria sinal artificial).
    """
    n = signal.shape[0]
    if n < WINDOW_SIZE:
        return []
    return [signal[start:start + WINDOW_SIZE] for start in range(0, n - WINDOW_SIZE + 1, STEP)]
