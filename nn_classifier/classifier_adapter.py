"""Adaptadores que plugam o modelo treinado nas duas interfaces de
classificador que ja existem no projeto:

  - NNGestureClassifier: implementa o mesmo "contrato de pato" que
    Classifier/Live_Classifier em src/emgGestureTrainer.py (atributos name,
    color, X, Y; metodos classify/train/store_data/read_data/delete_data),
    para usar direto com MyoClassifier/EMGHandler/run_gui de la sem alterar
    esse arquivo. Nao herda dessas classes porque o __init__ delas faz IO em
    'data/vals*.dat' relativo ao cwd, que nao se aplica aqui (o modelo ja vem
    treinado de um checkpoint).

  - predict_window: funcao simples pra quem usa o padrao de janela deslizante
    do myTry/ (ver nn_classifier/live_infer.py).
"""

from collections import deque

import numpy as np
import torch

from nn_classifier.data.label_map import PROJECT_CLASSES
from nn_classifier.data.windowing import WINDOW_SIZE
from nn_classifier.model import build_model


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_window(model, window: np.ndarray, device="cpu") -> int:
    """window: array (WINDOW_SIZE, 8) JA normalizado para ~[-1,1] (mesma
    normalizacao /128 aplicada por nn_classifier/data/own_dataset.py e
    epn612_dataset.py). Quem tem EMG bruto direto do Myo (ex.
    NNGestureClassifier.classify, live_infer.py) precisa dividir por 128
    antes de chamar esta funcao -- normalizar duas vezes silenciosamente
    quebraria a inferencia."""
    x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x)
    return int(logits.argmax(1).item())


class NNGestureClassifier:
    def __init__(self, checkpoint_path, name="NN", color=(0, 180, 220), device="cpu"):
        self.name = name
        self.color = color
        self.device = device
        self.model = load_model(checkpoint_path, device)

        self.buffer = deque(maxlen=WINDOW_SIZE)
        # Cosmeticos p/ MyoClassifier.run_gui (barras de contagem por classe):
        # aqui refletem previsoes feitas ao vivo, nao amostras de treino.
        self.X = np.zeros((0, 8))
        self.Y = np.zeros((0,))
        self._prediction_counts = {label: 0 for label in PROJECT_CLASSES}

    def classify(self, emg) -> int:
        self.buffer.append(emg)
        if len(self.buffer) < WINDOW_SIZE:
            return 0
        window = np.array(self.buffer, dtype=np.float32) / 128.0  # EMG bruto -> mesma escala do treino
        label = predict_window(self.model, window, self.device)
        self._prediction_counts[label] = self._prediction_counts.get(label, 0) + 1
        self.Y = np.array(
            [label for label, count in self._prediction_counts.items() for _ in range(count)]
        )
        return label

    def train(self, X, Y):
        pass  # o treino acontece offline via nn_classifier/train.py

    def store_data(self, cls, vals):
        pass  # gravacao ao vivo por tecla nao se aplica a este classificador --
        # use src/emgGestureTrainer.py para coletar e nn_classifier/train.py --stage finetune pra retreinar

    def read_data(self):
        pass

    def delete_data(self):
        pass
