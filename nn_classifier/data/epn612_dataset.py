"""Dataset PyTorch para o EMG-EPN612.

Schema do JSON (verificado contra o preprocessador de referencia em
https://github.com/j0shmillar/physiolite/blob/main/data_prep/epn612_preprocess.py,
que roda contra o dataset real):

    <root>/trainingJSON/user*/user*.json
    <root>/testingJSON/user*/user*.json

Cada arquivo de usuario e um dict com chaves "trainingSamples" e
"testingSamples" (dict de amostra -> sample). Cada sample tem:
    sample["emg"]         -> dict de 8 canais (chaves ordenaveis, ex. "ch1".."ch8"),
                             cada um uma lista de valores brutos (~ -128..127)
    sample["gestureName"] -> "noGesture" | "waveIn" | "waveOut" | "pinch" |
                             "open" | "fist" | "notProvided"

Splits (mesma logica do preprocessador de referencia, para reproduzir o
benchmark oficial da forma como a comunidade avalia este dataset):
    train = trainingJSON.trainingSamples + (testingJSON.trainingSamples,
            as primeiras 10 repeticoes de cada gesto por usuario -- dados de
            calibracao)
    val   = trainingJSON.testingSamples
    test  = testingJSON.trainingSamples, repeticoes >= 10 por usuario/gesto
            (held-out real; testingJSON.testingSamples nao tem rotulo --
            e reservado para submissao ao benchmark oficial e nao e usado aqui)
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nn_classifier.data.label_map import EPN612_GESTURE_TO_LABEL, WAVE_IN, WAVE_OUT, PINCH
from nn_classifier.data.windowing import slice_windows

import json

EXTRA_LABELS = {WAVE_IN, WAVE_OUT, PINCH}


def _find_dir(root: Path, name: str) -> Path:
    root = Path(root)
    direct = root / name
    if direct.is_dir():
        return direct
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"Nao encontrei uma pasta '{name}' dentro de {root}. "
            "Rode nn_classifier/data/download_epn612.py primeiro, ou aponte --epn612-root "
            "para onde o zip foi extraido."
        )
    return matches[0]


def _user_files(json_dir: Path):
    return sorted(json_dir.glob("user*/user*.json"))


def _emg_window_matrix(sample: dict) -> np.ndarray:
    emg = sample["emg"]
    channels = [np.asarray(emg[k], dtype=np.float32) for k in sorted(emg.keys())]
    matrix = np.stack(channels, axis=1)  # (n_amostras, 8)
    return matrix / 128.0  # mesma normalizacao usada em myTry/myoFeatures.py


def _windows_for_sample(sample: dict, label: int, max_subjects_windows: list):
    matrix = _emg_window_matrix(sample)
    for window in slice_windows(matrix):
        max_subjects_windows.append((window, label))


class EPN612Dataset(Dataset):
    def __init__(self, root, split: str = "train", extra_classes: bool = True, max_subjects: int | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split invalido: {split!r} (use train/val/test)")

        root = Path(root)
        self.samples: list[tuple[np.ndarray, int]] = []

        training_dir = _find_dir(root, "trainingJSON")
        testing_dir = _find_dir(root, "testingJSON")

        if split == "train":
            self._load_training_json(training_dir, "trainingSamples", extra_classes, max_subjects)
        elif split == "val":
            self._load_training_json(training_dir, "testingSamples", extra_classes, max_subjects)
        if split in ("train", "test"):
            self._load_testing_json(testing_dir, split, extra_classes, max_subjects)

    def _label_for(self, gesture_name: str, extra_classes: bool):
        label = EPN612_GESTURE_TO_LABEL.get(gesture_name)
        if label is None:
            return None
        if not extra_classes and label in EXTRA_LABELS:
            return None
        return label

    def _load_training_json(self, json_dir: Path, bucket: str, extra_classes: bool, max_subjects):
        files = _user_files(json_dir)
        if max_subjects:
            files = files[:max_subjects]
        for path in files:
            user_data = json.loads(path.read_text(encoding="utf-8"))
            for sample in user_data.get(bucket, {}).values():
                label = self._label_for(sample.get("gestureName"), extra_classes)
                if label is not None:
                    _windows_for_sample(sample, label, self.samples)

    def _load_testing_json(self, json_dir: Path, split: str, extra_classes: bool, max_subjects):
        files = _user_files(json_dir)
        if max_subjects:
            files = files[:max_subjects]
        for path in files:
            user_data = json.loads(path.read_text(encoding="utf-8"))
            grouped = defaultdict(list)
            for sample in user_data.get("trainingSamples", {}).values():
                grouped[sample.get("gestureName")].append(sample)
            for gesture_name, reps in grouped.items():
                label = self._label_for(gesture_name, extra_classes)
                if label is None:
                    continue
                for idx, sample in enumerate(reps):
                    is_calibration = idx < 10
                    if (split == "train" and is_calibration) or (split == "test" and not is_calibration):
                        _windows_for_sample(sample, label, self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        return torch.from_numpy(window.T.copy()), label  # (8, WINDOW_SIZE)
