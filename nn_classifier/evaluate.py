"""Compara o classificador NN (zero-shot / fine-tuned) com um kNN treinado no
MESMO split dos dados proprios -- responde com numeros a pergunta original
("a rede neural traz resultados melhores?") antes de trocar o classificador
em uso real. Ver docs/plans/emg-nn-pretrain-finetune/PLAN.md.

Uso:
    python -m nn_classifier.evaluate \
        --pretrained nn_classifier/checkpoints/epn612_pretrained.pt \
        --finetuned nn_classifier/checkpoints/finetuned.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from myTry.myoFeatures import extract_features

from nn_classifier.classifier_adapter import predict_window
from nn_classifier.data.label_map import CLASS_NAMES
from nn_classifier.data.own_dataset import DEFAULT_DATA_DIR, load_own_windows
from nn_classifier.model import build_model
from nn_classifier.train import pick_device

MIN_WINDOWS_PER_CLASS = 2  # abaixo disso nao ha como fazer holdout estratificado


def _split(windows_by_label, test_size, seed=0):
    present = {label: windows for label, windows in windows_by_label.items() if len(windows) >= MIN_WINDOWS_PER_CLASS}
    dropped = [CLASS_NAMES[l] for l in windows_by_label if l not in present]
    if dropped:
        print(f"Aviso: classes com menos de {MIN_WINDOWS_PER_CLASS} janelas ficam fora desta avaliacao: {dropped}. "
              "Grave mais amostras para essas classes (veja nn_classifier/README.md).")

    X = np.stack([w for windows in present.values() for w in windows])
    y = np.array([label for label, windows in present.items() for _ in windows], dtype=np.int64)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def _knn_baseline(X_train, y_train, X_test, y_test):
    feats_train = np.array([extract_features(w) for w in X_train])
    feats_test = np.array([extract_features(w) for w in X_test])
    scaler = StandardScaler().fit(feats_train)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(y_train)))
    knn.fit(scaler.transform(feats_train), y_train)
    return knn.predict(scaler.transform(feats_test))


def _nn_predictions(checkpoint_path, X_test, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return np.array([predict_window(model, window, device) for window in X_test])


def _report(name, y_test, y_pred):
    labels_present = sorted(set(y_test) | set(y_pred))
    print(f"\n=== {name} ===")
    print(f"acuracia: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(
        y_test, y_pred, labels=labels_present,
        target_names=[CLASS_NAMES[l] for l in labels_present], zero_division=0,
    ))
    print(f"matriz de confusao (linhas=real, colunas=previsto), ordem das classes {labels_present}:")
    print(confusion_matrix(y_test, y_pred, labels=labels_present))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own-data-root", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--pretrained", type=Path, default=None, help="checkpoint do pretrain (avaliacao zero-shot)")
    ap.add_argument("--finetuned", type=Path, default=None, help="checkpoint do fine-tune")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test-size", type=float, default=0.3)
    args = ap.parse_args()

    if args.pretrained is None and args.finetuned is None:
        raise SystemExit("Informe pelo menos --pretrained ou --finetuned para ter algo pra comparar com o kNN.")

    windows_by_label = load_own_windows(args.own_data_root)
    X_train, X_test, y_train, y_test = _split(windows_by_label, args.test_size)
    print(f"Split proprio: {len(y_train)} janelas de treino / {len(y_test)} de teste. "
          f"Classes no teste: {sorted(CLASS_NAMES[l] for l in set(y_test))}")

    device = pick_device(args.device)

    knn_pred = _knn_baseline(X_train, y_train, X_test, y_test)
    _report("kNN (baseline, treinado no mesmo split)", y_test, knn_pred)

    if args.pretrained:
        _report("NN pretreinada no EPN612, sem fine-tune (zero-shot)",
                 y_test, _nn_predictions(args.pretrained, X_test, device))

    if args.finetuned:
        _report("NN fine-tuned nos dados proprios",
                 y_test, _nn_predictions(args.finetuned, X_test, device))


if __name__ == "__main__":
    main()
