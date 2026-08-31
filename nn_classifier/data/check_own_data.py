"""Mostra quantas amostras/janelas existem hoje em src/data/vals*.dat por
classe -- roda depois de coletar dados com 'python src/emgGestureTrainer.py'
para saber se ja da pra fazer um fine-tune razoavel.

Uso: python -m nn_classifier.data.check_own_data
"""

from nn_classifier.data.label_map import CLASS_NAMES, PROJECT_CLASSES
from nn_classifier.data.own_dataset import DEFAULT_DATA_DIR, load_own_windows


def main():
    windows_by_label = load_own_windows()
    print(f"Lendo de: {DEFAULT_DATA_DIR}\n")
    print(f"{'classe':<10}{'nome':<12}{'amostras brutas':<18}{'janelas (100/50%)':<20}")
    for label in PROJECT_CLASSES:
        windows = windows_by_label[label]
        n_raw = len(windows) * 50 + 50 if windows else 0  # aproximado (overlap 50%)
        flag = "  <-- sem dados" if not windows else ""
        print(f"{label:<10}{CLASS_NAMES[label]:<12}{'~' + str(n_raw):<18}{len(windows):<20}{flag}")

    empty = [CLASS_NAMES[l] for l, w in windows_by_label.items() if not w]
    if empty:
        print(
            f"\nSem nenhuma amostra para: {', '.join(empty)}. "
            "O fine-tune vai pular essas classes (o cabecote da rede fica "
            "sem gradiente pra elas) -- grave essas classes antes de treinar. "
            "Veja o passo a passo em nn_classifier/README.md."
        )


if __name__ == "__main__":
    main()
