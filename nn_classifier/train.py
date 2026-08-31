"""Treino em duas etapas do classificador de gestos por EMG.

Pretrain -- aprende representacoes gerais de ativacao muscular em 612
sujeitos do EMG-EPN612 (mesmo hardware: 1 Myo, 8 canais, 200Hz):

    python -m nn_classifier.train --stage pretrain \
        --epn612-root nn_classifier/data/raw/epn612 \
        --out nn_classifier/checkpoints/epn612_pretrained.pt

Fine-tune -- especializa esse backbone nos seus proprios dados
(src/data/vals*.dat, gravados com src/emgGestureTrainer.py -- ver o passo a
passo de coleta em nn_classifier/README.md):

    python -m nn_classifier.train --stage finetune \
        --pretrained nn_classifier/checkpoints/epn612_pretrained.pt \
        --out nn_classifier/checkpoints/finetuned.pt
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nn_classifier.data.epn612_dataset import EPN612Dataset
from nn_classifier.data.label_map import CLASS_NAMES, PROJECT_CLASSES, NUM_CLASSES
from nn_classifier.data.own_dataset import DEFAULT_DATA_DIR, load_own_windows
from nn_classifier.model import build_model

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def pick_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_pretrain(args):
    device = pick_device(args.device)

    train_ds = EPN612Dataset(args.epn612_root, split="train", extra_classes=args.extra_classes, max_subjects=args.max_subjects)
    val_ds = EPN612Dataset(args.epn612_root, split="val", extra_classes=args.extra_classes, max_subjects=args.max_subjects)
    print(f"EPN612: {len(train_ds)} janelas de treino, {len(val_ds)} de validacao.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device)
        print(f"epoca {epoch + 1}/{args.epochs}  train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"backbone": model.backbone_state_dict(), "state_dict": model.state_dict(),
                        "num_classes": NUM_CLASSES}, args.out)
    print(f"Melhor val_acc={best_val_acc:.3f}. Checkpoint salvo em {args.out}")


def _train_loop(model, X, y, device, epochs, lr, batch_size=32):
    dataset = TensorDataset(torch.from_numpy(X.transpose(0, 2, 1).copy()), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(epochs):
        _run_epoch(model, loader, device, optimizer)


def _evaluate_np(model, X, y, device):
    dataset = TensorDataset(torch.from_numpy(X.transpose(0, 2, 1).copy()), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=32)
    _, acc = _run_epoch(model, loader, device)
    return acc


def run_finetune(args):
    device = pick_device(args.device)
    windows_by_label = load_own_windows(args.own_data_root)
    present = {label: windows for label, windows in windows_by_label.items() if windows}
    missing = [CLASS_NAMES[l] for l in PROJECT_CLASSES if l not in present]
    if missing:
        print(f"Aviso: sem nenhuma amostra propria para {missing}. O fine-tune vai ignorar essas classes "
              "(o cabecote da rede fica sem gradiente pra elas) -- veja o passo a passo de coleta em "
              "nn_classifier/README.md antes de confiar no modelo pra essas classes.")
    if not present:
        raise SystemExit(f"Nenhum dado proprio encontrado em {args.own_data_root}. Rode src/emgGestureTrainer.py primeiro.")

    X = np.stack([w for windows in present.values() for w in windows])
    y = np.array([label for label, windows in present.items() for _ in windows], dtype=np.int64)
    print(f"Dados proprios: {len(y)} janelas, classes presentes: {[CLASS_NAMES[l] for l in present]}")

    if args.k_folds > 1 and min(Counter(y).values()) >= args.k_folds:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=0)
        accs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model = build_model(NUM_CLASSES, pretrained_backbone=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
            _train_loop(model, X[train_idx], y[train_idx], device, args.epochs, args.lr)
            acc = _evaluate_np(model, X[val_idx], y[val_idx], device)
            print(f"  fold {fold + 1}/{args.k_folds}: acc={acc:.3f}")
            accs.append(acc)
        print(f"CV: acc media={np.mean(accs):.3f} desvio={np.std(accs):.3f}")
    else:
        print("Poucas amostras para k-fold estratificado nessa contagem de folds -- pulando CV, so treino final.")

    model = build_model(NUM_CLASSES, pretrained_backbone=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
    _train_loop(model, X, y, device, args.epochs, args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.backbone_state_dict(), "state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES}, args.out)
    print(f"Checkpoint final salvo em {args.out}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)

    ap.add_argument("--epn612-root", type=Path, default=Path(__file__).resolve().parent / "data" / "raw" / "epn612")
    ap.add_argument("--max-subjects", type=int, default=None,
                     help="limita o numero de sujeitos do EPN612 lidos (RAM/tempo) -- default: todos os 612")
    ap.add_argument("--extra-classes", action=argparse.BooleanOptionalAction, default=True,
                     help="inclui waveIn/waveOut/pinch no pretrain (nao existem no vocabulario do projeto)")

    ap.add_argument("--pretrained", type=Path, default=None, help="checkpoint do estagio pretrain")
    ap.add_argument("--own-data-root", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--freeze-backbone", action="store_true", help="so treina o cabecote linear no fine-tune")
    ap.add_argument("--k-folds", type=int, default=5)
    return ap


def main():
    args = build_arg_parser().parse_args()
    if args.out is None:
        args.out = DEFAULT_CHECKPOINT_DIR / (
            "epn612_pretrained.pt" if args.stage == "pretrain" else "finetuned.pt"
        )
    if args.stage == "pretrain":
        run_pretrain(args)
    else:
        if args.pretrained is None:
            print("Aviso: --pretrained nao informado, treinando do zero (sem transfer learning do EPN612).")
        run_finetune(args)


if __name__ == "__main__":
    main()
