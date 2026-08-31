"""Baixa o dataset EMG-EPN612 (Zenodo, CC-BY 4.0, acesso aberto) e extrai.

O dataset tem ~5.5GB compactado, gravado com o mesmo hardware deste projeto
(1 Myo armband, 8 canais, 200Hz) -- ver docs/plans/emg-nn-pretrain-finetune/PLAN.md.

Uso:
    python -m nn_classifier.data.download_epn612 [--out-dir DIR] [--record-id ID]
"""

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_RECORD_ID = "4421500"  # EMG-EPN-612 Dataset, v2.1 (10.5281/zenodo.4421500)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "raw" / "epn612"
CHUNK_SIZE = 1024 * 1024


def _zenodo_file_info(record_id: str):
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        record = json.load(resp)
    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"Registro Zenodo {record_id} nao tem arquivos listados.")
    # O EPN612 e publicado como um unico zip; pega o maior arquivo por seguranca.
    biggest = max(files, key=lambda f: f.get("size", 0))
    checksum = biggest.get("checksum", "")  # formato "md5:<hash>"
    md5 = checksum.split(":", 1)[1] if ":" in checksum else None
    return biggest["links"]["self"], biggest["key"], md5


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {read / 1e6:.0f}MB / {total / 1e6:.0f}MB", end="", flush=True)
    print()


def _md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(out_dir: Path = DEFAULT_OUT_DIR, record_id: str = DEFAULT_RECORD_ID, keep_zip: bool = False):
    out_dir = Path(out_dir)
    url, filename, expected_md5 = _zenodo_file_info(record_id)
    zip_path = out_dir / filename

    if not zip_path.exists():
        _download(url, zip_path)
    else:
        print(f"{zip_path} ja existe, pulando download.")

    if expected_md5:
        actual_md5 = _md5sum(zip_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 nao bate: esperado {expected_md5}, obtido {actual_md5}. Baixe de novo.")
        print(f"MD5 confirmado: {actual_md5}")

    print(f"Extraindo em {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    if not keep_zip:
        zip_path.unlink()

    print(f"Pronto. Dados extraidos em: {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--record-id", type=str, default=DEFAULT_RECORD_ID)
    ap.add_argument("--keep-zip", action="store_true", help="Nao apaga o zip apos extrair.")
    args = ap.parse_args()

    try:
        download_and_extract(args.out_dir, args.record_id, args.keep_zip)
    except Exception as exc:
        print(f"Falhou: {exc}", file=sys.stderr)
        sys.exit(1)
