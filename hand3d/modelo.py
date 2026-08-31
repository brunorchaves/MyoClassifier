"""modelo.py — os dados da mao para o modo desktop (rota B).

O `assimp-py` (unico pacote instalavel so com pip nesta maquina — ver
PLANO-desktop.md, Passo 0) nao expoe ossos, pesos nem animacoes do
hand.fbx: so malha estatica. Ler o rig em Python virou inviavel sem
compilar nada, entao os dados vem prontos do lado que ja sabe fazer
skinning: o three.js, na pagina web.

`web/hand.js` tem uma funcao `exportarPoses()` que, para cada uma das 4
poses, calcula a posicao e a normal JA SKINADAS por vertice (a mesma
formula do skinning_vertex.glsl.js do three.js) e devolve tudo em base64.
O botao "exportar poses" na pagina grava isso num JSON; o utilitario
`exportar_poses_headless.py` faz o mesmo sem abrir navegador (usa
playwright — dependencia so de desenvolvimento, nao do app).

Este modulo so LE o resultado disso, cacheado em `web/model/hand.cache.npz`
(ignorado pelo git). O runtime do desktop.py nunca abre o FBX.
"""
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(AQUI, "web", "model", "hand.cache.npz")

MSG_SEM_CACHE = (
    "ERRO: nao achei %s.\n"
    "  Gere o cache uma vez:\n"
    "    - abra a pagina (python serve.py) e clique em 'exportar poses'\n"
    "      na lateral, depois mova o hand_poses.json baixado para o lugar\n"
    "      certo com: python modelo.py --importar caminho/hand_poses.json\n"
    "    - ou, sem abrir navegador: python exportar_poses_headless.py\n"
    "      (precisa de 'pip install playwright' + 'playwright install chromium')\n"
    "  Veja PLANO-desktop.md, Passo 0 e Passo 1."
)


class Modelo:
    """Posicao e normal por vertice, uma vez por pose. Sem ossos: a rota B
    perde a mistura em espaco de osso, ganha nao precisar ler o FBX."""

    def __init__(self, n_vertices, max_dim, ordem, posicoes, normais):
        self.n_vertices = n_vertices
        self.max_dim = max_dim        # maior dimensao da caixa (mesmo valor do hand.js), pra camera
        self.ordem = ordem            # nomes de clipe, na ordem dos pesos (0..3)
        self.posicoes = posicoes      # {nome: (N,3) float32}
        self.normais = normais        # {nome: (N,3) float32}


def _validar(nome, arr, n):
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.shape != (n, 3):
        raise SystemExit("ERRO: pose '%s' com forma %s, esperava (%d, 3)"
                          % (nome, arr.shape, n))
    if not np.isfinite(arr).all():
        raise SystemExit("ERRO: pose '%s' tem valor nao finito (NaN/Inf) — rig torto?" % nome)
    return arr


def carregar(ordem, caminho=CACHE):
    """ordem: lista de nomes de clipe (vem de gestos.json), na ordem dos
    pesos que o desktop.py vai usar."""
    if not os.path.exists(caminho):
        raise SystemExit(MSG_SEM_CACHE % caminho)
    d = np.load(caminho)
    n = int(d["n_vertices"])
    if n <= 0 or n % 3 != 0:
        raise SystemExit("ERRO: n_vertices=%d em %s nao e um numero de triangulos soltos "
                          "valido (deveria ser multiplo de 3, sem indice)" % (n, caminho))
    faltando = [p for p in ordem if ("pos_" + p) not in d.files or ("nrm_" + p) not in d.files]
    if faltando:
        raise SystemExit("ERRO: %s nao tem a(s) pose(s) %s — reexporte "
                          "(veja PLANO-desktop.md, Passo 0/1)" % (caminho, ", ".join(faltando)))
    posicoes = {p: _validar(p, d["pos_" + p], n) for p in ordem}
    normais = {p: _validar(p, d["nrm_" + p], n) for p in ordem}
    max_dim = float(d["max_dim"]) if "max_dim" in d.files else 1.0
    return Modelo(n, max_dim, list(ordem), posicoes, normais)


def _importar_json(caminho_json, destino=CACHE):
    """Converte o hand_poses.json baixado pelo botao 'exportar poses' da
    pagina no mesmo .npz que exportar_poses_headless.py gera."""
    import base64
    import json

    with open(caminho_json, encoding="utf-8") as f:
        out = json.load(f)
    if out.get("erro"):
        raise SystemExit("ERRO: o JSON exportado tem um erro registrado: %s" % out["erro"])
    n = int(out["n_vertices"])
    dados = {"n_vertices": np.int64(n), "max_dim": np.float64(out.get("max_dim") or 1.0)}
    for nome, p in out["poses"].items():
        pos = np.frombuffer(base64.b64decode(p["position_f32"]), dtype=np.float32).reshape(-1, 3)
        nrm = np.frombuffer(base64.b64decode(p["normal_f32"]), dtype=np.float32).reshape(-1, 3)
        dados["pos_" + nome] = pos
        dados["nrm_" + nome] = nrm
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    np.savez_compressed(destino, **dados)
    print("gravado %s (%.1f MB)" % (destino, os.path.getsize(destino) / 1e6))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--importar", metavar="hand_poses.json",
                     help="converte o JSON baixado do botao 'exportar poses' em hand.cache.npz")
    a = ap.parse_args()
    if a.importar:
        _importar_json(a.importar)
    else:
        import json
        with open(os.path.join(AQUI, "gestos.json"), encoding="utf-8") as f:
            ordem = [item["clip"] for item in json.load(f)["ordem"]]
        m = carregar(ordem)
        print("modelo ok: %d vertices, poses %s" % (m.n_vertices, m.ordem))
