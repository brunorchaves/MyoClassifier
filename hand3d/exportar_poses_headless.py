"""Ferramenta descartavel: roda a exportacao (window.exportarPoses, em hand.js)
num Chromium headless via playwright, e grava web/model/hand.cache.npz.

Isto substitui o clique manual no botao "exportar poses" da pagina — util so
para gerar o cache uma vez. Nao faz parte do app final (nem do runtime do
modo desktop, que so LE o .npz).
"""
import base64
import os
import subprocess
import sys
import time

import numpy as np
from playwright.sync_api import sync_playwright

AQUI = os.path.dirname(os.path.abspath(__file__))
PORTA = 8011
URL = "http://127.0.0.1:%d/" % PORTA
CACHE = os.path.join(AQUI, "web", "model", "hand.cache.npz")


def b64_f32(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


def main():
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    servidor = subprocess.Popen(
        [sys.executable, "serve.py", str(PORTA), "--sem-navegador"],
        cwd=AQUI, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        print("esperando o servidor da pagina subir em %s ..." % URL)
        for _ in range(50):
            if servidor.poll() is not None:
                raise SystemExit("serve.py caiu: %s" % servidor.stdout.read())
            try:
                import urllib.request
                urllib.request.urlopen(URL, timeout=1).read(1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("serve.py nao respondeu")

        with sync_playwright() as p:
            navegador = p.chromium.launch(headless=True)
            pagina = navegador.new_page()
            erros_console = []
            pagina.on("console", lambda msg: erros_console.append(msg.text) if msg.type == "error" else None)
            pagina.goto(URL)
            print("pagina carregada, esperando o modelo e as animacoes...")
            pagina.wait_for_function(
                "() => window.exportarPoses && document.getElementById('l-modelo')"
                ".textContent.indexOf('/') > 0 "
                "&& document.getElementById('l-modelo').textContent !== '0 / 0'",
                timeout=20000,
            )
            ossos_poses = pagina.eval_on_selector("#l-modelo", "el => el.textContent")
            print("ossos / clipes no FBX:", ossos_poses)
            erro_txt = pagina.eval_on_selector(
                "#erro", "el => el.style.display !== 'none' ? el.innerText : ''"
            )
            if erro_txt:
                print("AVISO exibido na pagina:", erro_txt)

            print("exportando poses (skinning por vertice em JS)...")
            t0 = time.time()
            out = pagina.evaluate("() => window.exportarPoses()")
            print("  levou %.1fs" % (time.time() - t0))
            if erros_console:
                print("console.error da pagina:", erros_console[:5])
            navegador.close()

        if out.get("erro"):
            raise SystemExit("exportarPoses() falhou: %s" % out["erro"])

        n = out["n_vertices"]
        indices = np.frombuffer(base64.b64decode(out["indices_u32"]), dtype=np.uint32) \
            if out.get("indices_u32") else None
        print("vertices:", n, "indices:", None if indices is None else indices.size)

        dados = {"n_vertices": np.int64(n), "max_dim": np.float64(out["max_dim"])}
        if indices is not None:
            dados["indices"] = indices
        nomes = sorted(out["poses"].keys())
        if not nomes:
            raise SystemExit("exportarPoses() nao devolveu poses — clipes ausentes no FBX?")
        dados["poses"] = np.array(nomes)
        for nome in nomes:
            p = out["poses"][nome]
            pos = b64_f32(p["position_f32"]).reshape(-1, 3)
            nrm = b64_f32(p["normal_f32"]).reshape(-1, 3)
            if pos.shape[0] != n or nrm.shape[0] != n:
                raise SystemExit("pose %s com tamanho inconsistente: pos=%s nrm=%s n=%s"
                                  % (nome, pos.shape, nrm.shape, n))
            dados["pos_" + nome] = pos.astype(np.float32)
            dados["nrm_" + nome] = nrm.astype(np.float32)

        np.savez_compressed(CACHE, **dados)
        tam = os.path.getsize(CACHE) / 1e6
        print("gravado %s (%.1f MB) — poses: %s" % (CACHE, tam, ", ".join(nomes)))
    finally:
        servidor.terminate()
        try:
            servidor.wait(timeout=4)
        except Exception:
            servidor.kill()


if __name__ == "__main__":
    main()
