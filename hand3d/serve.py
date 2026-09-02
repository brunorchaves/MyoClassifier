"""
serve.py — serve a pagina da mao 3D.

    python serve.py            # porta 8010
    python serve.py 9000       # outra porta

Precisa de servidor: abrir o web/index.html direto do disco nao funciona,
porque o navegador bloqueia a leitura do .fbx local (CORS em file://).
"""

import base64
import http.server
import json
import os
import socketserver
import struct
import sys
import threading
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "web")
GESTOS = os.path.join(AQUI, "gestos.json")
PORTA = int(sys.argv[1]) if len(sys.argv) > 1 else 8010

# Aba de treinamento (dashboard.html -> treino.js): grava direto no mesmo
# dataset que nn_classifier/data/own_dataset.py le. Mesma resolucao de
# caminho que DEFAULT_DATA_DIR ali (<repo>/src/data), sem importar aquele
# modulo -- hand3d e nn_classifier ficam desacoplados de proposito.
DADOS_DIR = os.path.join(os.path.dirname(AQUI), "src", "data")
# Mantido em sincronia manualmente com PROJECT_CLASSES em
# nn_classifier/data/label_map.py -- sao os unicos gestos com pose/nome
# definidos (gestos.json + o "Repouso" fixo da classe 0 no treino.js).
CLASSES_VALIDAS = {0, 1, 2, 3, 4}
MAX_CORPO = 8_000_000            # ~8MB, bem acima de uma sessao de 8s (~400 amostras)
MAX_AMOSTRAS_POR_REQUISICAO = 20000

# Calibracao da orientacao (web/calibra.html): cada amostra casa a
# orientacao REAL da mao, medida pelo MediaPipe na webcam, com o quaternion
# cru da IMU do Myo no mesmo instante. Uma linha JSON por amostra, para
# poder ler/derivar a transformacao fora do navegador.
CALIB_DIR = os.path.join(AQUI, "calib")
CALIB_ARQ = os.path.join(CALIB_DIR, "amostras.jsonl")
# Frames da webcam (com o esqueleto do MediaPipe desenhado por cima) salvos
# durante as varreduras. Servem pra AUDITAR a captura depois: olhando as
# imagens da-se pra ver se o punho dobrou, se a mao saiu do quadro ou se a
# pose feita nao era a pedida — antes de confiar nos numeros.
CALIB_FRAMES = os.path.join(CALIB_DIR, "frames")
# A vertical do mundo do Myo, medida pela gravidade (acelerometro com o
# braco parado). E a UNICA coisa que a orientacao precisa alinhar: casada a
# vertical, todo movimento sai certo, e o que sobra e o heading — que um IMU
# sem bussola nao tem como saber e a tecla espaco do hand.js ja zera.
CALIB_VERTICAL = os.path.join(CALIB_DIR, "vertical.json")


def _contar_linhas(caminho):
    """Quantas amostras de calibracao ja foram gravadas (linhas do .jsonl)."""
    try:
        with open(caminho, encoding="utf-8") as f:
            return sum(1 for linha in f if linha.strip())
    except OSError:
        return 0


def _contar_amostras(classe):
    """Total de amostras hoje em vals{classe}.dat -- sempre relido do
    disco (nunca um contador em memoria), 16 bytes = 8 canais x uint16."""
    caminho = os.path.join(DADOS_DIR, "vals%d.dat" % classe)
    try:
        return os.path.getsize(caminho) // 16
    except OSError:
        return 0


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=RAIZ, **kw)

    def end_headers(self):
        # sem cache: F5 sempre pega a ultima versao
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_GET(self):
        # gestos.json fica em hand3d/ (fonte unica com bridge.py e
        # desktop.py), fora da raiz estatica hand3d/web/ — serve-o na mao.
        if self.path == "/gestos.json":
            try:
                with open(GESTOS, "rb") as f:
                    corpo = f.read()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
            return
        if self.path == "/api/calibra-status":
            self._responder_json(200, {"total": _contar_linhas(CALIB_ARQ)})
            return
        if self.path == "/api/dataset-status":
            classes = [
                {"classe": c, "amostras": _contar_amostras(c)}
                for c in sorted(CLASSES_VALIDAS)
            ]
            self._responder_json(200, {"classes": classes})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/gravar":
            self._gravar()
            return
        if self.path == "/api/calibra":
            self._calibra()
            return
        if self.path == "/api/calibra-frame":
            self._calibra_frame()
            return
        if self.path == "/api/calibra-vertical":
            self._calibra_vertical()
            return
        self.send_error(404)

    def _calibra_vertical(self):
        # {vertical: [x,y,z], espalhamento_graus, n, ...} -> calib/vertical.json
        # Sobrescreve: e uma medida unica, nao um historico.
        tam = self.headers.get("Content-Length")
        if tam is None or not tam.isdigit() or int(tam) > 100000:
            self._responder_json(413, {"erro": "corpo ausente ou grande demais"})
            return
        try:
            d = json.loads(self.rfile.read(int(tam)))
        except ValueError:
            self._responder_json(400, {"erro": "json invalido"})
            return
        v = d.get("vertical")
        if (not isinstance(v, list) or len(v) != 3
                or not all(isinstance(x, (int, float)) for x in v)):
            self._responder_json(400, {"erro": "esperava vertical [x,y,z]"})
            return
        norma = sum(float(x) ** 2 for x in v) ** 0.5
        if not 0.9 <= norma <= 1.1:
            self._responder_json(400, {
                "erro": "vertical nao normalizada (|v|=%.3f)" % norma
            })
            return

        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(CALIB_VERTICAL, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        self._responder_json(200, {"ok": True, "arquivo": "vertical.json"})

    def _calibra_frame(self):
        # {nome: "sweep_roll-0007", jpeg_base64: "..."} -> calib/frames/<nome>.jpg
        tam = self.headers.get("Content-Length")
        if tam is None or not tam.isdigit() or int(tam) > MAX_CORPO:
            self._responder_json(413, {"erro": "corpo ausente ou maior que 8MB"})
            return
        try:
            d = json.loads(self.rfile.read(int(tam)))
        except ValueError:
            self._responder_json(400, {"erro": "json invalido"})
            return
        nome = d.get("nome")
        b64 = d.get("jpeg_base64")
        if not isinstance(nome, str) or not isinstance(b64, str):
            self._responder_json(400, {"erro": "esperava nome e jpeg_base64"})
            return
        # nome vem do navegador: nao deixa escapar da pasta de frames
        seguro = "".join(c for c in nome if c.isalnum() or c in "-_")[:80]
        if not seguro:
            self._responder_json(400, {"erro": "nome invalido"})
            return
        if b64.startswith("data:"):                 # aceita data URL inteira
            b64 = b64.split(",", 1)[-1]
        try:
            bruto = base64.b64decode(b64, validate=True)
        except Exception:
            self._responder_json(400, {"erro": "base64 invalido"})
            return

        os.makedirs(CALIB_FRAMES, exist_ok=True)
        with open(os.path.join(CALIB_FRAMES, seguro + ".jpg"), "wb") as f:
            f.write(bruto)
        self._responder_json(200, {"ok": True, "arquivo": seguro + ".jpg"})

    def _calibra(self):
        # Uma amostra de calibracao, ou uma LISTA delas (uma varredura manda
        # ~80 de uma vez, em lote: 80 POSTs num servidor single-thread
        # atrapalhariam a propria captura). Guarda o corpo como veio, uma
        # linha JSON por amostra, sem impor esquema alem dos dois lados do
        # casamento (IMU e mao real) — quem resolve le isso depois.
        tam = self.headers.get("Content-Length")
        if tam is None or not tam.isdigit() or int(tam) > MAX_CORPO:
            self._responder_json(413, {"erro": "corpo ausente ou maior que 8MB"})
            return
        try:
            dados = json.loads(self.rfile.read(int(tam)))
        except ValueError:
            self._responder_json(400, {"erro": "json invalido"})
            return

        lote = dados if isinstance(dados, list) else [dados]
        if not lote:
            self._responder_json(400, {"erro": "lote vazio"})
            return
        for i, amostra in enumerate(lote):
            if not isinstance(amostra, dict):
                self._responder_json(400, {"erro": "amostra %d nao e objeto" % i})
                return
            faltando = [c for c in ("quat_imu", "r_real") if c not in amostra]
            if faltando:
                self._responder_json(400, {
                    "erro": "amostra %d sem: %s" % (i, ", ".join(faltando))
                })
                return

        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(CALIB_ARQ, "a", encoding="utf-8") as f:
            for amostra in lote:
                f.write(json.dumps(amostra, ensure_ascii=False) + "\n")

        self._responder_json(200, {
            "ok": True, "gravadas": len(lote), "total": _contar_linhas(CALIB_ARQ),
        })

    def _gravar(self):
        # aba de treinamento (web/treino.js): acumula os lotes de EMG que
        # ja chegam pro navegador via bridge.py e manda tudo de uma vez pra
        # ca no fim da sessao de gravacao. Nada e gravado parcialmente: ou
        # o corpo inteiro passa nas validacoes, ou nada e escrito.
        tam = self.headers.get("Content-Length")
        if tam is None or not tam.isdigit() or int(tam) > MAX_CORPO:
            self._responder_json(413, {"erro": "corpo ausente ou maior que 8MB"})
            return
        corpo = self.rfile.read(int(tam))
        try:
            dados = json.loads(corpo)
        except ValueError:
            self._responder_json(400, {"erro": "json invalido"})
            return

        classe = dados.get("classe")
        if not isinstance(classe, int) or classe not in CLASSES_VALIDAS:
            self._responder_json(400, {
                "erro": "classe invalida: use 0(rest),1(open),2(fist),3(spock),4(pointing)"
            })
            return

        amostras = dados.get("amostras")
        if not isinstance(amostras, list) or not amostras:
            self._responder_json(400, {"erro": "amostras vazio ou ausente"})
            return
        if len(amostras) > MAX_AMOSTRAS_POR_REQUISICAO:
            self._responder_json(400, {
                "erro": "muitas amostras numa unica requisicao (max %d)" % MAX_AMOSTRAS_POR_REQUISICAO
            })
            return

        empacotado = bytearray()
        for i, amostra in enumerate(amostras):
            if (not isinstance(amostra, list) or len(amostra) != 8
                    or not all(isinstance(v, int) and 0 <= v <= 65535 for v in amostra)):
                self._responder_json(400, {
                    "erro": "amostra invalida na posicao %d: precisa ser 8 inteiros 0-65535" % i
                })
                return
            empacotado += struct.pack("<8H", *amostra)

        os.makedirs(DADOS_DIR, exist_ok=True)
        caminho = os.path.join(DADOS_DIR, "vals%d.dat" % classe)
        with open(caminho, "ab") as f:
            f.write(empacotado)

        self._responder_json(200, {
            "ok": True,
            "classe": classe,
            "gravadas": len(amostras),
            "total_amostras": _contar_amostras(classe),
        })

    def _responder_json(self, status, obj):
        corpo = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, fmt, *args):
        if ".fbx" in self.path or self.path in ("/", "/index.html"):
            sys.stderr.write("  %s\n" % (fmt % args))


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORTA), Handler) as s:
        url = "http://127.0.0.1:%d/" % PORTA
        print("mao 3D em %s" % url)
        print("ctrl+c para parar\n")
        if "--sem-navegador" not in sys.argv:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            print("\nate mais.")


if __name__ == "__main__":
    main()
