"""
bridge.py — substitui o Unity como destino do classificador do Myo.

O que o Unity fazia, em duas partes:

  1. myListener.cs   — servidor TCP na porta 25001; o Python mandava
                       "roll,-yaw,pitch" e ele aplicava em transform.rotation.
  2. handController.cs — Animator com blend tree (Grip x Trigger); o Python
                       escolhia a pose EMULANDO TECLAS 1-4 com a lib keyboard.

Esta ponte faz as duas coisas e entrega para o navegador:

  TCP  :25001  <- fala o MESMO protocolo do myListener.cs, inclusive o eco
                  que o seu myoControlsHand.py espera no sock.recv().
                  Então ele roda SEM ALTERAÇÃO para a orientação.
  WS   :8765   -> transmite JSON para a pagina web/index.html.

Para o gesto chegar junto usando o SEU src/myoControlsHand.py (e
aposentar o truque do teclado), basta uma linha nele:

    data_str = f"{roll},{-yaw},{pitch}"                  # antes
    data_str = f"{roll},{-yaw},{pitch},{int(pose)}"      # depois

O quarto campo é opcional: sem ele a mão só gira, e o gesto continua
podendo ser escolhido pelos botões da página.

Uso (de dentro de hand3d/):
    python run.py                      # sobe ponte + Myo + navegador
    python bridge.py                   # so a ponte, esperando o Myo
    python bridge.py --sim             # sem hardware: inventa movimento
    python bridge.py --port 8765 --tcp 25001

Só biblioteca padrão — nada de pip install.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import random
import socket
import struct
import sys
import threading
import time

# GUID do RFC 6455, secao 1.3 — confira contra o vetor de teste da
# propria RFC antes de mexer: uma letra fora do lugar faz o navegador
# recusar a conexao com code 1006 e nenhuma mensagem util.
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

AQUI = os.path.dirname(os.path.abspath(__file__))


def carregar_gestos():
    """classe do classificador -> nome do clipe no FBX.

    Fonte unica em gestos.json (hand3d/PLANO-desktop.md, Passo 5): antes disto
    o mapa vivia duplicado aqui e em web/hand.js (POSES). desktop.py le o
    mesmo arquivo; hand.js busca por fetch.
    """
    caminho = os.path.join(AQUI, "gestos.json")
    with open(caminho, encoding="utf-8") as f:
        d = json.load(f)
    padrao = d["classe_desconhecida"]
    # extras (poses desenhadas por osso, sem clipe no FBX) tambem entram no
    # mapa classe->nome — nunca vem do classificador de verdade (sem dado
    # de treino), mas nao faz mal a ponte saber o nome se algum dia vier
    mapa = {item["classe"]: item["clip"] for item in d["ordem"] + d.get("extras", [])}
    return mapa, padrao


GESTOS, GESTO_PADRAO = carregar_gestos()

# estado compartilhado, publicado para todos os navegadores conectados
ESTADO = {
    "t": 0.0,
    "gesture": 0,
    "name": "Relaxed",
    "euler": [0.0, 0.0, 0.0],   # roll, pitch, yaw em graus
    # quaternion CRU da IMU (w,x,y,z), sem passar por Euler. O euler acima
    # e derivado dele no feed.py e serve pros paineis/telemetria; quem quer
    # orientar a mao 3D deve usar este quat: Euler tem ordem/permutacao/
    # sinal ambiguos e foi exatamente o que fez a mao sair na diagonal
    # (ver web/calibra.html e hand.js).
    "quat": [1.0, 0.0, 0.0, 0.0],
    # acelerometro normalizado, no referencial do CORPO. Parado, e a
    # gravidade: R_imu * acc da a VERTICAL do mundo do Myo, que e o que a
    # calibracao da orientacao precisa (ver web/calibra.html).
    "acc": [0.0, 0.0, 1.0],
    "rms": [0.0] * 8,
    "fs": 0.0,                  # taxa de amostragem medida no bracelete
    "src": "—",
}

# Amostras de EMG esperando envio. Diferente do resto do ESTADO, isto é
# uma FILA: cada transmissão leva o lote e o esvazia. Se fosse tratado
# como instantâneo, o navegador receberia a mesma amostra várias vezes
# (transmissão a 50 Hz, Myo a ~50 Hz) e o traço sairia com degraus.
PENDENTES = []
MAX_PENDENTES = 500            # ~10 s de folga; acima disso descarta o antigo
LOCK = threading.Lock()


def quat_de_euler(roll, pitch, yaw):
    """(roll,pitch,yaw) em GRAUS -> quaternion (w,x,y,z).

    Inversa exata do euler_de_quaternion() do feed.py: mesma convencao de
    aeronautica R = Rz(yaw)*Ry(pitch)*Rx(roll). Usada pelos caminhos que
    so tem Euler (modo --sim e o protocolo de texto antigo do
    myoControlsHand.py) para que eles tambem publiquem ESTADO["quat"] —
    a mao 3D orienta pelo quat.
    """
    r, p, y = (math.radians(roll) / 2, math.radians(pitch) / 2, math.radians(yaw) / 2)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


# ----------------------------------------------------------------------
# WebSocket mínimo (só o que precisa: handshake + frames de texto)
# ----------------------------------------------------------------------

class WSClient:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.alive = True

    def handshake(self):
        data = b""
        self.conn.settimeout(5)
        while b"\r\n\r\n" not in data:
            chunk = self.conn.recv(2048)
            if not chunk:
                return False
            data += chunk
            if len(data) > 65536:
                return False
        key = None
        for line in data.split(b"\r\n"):
            if line.lower().startswith(b"sec-websocket-key:"):
                key = line.split(b":", 1)[1].strip().decode()
        if not key:
            return False
        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.conn.sendall(
            ("HTTP/1.1 101 Switching Protocols\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Accept: " + accept + "\r\n\r\n").encode())
        self.conn.settimeout(None)
        return True

    def send_text(self, s):
        payload = s.encode("utf-8")
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x81, n)
        elif n < 65536:
            head = struct.pack("!BBH", 0x81, 126, n)
        else:
            head = struct.pack("!BBQ", 0x81, 127, n)
        self.conn.sendall(head + payload)

    def drain(self):
        """Lê e descarta o que o navegador manda; detecta o fechamento."""
        try:
            while self.alive:
                b = self.conn.recv(4096)
                if not b:
                    break
                if b and (b[0] & 0x0F) == 0x8:      # close
                    break
        except Exception:
            pass
        self.alive = False


CLIENTES = []
CLI_LOCK = threading.Lock()


_ULTIMO_AVISO = [0.0]


def avisar(msg):
    """Avisa no console, no maximo uma vez a cada 2 s (nao enche o log)."""
    agora = time.time()
    if agora - _ULTIMO_AVISO[0] > 2.0:
        _ULTIMO_AVISO[0] = agora
        print("  ? %s" % msg, flush=True)


class PortaOcupada(Exception):
    pass


def abrir_porta(host, port, nome):
    """Abre a porta recusando subir se ja tem alguem ali.

    No Windows o SO_REUSEADDR permite DOIS processos ligarem na mesma
    porta, e as conexoes se dividem entre eles de forma imprevisivel —
    duas pontes rodando dao um comportamento impossivel de depurar. Por
    isso aqui a gente testa antes e morre com mensagem clara.
    """
    try:
        t = socket.create_connection((host if host != "0.0.0.0" else "127.0.0.1", port),
                                     timeout=0.4)
        t.close()
        print("ERRO: a porta %d (%s) ja esta ocupada.\n"
              "      Provavelmente outra ponte esta rodando. Feche a outra\n"
              "      (ou use --port / --tcp para escolher outra porta)."
              % (port, nome), file=sys.stderr)
        sys.exit(1)
    except OSError:
        pass                                     # ninguem atendeu: porta livre
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.bind((host, port))
    except OSError as e:
        print("ERRO: nao consegui abrir a porta %d (%s): %s" % (port, nome, e),
              file=sys.stderr)
        sys.exit(1)
    srv.listen(8)
    return srv


def ws_server(srv, port):
    print("  WebSocket  ws://127.0.0.1:%d   (a pagina conecta aqui)" % port, flush=True)
    while True:
        try:
            conn, addr = srv.accept()
        except OSError as e:
            avisar("accept do WebSocket falhou (%s), tentando de novo" % e)
            time.sleep(0.2)
            continue
        c = WSClient(conn, addr)
        try:
            ok = c.handshake()
        except Exception as e:
            avisar("handshake recusado: %s" % e)
            ok = False
        if not ok:
            try:
                conn.close()
            except Exception:
                pass
            continue
        with CLI_LOCK:
            CLIENTES.append(c)
        print("  + navegador conectado (%d ativo(s))" % len(CLIENTES), flush=True)
        threading.Thread(target=c.drain, daemon=True).start()


def broadcast():
    """Publica o estado a ~50 Hz para quem estiver ouvindo."""
    while True:
        time.sleep(0.02)
        with LOCK:
            pacote = dict(ESTADO)
            # o lote sai uma única vez, e é o mesmo para todos os clientes
            # desta rodada — ninguém perde nem repete amostra
            if PENDENTES:
                pacote["emg"] = list(PENDENTES)
                del PENDENTES[:]
            msg = json.dumps(pacote, separators=(",", ":"))
        mortos = []
        with CLI_LOCK:
            alvos = list(CLIENTES)
        for c in alvos:
            if not c.alive:
                mortos.append(c)
                continue
            try:
                c.send_text(msg)
            except Exception:
                c.alive = False
                mortos.append(c)
        if mortos:
            with CLI_LOCK:
                for c in mortos:
                    if c in CLIENTES:
                        CLIENTES.remove(c)
                    try:
                        c.conn.close()
                    except Exception:
                        pass
            print("  - navegador saiu (%d ativo(s))" % len(CLIENTES), flush=True)


# ----------------------------------------------------------------------
# Lado do Myo: mesmo protocolo TCP que o Unity usava
# ----------------------------------------------------------------------

def tcp_server(srv, port):
    print("  TCP        :%d   (myoControlsHand.py conecta aqui, "
          "igual ao myListener.cs)" % port, flush=True)
    while True:
        try:
            conn, addr = srv.accept()
        except OSError as e:
            avisar("accept do TCP falhou (%s), tentando de novo" % e)
            time.sleep(0.2)
            continue
        print("  + Myo conectado de %s:%s" % addr, flush=True)
        try:
            atender_myo(conn)
        except Exception as e:
            print("  ! erro no TCP: %s" % e, flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("  - Myo desconectou", flush=True)


def atender_myo(conn):
    resto = ""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return
        # o myListener.cs devolvia o buffer; o script Python espera esse eco
        conn.sendall(chunk)
        texto = resto + chunk.decode("utf-8", "replace")
        resto = ""
        if "\n" in texto:
            # o myo_feed.py manda uma mensagem por linha
            partes = texto.replace("\r", "").split("\n")
            resto = partes.pop()             # pode ser uma linha incompleta
        else:
            # o myoControlsHand.py manda um pacote por sendall, sem \n
            partes = [texto]
        for linha in partes:
            if not linha.strip():
                continue
            if aplicar(linha):
                continue
            # ultimo recurso: varios valores separados por espaco no mesmo pacote
            entendeu = False
            for pedaco in linha.split():
                entendeu = aplicar(pedaco) or entendeu
            if not entendeu:
                avisar("nao entendi esta linha, ignorada: %s" % linha[:70])


def aplicar(s):
    """Interpreta uma mensagem. Devolve True se entendeu.

    Aceita '(r,y,p)', 'r,y,p', 'r,y,p,gesto' ou um JSON completo.
    """
    s = s.strip()
    if not s:
        return True
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except Exception as e:
            avisar("JSON invalido, ignorado (%s): %s" % (e, s[:60]))
            return False
        with LOCK:
            if "euler" in d:
                ESTADO["euler"] = [float(x) for x in d["euler"]][:3]
            if "quat" in d and isinstance(d["quat"], list) and len(d["quat"]) == 4:
                ESTADO["quat"] = [float(x) for x in d["quat"]]
            if "acc" in d and isinstance(d["acc"], list) and len(d["acc"]) == 3:
                ESTADO["acc"] = [float(x) for x in d["acc"]]
            if "gesture" in d:
                g = int(d["gesture"])
                ESTADO["gesture"] = g
                ESTADO["name"] = GESTOS.get(g, GESTO_PADRAO)
            if "rms" in d:
                ESTADO["rms"] = [float(x) for x in d["rms"]][:8]
            if "fs" in d:
                ESTADO["fs"] = float(d["fs"])
            if "emg" in d and isinstance(d["emg"], list):
                for amostra in d["emg"]:
                    if isinstance(amostra, list) and len(amostra) == 8:
                        PENDENTES.append([int(v) for v in amostra])
                if len(PENDENTES) > MAX_PENDENTES:
                    del PENDENTES[:-MAX_PENDENTES]
            ESTADO["src"] = "myo"
            ESTADO["t"] = time.time()
        return True
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    partes = s.split(",")
    if len(partes) < 3:
        return False
    try:
        vals = [float(p) for p in partes[:4]]
    except ValueError:
        return False
    with LOCK:
        # o script manda "roll, -yaw, pitch" nessa ordem
        ESTADO["euler"] = [vals[0], vals[2], vals[1]]
        # este caminho antigo so tem Euler; converte pra quat porque a mao
        # 3D orienta por ele (ver ESTADO["quat"])
        ESTADO["quat"] = quat_de_euler(*ESTADO["euler"])
        if len(vals) >= 4:
            g = int(vals[3])
            ESTADO["gesture"] = g
            ESTADO["name"] = GESTOS.get(g, GESTO_PADRAO)
        ESTADO["src"] = "myo"
        ESTADO["t"] = time.time()
    return True


# ----------------------------------------------------------------------
# Modo simulado: dá pra ensaiar sem o bracelete
# ----------------------------------------------------------------------

def simulador():
    print("  SIM        inventando gesto e orientação (sem hardware)", flush=True)
    ordem = [1, 2, 3, 4]
    i = 0
    t0 = time.time()
    prox = t0 + 2.2
    while True:
        time.sleep(0.03)
        t = time.time() - t0
        if time.time() > prox:
            i = (i + 1) % len(ordem)
            prox = time.time() + 2.2
        g = ordem[i]
        base = [0.05] * 8
        if g == 2:
            base = [0.9, 0.85, 0.6, 0.35, 0.45, 0.7, 0.88, 0.95]
        elif g == 3:
            base = [0.3, 0.4, 0.68, 0.9, 0.8, 0.5, 0.32, 0.26]
        elif g == 4:
            base = [0.26, 0.18, 0.16, 0.52, 0.94, 0.98, 0.62, 0.34]
        with LOCK:
            ESTADO["gesture"] = g
            ESTADO["name"] = GESTOS.get(g, GESTO_PADRAO)
            ESTADO["euler"] = [22 * math.sin(t * 0.7),
                               14 * math.sin(t * 0.5 + 1.0),
                               30 * math.sin(t * 0.33)]
            # o mesmo estado tambem como quaternion: a mao 3D usa o quat
            # (ver ESTADO["quat"] acima), entao sem isto o modo --sim
            # ficaria congelado depois que hand.js passou a preferi-lo.
            ESTADO["quat"] = quat_de_euler(*ESTADO["euler"])
            ESTADO["rms"] = [min(1.0, max(0.0, v + random.uniform(-0.05, 0.05)))
                             for v in base]
            ESTADO["fs"] = 50.0
            ESTADO["src"] = "sim"
            ESTADO["t"] = time.time()
            # duas amostras por volta ~= 50 Hz, na escala crua do Myo
            for _ in range(2):
                PENDENTES.append([int(max(0, 300 * v * random.uniform(0.55, 1.45)
                                          + random.uniform(0, 22))) for v in base])
            if len(PENDENTES) > MAX_PENDENTES:
                del PENDENTES[:-MAX_PENDENTES]


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ponte Myo -> navegador (sem Unity)")
    ap.add_argument("--port", type=int, default=8765, help="porta do WebSocket")
    ap.add_argument("--tcp", type=int, default=25001, help="porta TCP do Myo")
    ap.add_argument("--sim", action="store_true",
                    help="sem hardware: gera gesto e orientação de teste")
    a = ap.parse_args()

    print("ponte Myo -> navegador")
    # abre tudo antes de criar thread: assim um erro de porta sai do processo
    try:
        ws_srv = abrir_porta("127.0.0.1", a.port, "WebSocket")
        tcp_srv = None if a.sim else abrir_porta("0.0.0.0", a.tcp, "TCP do Myo")
    except PortaOcupada as e:
        print("ERRO: %s" % e, file=sys.stderr)
        return 1

    threading.Thread(target=ws_server, args=(ws_srv, a.port), daemon=True).start()
    threading.Thread(target=broadcast, daemon=True).start()
    if a.sim:
        threading.Thread(target=simulador, daemon=True).start()
    else:
        threading.Thread(target=tcp_server, args=(tcp_srv, a.tcp), daemon=True).start()
    print("\nabra a página da mão 3D. ctrl+c para parar.\n", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nate mais.")


if __name__ == "__main__":
    main()
