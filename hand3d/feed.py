"""
feed.py — le o Myo e alimenta a ponte, sem pygame e sem keyboard.

Por que existe: o src/myoControlsHand.py depende de pygame, keyboard,
joblib e scikit-learn, e usa a busca por scan do pyomyo — que trava
quando o bracelete adormece no meio da varredura. Este alimentador faz
o essencial com o que ja vem no repo (pyomyo) mais numpy:

  * conecta pelo MAC (instantaneo). Na primeira vez descobre por scan
    e guarda em py/myo_mac.txt, para as proximas serem rapidas.
  * classifica com o MESMO 1-NN do seu script, sobre os MESMOS arquivos
    data/vals*.dat.
  * repete a MESMA suavizacao por voto de maioria (janela de 25).
  * manda para a ponte pela porta 25001, no formato que ela entende.

Nada de emular teclado: o gesto vai no proprio pacote.

Uso (de dentro de hand3d/):
    python run.py                           # sobe tudo de uma vez

    python bridge.py                        # ou, na mao: terminal 1
    python feed.py                          #              terminal 2
    python serve.py                         #              terminal 3

    python feed.py --scan                   # forca redescobrir o MAC
    python feed.py --dry                    # so imprime, nao manda
"""

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from collections import Counter, deque

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARQ_MAC = os.path.join(AQUI, "myo_mac.txt")
# hand3d/ fica dentro do MyoClassifier: o src/ e o vizinho de cima
REPO_PADRAO = os.path.dirname(AQUI)

# assinatura do servico do Myo no anuncio BLE (a mesma que o pyomyo procura)
ASSINATURA = (b"\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47"
              b"\xB9\xDE\x04\xA9\x01\x00\x06\xD5")

K = 15          # mesmo valor do seu script
SUBSAMPLE = 3
HIST = 25       # janela do voto de maioria


# ----------------------------------------------------------------------

def euler_de_quaternion(w, x, y, z):
    """Igual ao euler_from_quaternion do myoControlsHand.py."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n:
        w, x, y, z = w / n, x / n, y / n, z / n
    roll = int(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) * 180 / math.pi)
    t2 = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = int(math.asin(t2) * 180 / math.pi)
    yaw = int(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * 180 / math.pi)
    return roll, pitch, yaw


class Desembrulhador:
    """Acumula um angulo que sai de atan2/asin — presa a [-180,180] — num
    valor continuo, sem o salto quando o bracelete cruza esse limite (o
    mesmo problema que ja apareceu no Colibri). A cada amostra nova, em vez
    de guardar o angulo bruto, guarda o CAMINHO MAIS CURTO desde a amostra
    anterior e soma nesse acumulador — que pode passar de 360 ou cair abaixo
    de -360 sem se importar, porque so representa "quanto girou", nao "onde
    esta dentro de uma volta". mao3d/desktop.py e o navegador so leem esse
    valor continuo; nenhum dos dois precisa saber que o bracelete embrulha.
    """

    def __init__(self):
        self.anterior = None
        self.acumulado = 0.0

    def aplicar(self, bruto):
        if self.anterior is None:
            self.acumulado = float(bruto)
        else:
            delta = bruto - self.anterior
            delta = (delta + 180) % 360 - 180      # caminho mais curto
            self.acumulado += delta
        self.anterior = bruto
        return self.acumulado


class Classificador:
    """1-NN sobre os data/vals*.dat — o mesmo do seu Classifier."""

    def __init__(self, pasta):
        X, Y, self.por_classe = [], [], {}
        for i in range(10):
            p = os.path.join(pasta, "vals%d.dat" % i)
            if not os.path.exists(p):
                continue
            Xi = np.fromfile(p, dtype=np.uint16)
            if Xi.size < 8:
                continue
            Xi = Xi.reshape((-1, 8)).astype(np.float64)
            X.append(Xi)
            Y.append(np.full(Xi.shape[0], i, dtype=np.float64))
            self.por_classe[i] = Xi.shape[0]
        if not X:
            raise SystemExit("ERRO: nenhum dado de treino em %s" % pasta)
        self.X = np.vstack(X)
        self.Y = np.hstack(Y)

    def classificar(self, d):
        if self.X.shape[0] < K * SUBSAMPLE:
            return 0
        dist = ((self.X - np.asarray(d, dtype=np.float64)) ** 2).sum(1)
        return int(self.Y[int(dist.argmin())])


def descobrir_mac(bt, segundos=12):
    """Scan com prazo. O pyomyo faz isso sem timeout e trava se o
    bracelete dormir no meio — aqui a gente desiste e avisa."""
    bt.end_scan()
    for c in (0, 1, 2):
        bt.disconnect(c)
    print("  procurando o Myo (mexa o bracelete se ele estiver dormindo)...", flush=True)
    bt.discover()
    fim = time.time() + segundos
    achado = None
    while time.time() < fim:
        p = bt.recv_packet()
        if p is None:
            continue
        if p.payload.endswith(ASSINATURA):
            achado = list(p.payload[2:8])
            break
    bt.end_scan()
    return achado


def ler_mac_salvo():
    try:
        with open(ARQ_MAC) as f:
            v = [int(x) for x in f.read().strip().split(",")]
        return v if len(v) == 6 else None
    except Exception:
        return None


def salvar_mac(mac):
    try:
        with open(ARQ_MAC, "w") as f:
            f.write(",".join(str(x) for x in mac))
        print("  MAC guardado em %s (proximas conexoes serao instantaneas)"
              % os.path.basename(ARQ_MAC), flush=True)
    except Exception:
        pass


class Ponte:
    """Conexao com o myo_bridge.py, no mesmo protocolo TCP do Unity."""

    def __init__(self, host, porta, ativo=True):
        self.host, self.porta, self.ativo = host, porta, ativo
        self.sock = None
        self.avisou = False

    def garantir(self):
        if self.sock or not self.ativo:
            return self.sock is not None
        try:
            self.sock = socket.create_connection((self.host, self.porta), timeout=3)
            print("  ponte conectada em %s:%d" % (self.host, self.porta), flush=True)
            self.avisou = False
            return True
        except OSError:
            if not self.avisou:
                print("  ponte ainda nao esta no ar (%s:%d) — rode "
                      "'python bridge.py'" % (self.host, self.porta), flush=True)
                self.avisou = True
            return False

    def mandar(self, obj):
        if not self.garantir():
            return
        msg = json.dumps(obj, separators=(",", ":")) + "\n"
        try:
            self.sock.sendall(msg.encode())
            self.sock.recv(1024)          # a ponte devolve o eco, igual ao Unity
        except OSError:
            print("  ponte caiu; vou tentar reconectar", flush=True)
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Le o Myo e alimenta a ponte")
    ap.add_argument("--repo", default=REPO_PADRAO,
                    help="raiz do MyoClassifier (padrao: a pasta acima desta)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--porta", type=int, default=25001)
    ap.add_argument("--mac", default=None, help="6 numeros separados por virgula")
    ap.add_argument("--scan", action="store_true", help="ignora o MAC salvo e redescobre")
    ap.add_argument("--dry", action="store_true", help="so imprime, nao manda pra ponte")
    ap.add_argument("--espera", type=int, default=12,
                    help="segundos esperando o bracelete antes de desistir")
    a = ap.parse_args()

    src = os.path.join(a.repo, "src")
    if not os.path.isdir(src):
        raise SystemExit("ERRO: nao achei %s (use --repo)" % src)
    sys.path.insert(0, src)
    from pyomyo import Myo, emg_mode          # noqa: E402

    print("alimentador do Myo -> ponte")
    cls = Classificador(os.path.join(src, "data"))
    print("  treino: %s  (total %d amostras)"
          % (", ".join("classe %d=%d" % kv for kv in sorted(cls.por_classe.items())),
             cls.X.shape[0]), flush=True)
    vazias = [c for c in range(1, 5) if cls.por_classe.get(c, 0) == 0]
    if vazias:
        print("  AVISO: sem treino para a(s) classe(s) %s — esse(s) gesto(s) nunca "
              "vao sair.\n         Grave mais dados com src/emgGestureTrainer.py, ou remapeie GESTOS\n         no bridge.py (e POSES no web/hand.js)."
              % ", ".join(map(str, vazias)), flush=True)

    ponte = Ponte(a.host, a.porta, ativo=not a.dry)

    m = Myo(mode=emg_mode.PREPROCESSED)

    mac = None
    if a.mac:
        mac = [int(x) for x in a.mac.split(",")]
    elif not a.scan:
        mac = ler_mac_salvo()
    if mac is None:
        mac = descobrir_mac(m.bt)
        if mac is None:
            raise SystemExit("ERRO: nao achei o Myo no scan. Ele acorda quando voce\n"
                             "      mexe; tente de novo, ou passe --mac.")
        salvar_mac(mac)
    print("  MAC: %s" % ",".join(map(str, mac)), flush=True)

    estado = {"euler": (0, 0, 0), "quat": (1.0, 0.0, 0.0, 0.0),
              "acc": (0.0, 0.0, 1.0), "emg": (0,) * 8, "buf": [],
              "hist": deque([0] * HIST, HIST), "cnt": Counter([0] * HIST),
              "pose": None, "n": 0, "t0": time.time(), "ultimo_envio": 0.0}
    desembrulhar = [Desembrulhador(), Desembrulhador(), Desembrulhador()]  # roll, pitch, yaw

    def on_imu(quat, acc, gyro):
        bruto = euler_de_quaternion(*quat)
        estado["euler"] = tuple(desembrulhar[i].aplicar(bruto[i]) for i in range(3))
        # o quat CRU (w,x,y,z) tambem vai pra ponte: o pyomyo entrega int16
        # sem escala (unpack('10h'), ver src/pyomyo.py), entao normaliza aqui.
        # Quem orienta a mao 3D deve usar isto, nao o euler: a conversao pra
        # Euler introduz ordem/permutacao/sinal ambiguos (foi o que fez a mao
        # sair na diagonal). Ver hand3d/web/calibra.html.
        n = math.sqrt(sum(float(v) * float(v) for v in quat)) or 1.0
        estado["quat"] = tuple(float(v) / n for v in quat)
        # acelerometro, normalizado: com o braco parado isto E a gravidade,
        # no referencial do CORPO. Levado pro referencial do mundo pelo quat
        # (R_imu * acc), da a VERTICAL do mundo do Myo — que e a unica coisa
        # que a calibracao precisa achar (o heading e indeterminavel sem
        # bussola e a tecla espaco ja o zera). Ver hand3d/README.md.
        na = math.sqrt(sum(float(v) * float(v) for v in acc)) or 1.0
        estado["acc"] = tuple(float(v) / na for v in acc)

    def on_emg(emg, moving):
        estado["emg"] = emg
        estado["n"] += 1
        # guarda TODAS as amostras: quem desenha a forma de onda precisa
        # mandar só a última perderia 40% dos pontos (Myo ~50 Hz, envio ~30 Hz)
        estado["buf"].append([int(v) for v in emg])
        if len(estado["buf"]) > 400:            # se a ponte cair, não cresce sem fim
            del estado["buf"][:-400]
        y = cls.classificar(emg)
        # mesma suavizacao por voto de maioria do seu script
        estado["cnt"][estado["hist"][0]] -= 1
        estado["cnt"][y] += 1
        estado["hist"].append(y)
        r, n = estado["cnt"].most_common(1)[0]
        atual = estado["pose"]
        if atual is None or (n > estado["cnt"][atual] + 5 and n > HIST / 2):
            estado["pose"] = r

    m.add_imu_handler(on_imu)
    m.add_emg_handler(on_emg)

    # O connect() do pyomyo espera o evento de conexao SEM timeout: se o
    # bracelete estiver dormindo, ele fica preso para sempre. Nao da pra
    # matar a thread de dentro, entao um cao de guarda derruba o processo
    # com mensagem clara — e quem chamou (apresentar.py) tenta de novo.
    conectou = {"ok": False}

    def cao_de_guarda():
        fim = time.time() + a.espera
        while time.time() < fim:
            if conectou["ok"]:
                return
            time.sleep(0.2)
        if not conectou["ok"]:
            print("\nSEM RESPOSTA do bracelete em %ds.\n"
                  "  O Myo dorme quando fica parado: pegue ele e mexa (ou vista),\n"
                  "  que ele acorda. Vou sair para poder tentar de novo."
                  % a.espera, flush=True)
            os._exit(3)

    threading.Thread(target=cao_de_guarda, daemon=True).start()
    m.connect(mac)
    conectou["ok"] = True
    try:
        m.set_leds([0, 255, 0], [0, 255, 0])
        m.vibrate(1)
    except Exception:
        pass
    print("  CONECTADO. ctrl+c para parar.\n", flush=True)

    try:
        while True:
            m.run()
            agora = time.time()
            if agora - estado["ultimo_envio"] < 0.033:      # ~30 Hz
                continue
            estado["ultimo_envio"] = agora
            roll, pitch, yaw = estado["euler"]
            emg = estado["emg"]
            pico = max(max(emg), 1)
            lote, estado["buf"] = estado["buf"], []
            pacote = {
                "gesture": int(estado["pose"] or 0),
                "euler": [roll, pitch, yaw],
                "quat": list(estado["quat"]),
                "acc": list(estado["acc"]),
                "rms": [min(1.0, v / 300.0) for v in emg],
                "emg": lote,                    # forma de onda: todas as amostras
                "fs": round(estado["n"] / max(1e-6, agora - estado["t0"]), 1),
            }
            ponte.mandar(pacote)
            # uma linha inteira a cada 5 s. Sem "\r": quando o apresentar.py
            # captura a saida por pipe, retorno de carro nunca fecha a linha
            # e o status ficaria invisivel.
            if agora - estado.get("ultimo_log", 0) > 5.0:
                estado["ultimo_log"] = agora
                print("gesto=%-2s roll=%4d pitch=%4d yaw=%4d  pico=%-4d  %.0f amostras/s"
                      % (pacote["gesture"], roll, pitch, yaw, pico,
                         estado["n"] / max(1e-6, agora - estado["t0"])), flush=True)
    except KeyboardInterrupt:
        print("\n  parando...")
    finally:
        try:
            m.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
