"""
run.py — sobe a mao 3D inteira com um comando.

    python run.py              # ponte + Myo + pagina, e abre o navegador
    python run.py --sim        # sem bracelete: dado inventado
    python run.py --sem-myo    # ponte + pagina, sem o alimentador

Substitui o projeto Unity: nada de abrir editor, dar Play e deixar a
janela em foco para o truque do teclado funcionar.

  1. libera as portas se sobrou processo de um ensaio anterior
  2. sobe a ponte (WebSocket 8765 + TCP 25001) e espera atender
  3. sobe o alimentador do Myo, com 4 tentativas (o bracelete dorme)
  4. sobe o servidor da pagina e abre o navegador
  5. mostra uma linha de status com a taxa que esta chegando

Ctrl+C derruba tudo junto. So biblioteca padrao.
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# cor no terminal do Windows moderno; se nao pegar, sai texto puro
class Cor:
    ponte = "\033[36m"      # ciano
    myo = "\033[32m"        # verde
    web = "\033[35m"        # magenta
    aviso = "\033[33m"
    erro = "\033[31m"
    forte = "\033[1m"
    off = "\033[0m"


def diga(prefixo, cor, msg):
    # apaga a linha de status antes de escrever, senao sobra pedaco dela
    # no meio da mensagem (a status usa \r para se reescrever no lugar)
    sys.stdout.write("\r" + " " * 78 + "\r")
    print("%s%-6s%s %s" % (cor, prefixo, Cor.off, msg), flush=True)


# ----------------------------------------------------------------------
# portas
# ----------------------------------------------------------------------

def porta_ocupada(porta):
    try:
        s = socket.create_connection(("127.0.0.1", porta), timeout=0.35)
        s.close()
        return True
    except OSError:
        return False


def pids_na_porta(porta):
    """Quem esta escutando nessa porta (so Windows; netstat -ano)."""
    pids = set()
    try:
        saida = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                               timeout=8).stdout
    except Exception:
        return pids
    alvo = ":%d " % porta
    for linha in saida.splitlines():
        if alvo in linha and "LISTENING" in linha.upper():
            partes = linha.split()
            if partes and partes[-1].isdigit():
                pids.add(int(partes[-1]))
    return pids


def nome_do_pid(pid):
    try:
        saida = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH", "/FO", "CSV"],
                               capture_output=True, text=True, timeout=8).stdout
        return saida.split('","')[0].strip('"').lower() if '","' in saida else ""
    except Exception:
        return ""


def liberar(porta, rotulo):
    """Mata processo antigo NOSSO na porta. Nao mexe no que nao e python."""
    if not porta_ocupada(porta):
        return True
    for pid in pids_na_porta(porta):
        nome = nome_do_pid(pid)
        if "python" not in nome:
            diga("erro", Cor.erro,
                 "a porta %d (%s) esta ocupada por '%s' (PID %d), que nao e meu.\n"
                 "       Feche esse programa ou use outra porta." % (porta, rotulo, nome, pid))
            return False
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=8)
        diga("setup", Cor.aviso, "porta %d (%s) estava presa num python antigo "
             "(PID %d) — liberei." % (porta, rotulo, pid))
    for _ in range(20):
        if not porta_ocupada(porta):
            return True
        time.sleep(0.1)
    return not porta_ocupada(porta)


def esperar_porta(porta, segundos, nome):
    fim = time.time() + segundos
    while time.time() < fim:
        if porta_ocupada(porta):
            return True
        time.sleep(0.15)
    diga("erro", Cor.erro, "%s nao subiu em %ds (porta %d muda)" % (nome, segundos, porta))
    return False


# ----------------------------------------------------------------------
# processos filhos
# ----------------------------------------------------------------------

FILHOS = []


def subir(nome, cor, args, ao_ver=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen([sys.executable] + args, cwd=AQUI, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    FILHOS.append((nome, p))

    def ler():
        for linha in p.stdout:
            linha = linha.rstrip()
            if not linha:
                continue
            diga(nome, cor, linha.strip())
            if ao_ver:
                ao_ver(linha)
    threading.Thread(target=ler, daemon=True).start()
    return p


def derrubar():
    for nome, p in FILHOS:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    fim = time.time() + 4
    for nome, p in FILHOS:
        while p.poll() is None and time.time() < fim:
            time.sleep(0.1)
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


# ----------------------------------------------------------------------
# monitor: le a propria ponte como se fosse o navegador
# ----------------------------------------------------------------------

MON = {"src": "—", "fs": 0.0, "gesto": "—", "amostras": 0, "vivo": False, "t": 0.0}


def monitor(porta_ws):
    while True:
        try:
            chave = base64.b64encode(os.urandom(16)).decode()
            s = socket.create_connection(("127.0.0.1", porta_ws), timeout=4)
            s.sendall(("GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                       "Sec-WebSocket-Version: 13\r\n\r\n" % chave).encode())
            resp = s.recv(1024).decode("latin1")
            esperado = base64.b64encode(
                hashlib.sha1((chave + WS_GUID).encode()).digest()).decode()
            if esperado not in resp:
                s.close()
                time.sleep(2)
                continue
            MON["vivo"] = True
            while True:
                h = s.recv(2)
                if len(h) < 2:
                    break
                n = h[1] & 0x7F
                if n == 126:
                    n = struct.unpack("!H", s.recv(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", s.recv(8))[0]
                b = b""
                while len(b) < n:
                    pedaco = s.recv(n - len(b))
                    if not pedaco:
                        break
                    b += pedaco
                try:
                    d = json.loads(b.decode("utf-8", "replace"))
                except Exception:
                    continue
                MON["src"] = d.get("src", "—")
                MON["fs"] = float(d.get("fs") or 0)
                MON["gesto"] = d.get("name", "—")
                MON["amostras"] += len(d.get("emg") or ())
                MON["t"] = time.time()
        except Exception:
            MON["vivo"] = False
            time.sleep(2)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sobe a mao 3D inteira")
    ap.add_argument("--porta", type=int, default=8010, help="porta da pagina")
    ap.add_argument("--ws", type=int, default=8765)
    ap.add_argument("--tcp", type=int, default=25001)
    ap.add_argument("--sim", action="store_true", help="sem bracelete: ponte em simulacao")
    ap.add_argument("--sem-myo", action="store_true", help="nao sobe o alimentador")
    ap.add_argument("--sem-navegador", action="store_true")
    a = ap.parse_args()

    print()
    print("%sMyoClassifier — mao 3D (sem Unity)%s" % (Cor.forte, Cor.off))
    print("  ctrl+c derruba tudo\n")

    # 1. portas livres
    portas = [(a.ws, "WebSocket"), (a.porta, "pagina")]
    if not a.sim:
        portas.append((a.tcp, "TCP do Myo"))
    for porta, rotulo in portas:
        if not liberar(porta, rotulo):
            return 1

    # 2. ponte
    args_ponte = ["bridge.py", "--port", str(a.ws), "--tcp", str(a.tcp)]
    if a.sim:
        args_ponte.append("--sim")
    subir("ponte", Cor.ponte, args_ponte)
    if not esperar_porta(a.ws, 12, "a ponte"):
        derrubar()
        return 1
    threading.Thread(target=monitor, args=(a.ws,), daemon=True).start()

    # 3. alimentador do Myo, com novas tentativas
    #    O bracelete dorme quando fica parado. O alimentador desiste sozinho
    #    depois de --espera segundos, entao aqui a gente so tenta de novo:
    #    quando ele for pego e mexido, a tentativa seguinte pega.
    if not a.sim and not a.sem_myo:
        TENTATIVAS = 4
        conectou = False
        for tentativa in range(1, TENTATIVAS + 1):
            estado_myo = {"ok": False}

            def viu(linha, e=estado_myo):
                if "CONECTADO" in linha:
                    e["ok"] = True
            p = subir("myo", Cor.myo, ["feed.py", "--espera", "10"], ao_ver=viu)
            while p.poll() is None and not estado_myo["ok"]:
                time.sleep(0.2)
            if estado_myo["ok"]:
                conectou = True
                break
            if tentativa < TENTATIVAS:
                diga("myo", Cor.aviso,
                     "tentativa %d de %d falhou. %sPegue o bracelete e mexa%s "
                     "— tento de novo em 3 s."
                     % (tentativa, TENTATIVAS, Cor.forte, Cor.off))
                time.sleep(3)
        if not conectou:
            diga("aviso", Cor.aviso,
                 "o Myo nao conectou. A pagina funciona sem ele: os botoes da\n"
                 "       lateral comandam a pose.\n"
                 "       Para tentar de novo depois, num outro terminal:\n"
                 "       python feed.py")

    # 4. pagina
    subir("web", Cor.web, ["serve.py", str(a.porta), "--sem-navegador"])
    if not esperar_porta(a.porta, 12, "o servidor da pagina"):
        derrubar()
        return 1
    url = "http://127.0.0.1:%d/" % a.porta
    try:
        urllib.request.urlopen(url, timeout=4).read(64)
    except Exception as e:
        diga("aviso", Cor.aviso, "a pagina respondeu estranho (%s)" % e)

    print()
    diga("tudo", Cor.forte, "no ar — %s" % url)
    if not a.sem_navegador:
        webbrowser.open(url)

    # 5. linha de status
    print()
    try:
        while True:
            time.sleep(1.0)
            mortos = [n for n, p in FILHOS if p.poll() is not None]
            if mortos:
                diga("erro", Cor.erro, "processo caiu: %s" % ", ".join(mortos))
                break
            idade = time.time() - MON["t"] if MON["t"] else 999
            if MON["src"] == "myo" and idade < 3:
                fonte = "%sMyo ao vivo%s  %.0f Hz  gesto=%s" % (
                    Cor.myo, Cor.off, MON["fs"], MON["gesto"])
            elif MON["src"] == "sim" and idade < 3:
                fonte = "%ssimulacao%s  gesto=%s" % (Cor.aviso, Cor.off, MON["gesto"])
            else:
                fonte = "%ssem dado do bracelete%s" % (Cor.aviso, Cor.off)
            sys.stdout.write("\r  status: ponte %s · %s · %d amostras   " % (
                "ok" if MON["vivo"] else "?", fonte, MON["amostras"]))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        print()
        diga("tudo", Cor.forte, "derrubando...")
        derrubar()
        diga("tudo", Cor.forte, "ate mais.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        derrubar()
        sys.exit(130)
