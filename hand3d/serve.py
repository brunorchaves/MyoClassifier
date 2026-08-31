"""
serve.py — serve a pagina da mao 3D.

    python serve.py            # porta 8010
    python serve.py 9000       # outra porta

Precisa de servidor: abrir o web/index.html direto do disco nao funciona,
porque o navegador bloqueia a leitura do .fbx local (CORS em file://).
"""

import http.server
import os
import socketserver
import sys
import threading
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "web")
GESTOS = os.path.join(AQUI, "gestos.json")
PORTA = int(sys.argv[1]) if len(sys.argv) > 1 else 8010


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
        super().do_GET()

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
