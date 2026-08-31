"""desktop.py — a mao 3D numa janela nativa, sem navegador (PLANO-desktop.md).

Le o Myo NO MESMO PROCESSO (sem bridge.py, sem servidor, sem porta) e
renderiza com moderngl + moderngl-window + pyglet. O skinning nao roda mais
em bone-space: o assimp-py nao expoe ossos/pesos/animacoes deste FBX (Passo
0 do plano), entao os dados vem de um cache gerado pela pagina web (que ja
sabe fazer skinning) — modelo.py le esse cache. O que a GPU faz aqui e um
blend de 4 posicoes/normais por vertice, ponderado por pose — mais simples
que skinning por osso, e e exatamente essa simplicidade que a rota B troca
pela mistura em bone-space que a pagina web tem.

Uso (de dentro de hand3d/):
    python desktop.py                  # le o Myo de verdade
    python desktop.py --sim            # sem bracelete: gesto/orientacao inventados
    python desktop.py --foto saida.png # renderiza um quadro parado e sai
    python desktop.py --mac 1,2,3,4,5,6

Controles: mouse orbita, scroll zoom, 1-9 trocam a pose (conforme gestos.json), g gira, w
wireframe, i alterna seguir a IMU, f reenquadra.
"""
import argparse
import math
import os
import sys
import threading
import time
from collections import Counter, deque

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import feed     # noqa: E402  (Classificador, euler_de_quaternion, MAC — sem reescrever)
import modelo   # noqa: E402

D2R = math.pi / 180.0

# mesmo preset de camera do web/hand.js (CAM), pra Verificacao por imagem bater
CAM_AZ0, CAM_EL0, CAM_ZOOM0 = 32.0, 8.0, 1.55
FOV_GRAUS = 38.0

# ---------------------------------------------------------------------
# gestos.json — fonte unica com bridge.py e web/hand.js
# ---------------------------------------------------------------------


def carregar_gestos():
    import json
    with open(os.path.join(AQUI, "gestos.json"), encoding="utf-8") as f:
        d = json.load(f)
    # extras (gestos.json:extras) sao poses desenhadas por osso (nao clipes
    # do FBX — ver README.md, "Creating new poses"): sem dado de treino, so
    # alcancaveis por tecla. O desktop.py trata igual as de 'ordem' — a
    # unica diferenca ja aconteceu antes, na exportacao pro cache.
    itens = d["ordem"] + d.get("extras", [])
    ordem = [item["clip"] for item in itens]
    classe_para_indice = {item["classe"]: i for i, item in enumerate(itens)}
    nomes = {item["clip"]: item["nome"] for item in itens}
    indice_padrao = ordem.index(d["classe_desconhecida"])
    return ordem, classe_para_indice, indice_padrao, nomes


# ---------------------------------------------------------------------
# estado compartilhado com a thread do Myo/simulador
# ---------------------------------------------------------------------

LOCK = threading.Lock()
COMPARTILHADO = {"euler": (0.0, 0.0, 0.0), "rms": [0.0] * 8, "classe": 0,
                  "fs": 0.0, "fonte": "off", "ultimo_dado": 0.0}


def simulador_loop():
    print("  [sim] inventando gesto e orientacao (sem hardware)")
    ordem_classes = [1, 2, 3, 4]
    i, t0 = 0, time.time()
    prox = t0 + 2.2
    while True:
        time.sleep(0.03)
        t = time.time() - t0
        if time.time() > prox:
            i = (i + 1) % len(ordem_classes)
            prox = time.time() + 2.2
        g = ordem_classes[i]
        base = [0.05] * 8
        if g == 2:
            base = [0.9, 0.85, 0.6, 0.35, 0.45, 0.7, 0.88, 0.95]
        elif g == 3:
            base = [0.3, 0.4, 0.68, 0.9, 0.8, 0.5, 0.32, 0.26]
        elif g == 4:
            base = [0.26, 0.18, 0.16, 0.52, 0.94, 0.98, 0.62, 0.34]
        with LOCK:
            COMPARTILHADO["classe"] = g
            COMPARTILHADO["euler"] = (22 * math.sin(t * 0.7),
                                       14 * math.sin(t * 0.5 + 1.0),
                                       30 * math.sin(t * 0.33))
            COMPARTILHADO["rms"] = [min(1.0, max(0.0, v + np.random.uniform(-0.05, 0.05)))
                                     for v in base]
            COMPARTILHADO["fs"] = 50.0
            COMPARTILHADO["fonte"] = "sim"
            COMPARTILHADO["ultimo_dado"] = time.time()


def myo_loop(repo, mac_manual, espera):
    """Le o Myo direto (sem bridge). Se o connect() travar (bracelete
    dormindo — o pyomyo espera sem timeout), a janela continua respondendo:
    so esta thread fica esperando. Numa desconexao em execucao, reconecta
    sozinha em vez de matar o processo (feed.py mata; aqui a janela ja esta
    aberta, nao ha por que derrubar tudo)."""
    src = os.path.join(repo, "src")
    if not os.path.isdir(src):
        print("  [myo] ERRO: nao achei %s (--repo)" % src)
        return
    sys.path.insert(0, src)
    from pyomyo import Myo, emg_mode

    cls = feed.Classificador(os.path.join(src, "data"))
    print("  [myo] treino: %s (total %d amostras)"
          % (", ".join("classe %d=%d" % kv for kv in sorted(cls.por_classe.items())),
             cls.X.shape[0]))

    while True:
        m = None
        try:
            m = Myo(mode=emg_mode.PREPROCESSED)
            hist = deque([0] * feed.HIST, feed.HIST)
            contagem = Counter(hist)
            pose_atual = {"v": None}

            def on_imu(quat, acc, gyro):
                r, p, y = feed.euler_de_quaternion(*quat)
                with LOCK:
                    COMPARTILHADO["euler"] = (r, p, y)
                    COMPARTILHADO["fonte"] = "myo"
                    COMPARTILHADO["ultimo_dado"] = time.time()

            def on_emg(emg, moving):
                y = cls.classificar(emg)
                contagem[hist[0]] -= 1
                contagem[y] += 1
                hist.append(y)
                r, n = contagem.most_common(1)[0]
                atual = pose_atual["v"]
                if atual is None or (n > contagem[atual] + 5 and n > feed.HIST / 2):
                    pose_atual["v"] = r
                with LOCK:
                    COMPARTILHADO["rms"] = [min(1.0, v / 300.0) for v in emg]
                    COMPARTILHADO["classe"] = pose_atual["v"] or 0
                    COMPARTILHADO["fonte"] = "myo"
                    COMPARTILHADO["ultimo_dado"] = time.time()

            m.add_imu_handler(on_imu)
            m.add_emg_handler(on_emg)

            mac = mac_manual or feed.ler_mac_salvo()
            if mac is None:
                print("  [myo] procurando o Myo (mexa o bracelete se estiver dormindo)...")
                mac = feed.descobrir_mac(m.bt, segundos=espera)
                if mac is None:
                    print("  [myo] nao achei no scan; tento de novo em 3s")
                    time.sleep(3)
                    continue
                feed.salvar_mac(mac)
            print("  [myo] conectando (MAC %s)..." % ",".join(map(str, mac)))
            m.connect(mac)
            try:
                m.set_leds([0, 255, 0], [0, 255, 0])
                m.vibrate(1)
            except Exception:
                pass
            print("  [myo] CONECTADO.")

            n, t0, ultimo_fs = 0, time.time(), 0.0
            while True:
                m.run()
                n += 1
                agora = time.time()
                if agora - ultimo_fs > 1.0:
                    ultimo_fs = agora
                    with LOCK:
                        COMPARTILHADO["fs"] = round(n / max(1e-6, agora - t0), 1)
        except Exception as e:
            print("  [myo] desconectou (%s); tentando de novo em 2s" % e)
            if m is not None:
                try:
                    m.disconnect()
                except Exception:
                    pass
            time.sleep(2)


# ---------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------

def _gerar_vert_mao(n_poses):
    """Blend de N posicoes/normais por vertice, ponderado por pose (rota B —
    ver PLANO-desktop.md). N vem de len(ordem) em gestos.json: cada pose
    (clipe do FBX ou extra desenhada por osso) e um par de atributos
    in_posI/in_nrmI; nao ha limite de vertex attribute do GLSL perto disso
    (16 no minimo garantido, aqui usamos 2 por pose)."""
    entradas = "\n".join(
        "in vec3 in_pos%d; in vec3 in_nrm%d;" % (i, i) for i in range(n_poses))
    termos_pos = " + ".join("in_pos%d * peso[%d]" % (i, i) for i in range(n_poses))
    termos_nrm = " + ".join("in_nrm%d * peso[%d]" % (i, i) for i in range(n_poses))
    return """
#version 330
%s
uniform float peso[%d];
uniform mat3 mundoRot;
uniform mat4 vp;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    vec3 pos = %s;
    vec3 nrm = %s;
    vec3 posMundo = mundoRot * pos;
    v_normal = normalize(mundoRot * nrm);
    v_pos = posMundo;
    gl_Position = vp * vec4(posMundo, 1.0);
}
""" % (entradas, n_poses, termos_pos, termos_nrm)

# Lambert + luz de borda + fresnel suave, na paleta do web/hand.js
# (HemisphereLight 0xbcd4ff/0x141a26, luz 0xffffff, borda 0x22d3ee, preenche 0xf472b6)
FRAG_MAO = """
#version 330
in vec3 v_normal;
in vec3 v_pos;
uniform vec3 camPos;
uniform bool arame;
out vec4 fragColor;

const vec3 ALBEDO = vec3(0.867, 0.894, 0.933);
const vec3 CEU = vec3(0.737, 0.831, 1.0);
const vec3 SOLO = vec3(0.078, 0.102, 0.149);
const vec3 LUZ_DIR = normalize(vec3(3.0, 5.0, 4.0));
const vec3 LUZ_COR = vec3(0.9);
const vec3 BORDA_DIR = normalize(vec3(-4.0, 0.5, -3.0));
const vec3 BORDA_COR = vec3(0.133, 0.827, 0.933) * 0.7;
const vec3 PREENCHE_DIR = normalize(vec3(2.0, -3.0, -2.0));
const vec3 PREENCHE_COR = vec3(0.957, 0.447, 0.714) * 0.18;

void main() {
    if (arame) { fragColor = vec4(ALBEDO, 1.0); return; }
    vec3 n = normalize(v_normal);
    vec3 v = normalize(camPos - v_pos);

    vec3 ambiente = mix(SOLO, CEU, 0.5 + 0.5 * n.y) * 0.9;
    vec3 difusa = LUZ_COR * max(dot(n, LUZ_DIR), 0.0)
                + BORDA_COR * max(dot(n, BORDA_DIR), 0.0)
                + PREENCHE_COR * max(dot(n, PREENCHE_DIR), 0.0);
    float fres = pow(1.0 - max(dot(n, v), 0.0), 2.5);

    fragColor = vec4(ALBEDO * (ambiente + difusa) + BORDA_COR * fres * 0.6, 1.0);
}
"""

VERT_HUD = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
uniform vec4 retangulo;   // x0,y0,x1,y1 em NDC
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(mix(retangulo.x, retangulo.z, in_vert.x),
                        mix(retangulo.y, retangulo.w, in_vert.y), 0.0, 1.0);
}
"""

FRAG_HUD = """
#version 330
in vec2 v_uv;
uniform sampler2D tex;
out vec4 fragColor;
void main() { fragColor = texture(tex, v_uv); }
"""


def _mat_col_major(m):
    # ndarray.tobytes() sempre serializa em ordem C, mesmo se o array for
    # F-contiguous — por isso e a TRANSPOSTA (nao asfortranarray) que da os
    # bytes column-major que o GLSL espera para mat3/mat4.
    return np.asarray(m, dtype="f4").T.copy().tobytes()


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _perspectiva(fovy, aspecto, near, far):
    f = 1.0 / math.tan(fovy / 2.0)
    return np.array([
        [f / aspecto, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ])


def _look_at(olho, alvo, up):
    frente = olho - alvo
    frente = frente / np.linalg.norm(frente)
    direita = np.cross(up, frente)
    direita = direita / np.linalg.norm(direita)
    cima = np.cross(frente, direita)
    return np.array([
        [direita[0], direita[1], direita[2], -np.dot(direita, olho)],
        [cima[0], cima[1], cima[2], -np.dot(cima, olho)],
        [frente[0], frente[1], frente[2], -np.dot(frente, olho)],
        [0, 0, 0, 1],
    ])


class MaoDesktop(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MyoClassifier — mão 3D (desktop)"
    window_size = (1100, 760)
    resizable = True
    vsync = True
    aspect_ratio = None

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--repo", default=os.path.dirname(AQUI),
                             help="raiz do MyoClassifier (padrao: a pasta acima de hand3d/)")
        parser.add_argument("--sim", action="store_true", help="sem bracelete: dado inventado")
        parser.add_argument("--mac", default=None, help="6 numeros separados por virgula")
        parser.add_argument("--espera", type=int, default=12,
                             help="segundos esperando o bracelete no scan antes de desistir")
        parser.add_argument("--foto", default=None, metavar="ARQUIVO.png",
                             help="renderiza alguns quadros parado e salva, sem abrir de fato")
        parser.add_argument("--pose", default=None, metavar="NOME",
                             help="comeca nesta pose (nome do clipe em gestos.json, ex.: ThumbsUp) "
                                  "em vez da padrao — util com --foto pra conferir uma pose nova")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # sem CULL_FACE: nao ha garantia da direcao do winding depois do
        # skinning exportado do three.js — ligar isto sem checar visualmente
        # arrisca a mao inteira desaparecer (ver PLANO-desktop.md, Riscos)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        print("GL_RENDERER:", self.ctx.info["GL_RENDERER"])

        self.ordem, self.classe_para_indice, self.indice_padrao, self.nomes_pose = carregar_gestos()
        self.mod = modelo.carregar(self.ordem)
        print("modelo: %d vertices, poses %s" % (self.mod.n_vertices, self.ordem))

        self.prog_mao = self.ctx.program(
            vertex_shader=_gerar_vert_mao(len(self.ordem)), fragment_shader=FRAG_MAO)
        self.prog_hud = self.ctx.program(vertex_shader=VERT_HUD, fragment_shader=FRAG_HUD)
        self._montar_buffers()
        self._montar_hud()

        # ---- estado de pose (mesma suavizacao k = 1-exp(-dt/0.10) do hand.js) ----
        self.fonte = "off"
        idx_inicial = self.indice_padrao
        if self.argv.pose:
            if self.argv.pose not in self.ordem:
                raise SystemExit("ERRO: --pose '%s' nao esta em gestos.json (opcoes: %s)"
                                  % (self.argv.pose, ", ".join(self.ordem)))
            idx_inicial = self.ordem.index(self.argv.pose)
        self.idx_pose = idx_inicial
        self._por_pose(idx_inicial, manual=bool(self.argv.pose))
        self.peso = self.alvo.copy()   # comeca ja na pose padrao, sem flash

        # ---- orientacao (ke = 1-exp(-dt/0.12), giro +0.55 rad/s Y) ----
        self.euler = [0.0, 0.0, 0.0]
        self.euler_alvo = [0.0, 0.0, 0.0]
        self.seguir_imu = True
        self.girar = False
        self.spin = 0.0
        self.ultimo_dado = 0.0
        self.classe_atual = None

        # ---- camera orbital (preset inicial = CAM do hand.js) ----
        self.cam_az, self.cam_el = CAM_AZ0, CAM_EL0
        self.cam_dist = self.mod.max_dim / (2 * math.tan(FOV_GRAUS * D2R / 2)) * CAM_ZOOM0
        self.cam_dist_min = self.mod.max_dim * 0.15
        self.cam_dist_max = self.mod.max_dim * 8.0

        self.arame = False
        self.ultimo_tempo = time.time()
        self.contador_quadros = 0
        self.fps_visor = 0.0
        self._fps_acumulado, self._fps_contagem, self._fps_marca = 0.0, 0, time.time()

        self._iniciar_fonte_dado()

    # ------------------------------------------------------------------
    def _iniciar_fonte_dado(self):
        a = self.argv
        if a.sim:
            threading.Thread(target=simulador_loop, daemon=True).start()
        else:
            mac_manual = [int(x) for x in a.mac.split(",")] if a.mac else None
            threading.Thread(target=myo_loop, args=(a.repo, mac_manual, a.espera), daemon=True).start()

    # ------------------------------------------------------------------
    def _montar_buffers(self):
        vbos = []
        for nome in self.ordem:
            pos = self.mod.posicoes[nome]
            nrm = self.mod.normais[nome]
            vbos.append(self.ctx.buffer(pos.astype("f4").tobytes()))
            vbos.append(self.ctx.buffer(nrm.astype("f4").tobytes()))
        self._vbos = vbos     # mantem referencia viva
        conteudo = []
        for i in range(len(self.ordem)):
            conteudo.append((vbos[i * 2], "3f", "in_pos%d" % i))
            conteudo.append((vbos[i * 2 + 1], "3f", "in_nrm%d" % i))
        self.vao = self.ctx.vertex_array(self.prog_mao, conteudo)

    def _montar_hud(self):
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype="f4")
        uv = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype="f4")
        self.vbo_hud_pos = self.ctx.buffer(quad.tobytes())
        self.vbo_hud_uv = self.ctx.buffer(uv.tobytes())
        self.vao_hud = self.ctx.vertex_array(
            self.prog_hud,
            [(self.vbo_hud_pos, "2f", "in_vert"), (self.vbo_hud_uv, "2f", "in_uv")],
        )
        self.hud_w, self.hud_h = 460, 190
        self.tex_hud = self.ctx.texture((self.hud_w, self.hud_h), 4)
        self.tex_hud.filter = (moderngl.LINEAR, moderngl.LINEAR)
        fontes_win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        try:
            self._fonte_hud = ImageFont.truetype(os.path.join(fontes_win, "consola.ttf"), 15)
            self._fonte_hud_titulo = ImageFont.truetype(os.path.join(fontes_win, "consolab.ttf"), 26)
        except OSError:
            self._fonte_hud = ImageFont.load_default()
            self._fonte_hud_titulo = self._fonte_hud

    # ------------------------------------------------------------------
    def _por_pose(self, i, manual):
        self.idx_pose = i
        self.alvo = np.array([1.0 if k == i else 0.0 for k in range(len(self.ordem))], dtype="f4")
        if manual:
            self.fonte = "off" if self.fonte == "off" else "manual"

    # ------------------------------------------------------------------
    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        teclas_numero = [keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3, keys.NUMBER_4,
                          keys.NUMBER_5, keys.NUMBER_6, keys.NUMBER_7, keys.NUMBER_8, keys.NUMBER_9]
        n = (teclas_numero.index(key) + 1) if key in teclas_numero else None
        if n and n <= len(self.ordem):
            self._por_pose(n - 1, manual=True)
        elif key == keys.G:
            self.girar = not self.girar
        elif key == keys.W:
            self.arame = not self.arame
            self.ctx.wireframe = self.arame
        elif key == keys.I:
            self.seguir_imu = not self.seguir_imu
        elif key == keys.F:
            self.cam_az, self.cam_el = CAM_AZ0, CAM_EL0
            self.cam_dist = self.mod.max_dim / (2 * math.tan(FOV_GRAUS * D2R / 2)) * CAM_ZOOM0

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.cam_az -= dx * 0.3
        self.cam_el = max(-89.0, min(89.0, self.cam_el + dy * 0.3))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        fator = 0.9 ** y_offset
        self.cam_dist = max(self.cam_dist_min, min(self.cam_dist_max, self.cam_dist * fator))

    # ------------------------------------------------------------------
    def _camera(self):
        az, el = self.cam_az * D2R, self.cam_el * D2R
        olho = np.array([self.cam_dist * math.cos(el) * math.sin(az),
                          self.cam_dist * math.sin(el),
                          self.cam_dist * math.cos(el) * math.cos(az)])
        alvo = np.array([0.0, self.mod.max_dim * 0.07, 0.0])
        view = _look_at(olho, alvo, np.array([0.0, 1.0, 0.0]))
        aspecto = self.wnd.buffer_size[0] / max(1, self.wnd.buffer_size[1])
        proj = _perspectiva(FOV_GRAUS * D2R, aspecto, 0.01, 8000.0)
        return olho, proj @ view

    def _atualizar_hud(self, texto_linhas):
        img = Image.new("RGBA", (self.hud_w, self.hud_h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        titulo, cor_titulo = texto_linhas[0]
        dr.text((0, 0), titulo, font=self._fonte_hud_titulo, fill=cor_titulo)
        y = 34
        for linha, cor in texto_linhas[1:]:
            dr.text((0, y), linha, font=self._fonte_hud, fill=cor)
            y += 20
        self.tex_hud.write(img.tobytes())

    # ------------------------------------------------------------------
    def on_render(self, tempo, frame_time):
        dt = min(0.05, frame_time) if frame_time else 0.016

        with LOCK:
            self.classe_atual = COMPARTILHADO["classe"]
            euler_alvo_bruto = COMPARTILHADO["euler"]
            rms = list(COMPARTILHADO["rms"])
            fs = COMPARTILHADO["fs"]
            fonte_dado = COMPARTILHADO["fonte"]
            ultimo_dado = COMPARTILHADO["ultimo_dado"]

        agora = time.time()
        if fonte_dado != "off" and ultimo_dado and (agora - ultimo_dado) < 2.5:
            self.fonte = fonte_dado
            self.euler_alvo = list(euler_alvo_bruto)
            if self.classe_atual is not None:
                i = self.classe_para_indice.get(self.classe_atual, self.indice_padrao)
                if i != self.idx_pose:
                    self._por_pose(i, manual=False)
        elif self.fonte not in ("off", "manual"):
            self.fonte = "off"

        k = 1 - math.exp(-dt / 0.10)
        self.peso += (self.alvo - self.peso) * k

        ke = 1 - math.exp(-dt / 0.12)
        segue = self.seguir_imu and self.fonte in ("myo", "sim")
        for i in range(3):
            self.euler[i] += ((self.euler_alvo[i] if segue else 0.0) - self.euler[i]) * ke
        if self.girar:
            self.spin += dt * 0.55

        # R = Rx(roll) . Ry(yaw) . Rz(pitch) — mesma ordem XYZ do three.js
        # que o pivot do hand.js usa (ORIENT e a centralizacao ja estao
        # dentro dos vertices exportados, ver web/hand.js:exportarPoses)
        roll, pitch, yaw = self.euler[0] * D2R, self.euler[1] * D2R, self.euler[2] * D2R + self.spin
        mundo_rot = _rot_x(roll) @ _rot_y(yaw) @ _rot_z(pitch)

        olho, vp = self._camera()

        self.ctx.screen.use()
        self.ctx.clear(0.027, 0.035, 0.059)
        self.prog_mao["peso"].value = tuple(float(v) for v in self.peso)
        self.prog_mao["mundoRot"].write(_mat_col_major(mundo_rot))
        self.prog_mao["vp"].write(_mat_col_major(vp))
        self.prog_mao["camPos"].value = tuple(float(v) for v in olho)
        self.prog_mao["arame"].value = self.arame
        self.vao.render(moderngl.TRIANGLES, vertices=self.mod.n_vertices)

        # ---- HUD ----
        self.contador_quadros += 1
        self._fps_acumulado += dt
        self._fps_contagem += 1
        if agora - self._fps_marca > 0.5:
            self.fps_visor = self._fps_contagem / max(1e-6, agora - self._fps_marca)
            self._fps_marca, self._fps_contagem = agora, 0

        nome_pose = self.ordem[self.idx_pose]
        rot_fonte = {"myo": "Myo ao vivo", "sim": "simulação",
                     "manual": "manual (teclas/—)", "off": "sem dado"}[self.fonte]
        linhas = [
            (self.nomes_pose.get(nome_pose, nome_pose), (233, 238, 248, 255)),
            ("clipe \"%s\" · fonte: %s" % (nome_pose, rot_fonte), (185, 196, 216, 255)),
            ("euler  roll %+4.0f  pitch %+4.0f  yaw %+4.0f" % tuple(self.euler),
             (185, 196, 216, 255)),
            ("taxa   %s" % (("%.0f Hz" % fs) if fs else "—"), (185, 196, 216, 255)),
            ("fps    %.0f    GL: %s" % (self.fps_visor, self.ctx.info["GL_RENDERER"][:28]),
             (125, 139, 165, 255)),
        ]
        self._atualizar_hud(linhas)
        self.ctx.disable(moderngl.DEPTH_TEST)
        margem_x, margem_y = 20, 18
        w, h = self.wnd.buffer_size
        x0 = -1 + 2 * margem_x / w
        y1 = 1 - 2 * margem_y / h
        x1 = x0 + 2 * self.hud_w / w
        y0 = y1 - 2 * self.hud_h / h
        self.prog_hud["retangulo"].value = (x0, y0, x1, y1)
        self.tex_hud.use(0)
        self.vao_hud.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

        if self.argv.foto and self.contador_quadros >= 8:
            self._salvar_foto(self.argv.foto)
            self.wnd.close()

    def _salvar_foto(self, caminho):
        w, h = self.wnd.buffer_size
        dados = self.ctx.screen.read(components=3)
        img = Image.frombytes("RGB", (w, h), dados).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img.save(caminho)
        print("salvo", caminho)


if __name__ == "__main__":
    mglw.run_window_config(MaoDesktop)
