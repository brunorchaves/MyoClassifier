"""
resolver_calibracao.py — resolve a transformacao IMU -> cena a partir das
amostras gravadas por web/calibra.html.

    python resolver_calibracao.py                 # le calib/amostras.jsonl
    python resolver_calibracao.py --arq outro.jsonl

O QUE ESTE SCRIPT RESOLVE

A mao 3D precisa girar como o braco gira. O que a IMU entrega e uma
orientacao no referencial ARBITRARIO dela (o Myo nao tem bussola, e o
"zero" muda a cada conexao); o que a cena precisa e a mesma rotacao
expressa nos eixos do three.js. Falta portanto uma rotacao constante A:

    Q_pivot = A * Q_imu * B

Detalhe que simplifica tudo: em rotacoes RELATIVAS (que e o que se ve
quando o braco se move), o B cancela --

    Q_pivot_i * Q_pivot_j^-1 = A * (Q_imu_i * Q_imu_j^-1) * A^-1

-- ou seja, o B so decide a pose de repouso (e a tecla espaco do hand.js
ja resolve isso ao vivo), enquanto o A e o unico responsavel por "levantar
o braco levanta a mao" em vez de "sai na diagonal". Entao e o A que este
script estima, e ele estima a partir de PARES de amostras:

    dR_real = A * dR_imu * A^-1        para todo par (i, j)

Isso vira um problema classico de ajuste de rotacao (Wahba/Kabsch): o eixo
de cada rotacao relativa medida na camera tem que ser o eixo da rotacao
relativa da IMU levado por A. Com >= 3 pares independentes o A fica
determinado; com mais pares, sobra redundancia para medir o residuo — que
e o numero que diz se a calibracao presta.

CONVENCOES

- quat_imu: (w,x,y,z), normalizado no feed.py.
- r_real: colunas [lateral, palma, dedos] nos eixos da webcam do MediaPipe
  (x direita, y BAIXO, z afastando). Aqui viram eixos de cena (y para
  cima) por diag(1,-1,-1) — rotacao propria (det=+1), a mesma conversao
  OpenCV -> OpenGL.
- As amostras da 1a rodada sairam com r_real nulo (bug ja corrigido no
  calibra.js); por isso a base e sempre RECALCULADA aqui a partir de
  world_landmarks, que e o dado cru.
"""

import argparse
import json
import math
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARQ_PADRAO = os.path.join(AQUI, "calib", "amostras.jsonl")

# webcam (x direita, y baixo, z afastando) -> cena three.js (y para cima)
F_CAM_CENA = np.diag([1.0, -1.0, -1.0])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def base_da_mao(world):
    """[lateral, palma, dedos] a partir do pulso e das bases dos dedos.

    Mesma formula do calibra.js: so usa pulso (0), base do indicador (5) e
    base do minimo (17), entao nao muda quando os dedos dobram.
    """
    w = np.asarray(world, dtype=float)
    pulso, indice, minimo = w[0], w[5], w[17]
    dedos = unit((indice + minimo) / 2.0 - pulso)
    palma = unit(np.cross(dedos, unit(minimo - indice)))
    lateral = unit(np.cross(palma, dedos))
    return lateral, palma, dedos


def matriz_de_quat(q):
    """(w,x,y,z) -> matriz de rotacao 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_de_matriz(R):
    """Matriz de rotacao 3x3 -> (w,x,y,z), pelo maior denominador."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x, y, z = (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, y = (R[0, 2] - R[2, 0]) / s, 0.25 * s
        x, z = (R[0, 1] + R[1, 0]) / s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, z = (R[1, 0] - R[0, 1]) / s, 0.25 * s
        x, y = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def eixo_angulo(R):
    """Eixo unitario e angulo (rad) de uma rotacao 3x3."""
    cos = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(cos)
    if ang < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(math.pi - ang) < 1e-6:      # 180 graus: eixo pela parte simetrica
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        eixo = unit(M[:, i])
        return eixo, ang
    eixo = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return unit(eixo), ang


def carregar(arq, silencioso=False):
    if not os.path.exists(arq):
        outros = []
        pasta = os.path.dirname(arq)
        if os.path.isdir(pasta):
            outros = [n for n in sorted(os.listdir(pasta)) if n.endswith(".jsonl")]
        msg = ["nao achei %s" % arq,
               "",
               "O caminho recomendado NAO precisa deste arquivo nem da webcam:",
               "  1) python run.py --calibra",
               "  2) botao 'aferir a vertical (3s parado)'  -> grava calib/vertical.json",
               "  3) python resolver_calibracao.py",
               "",
               "As varreduras com camera sao so conferencia (--camera forca esse caminho).",
               "",
               "Na pagina, a varredura so e gravada se render >= 20 amostras validas",
               "e o braco girar o bastante — se ela recusou, o motivo aparece ali."]
        if outros:
            msg += ["", "Outros .jsonl nesta pasta (use --arq para escolher):"]
            msg += ["  " + os.path.join(pasta, n) for n in outros]
        raise SystemExit("\n".join(msg))

    amostras = []
    with open(arq, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            if not d.get("world_landmarks") or not d.get("quat_imu"):
                continue
            lateral, palma, dedos = base_da_mao(d["world_landmarks"])
            R_cam = np.column_stack([lateral, palma, dedos])
            amostras.append({
                "pose": d.get("pose", "?"),
                "R_real": F_CAM_CENA @ R_cam,          # eixos de cena
                "R_imu": matriz_de_quat(d["quat_imu"]),
            })
    return amostras


def montar_pares(amostras, ang_min=15.0, ang_max=150.0, tol_ang=15.0, maximo=4000, semente=7):
    """Pares (i,j) cujas rotacoes relativas servem de evidencia.

    Descarta rotacao pequena (eixo mal definido), rotacao perto de 180
    (eixo ambiguo de sinal) e — o filtro que mais importa — par em que os
    dois lados discordam no ANGULO: se a mao girou 120 e o antebraco 60, o
    punho dobrou e aquele par nao mede o mesmo movimento.
    """
    rng = np.random.default_rng(semente)
    n = len(amostras)
    candidatos = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(candidatos) > maximo:
        idx = rng.choice(len(candidatos), size=maximo, replace=False)
        candidatos = [candidatos[k] for k in idx]

    pares, motivos = [], {"pequena": 0, "perto_180": 0, "discordam": 0}
    for i, j in candidatos:
        e_imu, a_imu = eixo_angulo(amostras[i]["R_imu"] @ amostras[j]["R_imu"].T)
        e_real, a_real = eixo_angulo(amostras[i]["R_real"] @ amostras[j]["R_real"].T)
        gi, gr = math.degrees(a_imu), math.degrees(a_real)
        if min(gi, gr) < ang_min:
            motivos["pequena"] += 1
        elif max(gi, gr) > ang_max:
            motivos["perto_180"] += 1
        elif abs(gi - gr) > tol_ang:
            motivos["discordam"] += 1
        else:
            pares.append({"i": i, "j": j, "ang_imu": gi, "ang_real": gr,
                          "e_imu": e_imu, "e_real": e_real})
    return pares, motivos, len(candidatos)


def avaliar_grupos(amostras, tol_ang=20.0, maximo_por_grupo=400, semente=7):
    """Confiabilidade de cada varredura, medida sozinha.

    Se um par diz que o antebraco girou 60 e a mao 120, alguem esta errado.
    Numa varredura de PRONACAO isso e esperado e nao e culpa do usuario: a
    palma varre o eixo de PROFUNDIDADE, que e justamente o que o MediaPipe
    estima pior com uma camera so (medido: erro de 100+ graus no ajuste se
    esses dados entrarem). Ja subir/descer e varrer pro lado acontecem no
    plano da imagem, onde a medida e boa — e duas varreduras no plano ja
    determinam A por completo (o 3o eixo e o produto vetorial dos outros).

    Devolve {pose: {"n", "discrep_mediana", "confiavel"}}.
    """
    rng = np.random.default_rng(semente)
    grupos = {}
    for k, x in enumerate(amostras):
        grupos.setdefault(x["pose"], []).append(k)

    fora = {}
    for pose, idxs in grupos.items():
        difs = []
        candidatos = [(i, j) for a, i in enumerate(idxs) for j in idxs[a + 1:]]
        if len(candidatos) > maximo_por_grupo:
            sel = rng.choice(len(candidatos), size=maximo_por_grupo, replace=False)
            candidatos = [candidatos[k] for k in sel]
        for i, j in candidatos:
            _, a_imu = eixo_angulo(amostras[i]["R_imu"] @ amostras[j]["R_imu"].T)
            _, a_real = eixo_angulo(amostras[i]["R_real"] @ amostras[j]["R_real"].T)
            if max(math.degrees(a_imu), math.degrees(a_real)) < 15:
                continue                      # rotacao pequena nao informa nada
            difs.append(abs(math.degrees(a_imu) - math.degrees(a_real)))
        med = float(np.median(difs)) if difs else float("nan")
        fora[pose] = {
            "n": len(idxs),
            "discrep_mediana": med,
            "confiavel": bool(difs) and med <= tol_ang,
        }
    return fora


def rot_de_eixo_angulo(eixo, ang):
    e = unit(np.asarray(eixo, float))
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def eixo_vertical_das_varreduras(amostras, pose="sweep_yaw"):
    """Vertical estimada pelo movimento: varrer o braco na horizontal gira em
    torno da gravidade, entao o eixo dominante daquele trecho E a vertical.
    Serve de CONFERENCIA independente da medida do acelerometro."""
    g = [x for x in amostras if x["pose"] == pose]
    if len(g) < 5:
        return None, 0.0
    M = np.zeros((3, 3))
    ref = None
    for k in range(1, len(g)):
        e, a = eixo_angulo(g[k]["R_imu"] @ g[k - 1]["R_imu"].T)
        if math.degrees(a) < 3:
            continue
        if ref is None:
            ref = e
        if np.dot(e, ref) < 0:
            e = -e
        M += math.degrees(a) * np.outer(e, e)
    if ref is None:
        return None, 0.0
    vals, vecs = np.linalg.eigh(M)
    ordem = np.argsort(vals)[::-1]
    d = vecs[:, ordem[0]]
    if np.dot(d, ref) < 0:
        d = -d
    pureza = float(vals[ordem[0]] / max(1e-9, vals.sum()))
    return unit(d), pureza


def resolver_por_vertical(v_imu):
    """A rotacao MINIMA que leva a vertical medida do Myo na vertical da cena.

    So a vertical precisa casar: com ela alinhada, subir o braco sobe a mao,
    varrer pro lado varre pro lado, e pronacao gira a mao no proprio eixo. O
    que sobra e o heading (girar em torno da vertical), que um IMU sem
    bussola nao tem como saber — a tecla espaco do hand.js zera isso ao vivo.
    Por isso escolhe-se a rotacao minima: qualquer outra so embutiria um
    heading arbitrario.
    """
    cena_up = np.array([0.0, 1.0, 0.0])
    v = unit(np.asarray(v_imu, float))
    eixo = np.cross(v, cena_up)
    s, c = float(np.linalg.norm(eixo)), float(np.dot(v, cena_up))
    if s < 1e-9:                       # ja alinhada, ou exatamente oposta
        return np.eye(3) if c > 0 else rot_de_eixo_angulo([1, 0, 0], math.pi)
    return rot_de_eixo_angulo(eixo / s, math.atan2(s, c))


def a_pose_unica(amostras):
    """True se todas as amostras vem de uma unica varredura — um eixo de
    rotacao sozinho nao determina A (falta girar em outro eixo)."""
    return len({x["pose"] for x in amostras}) < 2


def kabsch(pares):
    """A que melhor leva os eixos da IMU nos eixos medidos (ponderado pelo
    angulo: movimento maior = eixo mais confiavel)."""
    H = np.zeros((3, 3))
    for p in pares:
        peso = math.radians(min(p["ang_imu"], p["ang_real"]))
        H += peso * np.outer(p["e_real"], p["e_imu"])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def erro_eixo(A, p):
    """Angulo (graus) entre o eixo previsto por A e o eixo medido."""
    cos = float(np.dot(unit(A @ p["e_imu"]), p["e_real"]))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def resolver(pares, limiar=15.0, iteracoes=300, semente=7):
    """RANSAC + refino: acha o maior conjunto de pares coerentes e ajusta A
    nele. Com varredura continua sobram pares de sobra, e alguns vao estar
    contaminados (punho, profundidade do MediaPipe) — melhor achar o
    consenso do que deixar os ruins entrarem na media."""
    if len(pares) < 3:
        return None, [], []
    rng = np.random.default_rng(semente)
    melhor_A, melhores = None, []
    for _ in range(iteracoes):
        amostra = [pares[k] for k in rng.choice(len(pares), size=3, replace=False)]
        A = kabsch(amostra)
        dentro = [p for p in pares if erro_eixo(A, p) < limiar]
        if len(dentro) > len(melhores):
            melhor_A, melhores = A, dentro
    if melhor_A is None or len(melhores) < 3:
        return None, [], pares
    A = kabsch(melhores)                                  # refino nos inliers
    dentro = [p for p in pares if erro_eixo(A, p) < limiar]
    fora = [p for p in pares if erro_eixo(A, p) >= limiar]
    return A, dentro, fora


def imprimir_constante(A, nota):
    q = quat_de_matriz(A)
    print("\n--- constante para o web/hand.js ---")
    print("  // A: IMU -> cena, de resolver_calibracao.py")
    print("  //    %s" % nota)
    print("  var Q_IMU_CENA = new THREE.Quaternion(%.6f, %.6f, %.6f, %.6f);"
          % (q[1], q[2], q[3], q[0]))     # THREE usa (x,y,z,w)
    print("\n  matriz A (linhas):")
    for linha in A:
        print("    [%+.4f, %+.4f, %+.4f]" % tuple(linha))


def caminho_vertical(arq_vertical, amostras):
    """Caminho PRINCIPAL: alinhar a vertical medida pela gravidade."""
    with open(arq_vertical, encoding="utf-8") as f:
        d = json.load(f)
    v = unit(np.asarray(d["vertical"], float))
    esp = d.get("espalhamento_graus")
    print("VERTICAL PELA GRAVIDADE (acelerometro, braco parado)")
    print(f"  vertical no mundo do Myo: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]")
    print(f"  estabilidade da medida: +-{esp:.1f}deg em {d.get('n')} leituras"
          if esp is not None else "")
    eixo_dom = int(np.argmax(np.abs(v)))
    print(f"  eixo dominante: {'xyz'[eixo_dom]} "
          f"({'+' if v[eixo_dom] > 0 else '-'}{abs(v[eixo_dom]):.3f})")

    if amostras:
        v_mov, pureza = eixo_vertical_das_varreduras(amostras)
        if v_mov is not None:
            for sinal in (1, -1):
                ang = math.degrees(math.acos(max(-1.0, min(1.0,
                      float(np.dot(sinal * v_mov, v))))))
                if ang <= 90:
                    break
            print("\nCONFERENCIA INDEPENDENTE (pelo movimento, nao pela gravidade)")
            print(f"  varrer o braco na horizontal gira em torno da vertical;")
            print(f"  eixo daquele trecho: [{v_mov[0]:+.3f}, {v_mov[1]:+.3f}, {v_mov[2]:+.3f}]"
                  f"  (pureza {pureza * 100:.0f}%)")
            print(f"  diferenca em relacao a gravidade: {ang:.1f}deg", end="  ")
            print("-> CONFEREM" if ang < 15 else "-> NAO conferem, desconfie")

    A = resolver_por_vertical(v)
    conferido = A @ v
    print(f"\n  checagem: A * vertical = [{conferido[0]:+.3f}, {conferido[1]:+.3f}, "
          f"{conferido[2]:+.3f}]  (queremos [0,+1,0])")
    imprimir_constante(A, "rotacao minima que alinha a vertical medida pela "
                          "gravidade; heading fica pra tecla espaco")
    print("\n  Se a mao aparecer de cabeca pra baixo, o sinal do acelerometro")
    print("  esta invertido nesta unidade: troque o sinal da vertical em")
    print("  calib/vertical.json e rode de novo.")
    return A


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arq", default=ARQ_PADRAO)
    ap.add_argument("--vertical", default=os.path.join(AQUI, "calib", "vertical.json"))
    ap.add_argument("--camera", action="store_true",
                    help="forca o caminho antigo (ajuste pela camera/MediaPipe)")
    ap.add_argument("--pose", action="append",
                    help="usa so as amostras destas poses (pode repetir)")
    ap.add_argument("--limiar", type=float, default=15.0,
                    help="erro de eixo (graus) para um par contar como coerente")
    a = ap.parse_args()

    # Caminho principal: a vertical pela gravidade. Robusto e de uma medida
    # so — nao depende do MediaPipe, que com uma camera so erra muito quando
    # a mao fica de perfil (medido: 179deg de "giro" em 150ms).
    if os.path.exists(a.vertical) and not a.camera:
        amostras = carregar(a.arq, silencioso=True) if os.path.exists(a.arq) else []
        caminho_vertical(a.vertical, amostras)
        return

    amostras = carregar(a.arq)
    if a.pose:
        amostras = [x for x in amostras if x["pose"] in set(a.pose)]
    print(f"lidas {len(amostras)} amostras de {a.arq}")

    if len(amostras) < 3:
        raise SystemExit("preciso de pelo menos 3 amostras")

    aval = avaliar_grupos(amostras)
    print("\nVARREDURAS (discrepancia = quanto os dois lados discordam no angulo)")
    for pose in sorted(aval):
        a = aval[pose]
        marca = "usada" if a["confiavel"] else "EXCLUIDA"
        print(f"  {pose:<22} {a['n']:4d} amostras   "
              f"discrep. mediana {a['discrep_mediana']:5.1f}deg   -> {marca}")

    usaveis = [x for x in amostras if aval[x["pose"]]["confiavel"]]
    excluidas = sorted(p for p in aval if not aval[p]["confiavel"])
    if excluidas:
        print(f"\n  Excluidas: {', '.join(excluidas)}.")
        print("  Numa varredura de pronacao isso e esperado (a palma varre o eixo")
        print("  de profundidade, o pior do MediaPipe com uma camera so) e nao")
        print("  invalida a calibracao: subir/descer + varrer pro lado ja bastam.")
    if len(usaveis) < 3:
        raise SystemExit("\nnenhuma varredura confiavel — regrave subir/descer e "
                         "varrer pro lado, que acontecem no plano da camera")
    if a_pose_unica(usaveis):
        print("\n  AVISO: so uma varredura confiavel — um eixo de rotacao nao")
        print("         determina A. Grave tambem a outra varredura no plano.")

    pares, motivos, testados = montar_pares(usaveis)
    print(f"PARES: {len(pares)} utilizaveis de {testados} testados")
    print(f"  descartados: {motivos['pequena']} rotacao pequena, "
          f"{motivos['perto_180']} perto de 180, "
          f"{motivos['discordam']} angulos discordam (punho/profundidade)")

    if motivos["discordam"] > 0.5 * testados:
        print("  AVISO: mais da metade dos pares discorda no angulo — sinal de")
        print("         punho dobrando durante a captura.")

    A, dentro, fora = resolver(pares, limiar=a.limiar)
    if A is None:
        raise SystemExit("\npares coerentes insuficientes — regrave as varreduras")

    residuos = sorted(erro_eixo(A, p) for p in dentro)
    mediana = residuos[len(residuos) // 2]
    frac = len(dentro) / max(1, len(pares))

    print(f"\nAJUSTE: {len(dentro)} pares coerentes ({frac * 100:.0f}%), "
          f"{len(fora)} fora")
    print(f"  residuo nos coerentes: mediana {mediana:.1f}deg, pior {residuos[-1]:.1f}deg")
    if frac > 0.6 and mediana < 8:
        veredito = "OTIMO — da pra aplicar no hand.js"
    elif frac > 0.4 and mediana < 12:
        veredito = "RAZOAVEL — aplicavel; regravar melhoraria"
    else:
        veredito = ("RUIM — nao confie. Regrave prendendo a mao ao antebraco "
                    "(a discrepancia ao vivo tem que ficar abaixo de 10 graus)")
    print(f"  {veredito}")

    q = quat_de_matriz(A)
    print("\n--- constante para o web/hand.js ---")
    print("  // A: IMU -> cena, resolvido por resolver_calibracao.py")
    print(f"  //    {len(dentro)} pares coerentes, residuo mediano {mediana:.1f}deg")
    print("  var Q_IMU_CENA = new THREE.Quaternion(%.6f, %.6f, %.6f, %.6f);"
          % (q[1], q[2], q[3], q[0]))     # THREE usa (x,y,z,w)
    print("\n  matriz A (linhas):")
    for linha in A:
        print("    [%+.4f, %+.4f, %+.4f]" % tuple(linha))


if __name__ == "__main__":
    main()
