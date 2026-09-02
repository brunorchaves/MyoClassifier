/* ============================================================
   calibra.js — mede a orientação REAL da mão (webcam + MediaPipe) e
   casa com o quaternion CRU da IMU do bracelete, no mesmo instante.

   Por que existe: a orientação da mão 3D vinha sendo acertada por
   tentativa e erro em cima de ângulos de Euler (ordem/permutação/sinal
   ambíguos), e sempre que uma coisa entrava no lugar outra saía. Aqui a
   ideia é medir em vez de adivinhar; resolver_calibracao.py lê o que isto
   grava e resolve a rotação constante entre os dois referenciais.

   POR QUE VARREDURA E NÃO POSE. A 1a versão gravava 6 poses estáticas e
   os dados saíram inúteis: a mão real girava MUITO mais que o antebraço
   (IMU 60° x mão 124°, por exemplo) porque o punho compensava sem que
   ninguém percebesse, e duas poses saíram idênticas. Agora cada item do
   roteiro grava 12 s de movimento contínuo (dezenas de amostras, muito
   mais material pro ajuste) e a tela mostra AO VIVO quanto cada lado
   girou — uma discrepância grande aparece com o braço ainda na posição,
   não depois de tudo gravado.

   O MediaPipe vem do CDN (precisa de internet na primeira carga; depois o
   navegador cacheia). A webcam só é liberada em contexto seguro — o
   serve.py serve em http://127.0.0.1, que conta como seguro.
   ============================================================ */
import {
  HandLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const WS = 'ws://127.0.0.1:8765';
const VERSAO_MP = '0.10.14';
const MODELO_MP = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/' +
                  'hand_landmarker/float16/1/hand_landmarker.task';

const DURACAO_VARREDURA_MS = 12000;
const INTERVALO_AMOSTRA_MS = 150;      // ~80 amostras por varredura
const INTERVALO_FRAME_MS = 1000;       // ~12 imagens por varredura (auditoria)
const GIRO_ALVO_GRAUS = 60;            // amplitude mínima desejável na varredura

/* Uma varredura por eixo de rotação do antebraço. Movimento lento, ida e
   volta: o ajuste usa pares de amostras, então amplitude importa mais que
   velocidade.

   ORDEM IMPORTA. As duas primeiras acontecem no PLANO da imagem, onde o
   MediaPipe mede bem — e duas já determinam a calibração por completo (o
   terceiro eixo é o produto vetorial dos outros dois). A de pronação vem
   por último e é opcional: ali a palma varre o eixo de PROFUNDIDADE, o
   pior de uma câmera só. Medido de verdade: a 1a rodada dessa varredura
   deu 178° de giro de mão contra 102° de antebraço (discrepância mediana
   de 57°), e um ajuste feito só com ela erra 100+ graus. O solver detecta
   e exclui varredura assim sozinho; esta fica como conferência. */
const VARREDURAS = [
  { id: 'sweep_pitch', nome: 'Subir e descer o antebraço',
    dica: 'palma sempre pra baixo, sobe devagar e desce — essencial' },
  { id: 'sweep_yaw', nome: 'Varrer na horizontal',
    dica: 'da esquerda pra direita e volta, mesma altura, palma pra baixo — essencial' },
  { id: 'sweep_roll', nome: 'Girar o antebraço no próprio eixo (opcional)',
    dica: 'palma pra baixo → pra cima → volta. A câmera mede mal este eixo; ' +
          'serve só de conferência' },
];

// ---------- estado ----------
const st = {
  landmarker: null,
  rodando: false,
  ultimoVideoT: -1,
  medida: null,           // {dedos, palma, lateral, world} do último frame válido
  imu: { quat: null, acc: null, euler: null, fonte: 'off', t: 0 },
  sel: 0,                 // varredura selecionada (clicável, nunca tranca)
  feitas: {},
  nSessao: 0,
  gravando: false,
  ref: null,              // referência da varredura em curso (pra checagem ao vivo)
  giro: { imu: 0, real: 0, discrep: 0, maxImu: 0, piorDiscrep: 0 },
  lmTela: null,           // landmarks 2D do último frame (pra salvar a imagem)
};

const el = {};
['video', 'sobrepor', 'avisoCam', 'erroCam', 'mMao', 'mDedos', 'mPalma', 'mQuat', 'mEuler',
 'mGirouImu', 'mGirouReal', 'mDiscrep', 'pCam', 'pMp', 'pPonte', 'listaPoses',
 'bCam', 'bCapturar', 'bPular', 'avisoCaptura', 'nSessao', 'nArquivo',
 'bVertical', 'vVetor', 'vEspalha', 'vAviso'
].forEach((id) => { el[id] = document.getElementById(id); });

const ctx = el.sobrepor.getContext('2d');

// ---------- vetores ----------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const meio = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const cruz = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norma = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norma(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
const fmt3 = (v) => (v ? v.map((x) => x.toFixed(2).padStart(6)).join(' ') : '—');

// esqueleto da mão do MediaPipe (pares de landmarks ligados)
const LIGACOES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],
                  [10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],
                  [18,19],[19,20],[0,17]];

// ---------- rotações (só o que a checagem ao vivo precisa) ----------
function matDeQuat(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}
// A * B^T (rotação relativa entre duas orientações)
function matMulT(A, B) {
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    R[i][j] = A[i][0] * B[j][0] + A[i][1] * B[j][1] + A[i][2] * B[j][2];
  }
  return R;
}
function anguloDeMat(R) {
  const cos = Math.max(-1, Math.min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2));
  return Math.acos(cos) * 180 / Math.PI;
}

/* Base ortonormal da mão a partir dos worldLandmarks (métricos, centrados
   na mão). Usa só o pulso e as bases (MCP) do indicador e do mínimo: assim
   a medida não muda quando os dedos dobram — aqui importa a orientação da
   palma, não a pose dos dedos.
   Convenção: x = lateral (indicador→mínimo), y = normal da palma,
   z = pulso→dedos. */
function baseDaMao(world) {
  const pulso = world[0], indice = world[5], minimo = world[17];
  if (!pulso || !indice || !minimo) return null;
  const dedos = unit(sub(meio(indice, minimo), pulso));
  const lateral0 = unit(sub(minimo, indice));
  const palma = unit(cruz(dedos, lateral0));
  if (norma(palma) < 1e-6) return null;
  const lateral = unit(cruz(palma, dedos));
  return { dedos, palma, lateral };
}
const matDaMao = (m) => [
  [m.lateral[0], m.palma[0], m.dedos[0]],
  [m.lateral[1], m.palma[1], m.dedos[1]],
  [m.lateral[2], m.palma[2], m.dedos[2]],
];

// ---------- MediaPipe / webcam ----------
async function iniciarMediaPipe() {
  marcarPill(el.pMp, false, 'MediaPipe: carregando…');
  const fileset = await FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSAO_MP}/wasm`);
  st.landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODELO_MP, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 1,
  });
  marcarPill(el.pMp, true, 'MediaPipe');
}

async function ligarWebcam() {
  el.erroCam.hidden = true;
  el.bCam.disabled = true;
  try {
    if (!st.landmarker) await iniciarMediaPipe();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 } }, audio: false,
    });
    el.video.srcObject = stream;
    await el.video.play();
    marcarPill(el.pCam, true, 'webcam');
    el.avisoCam.textContent = 'mostre a mão inteira pra câmera, com o antebraço aparecendo';
    st.rodando = true;
    el.bCapturar.disabled = false;
    el.bPular.disabled = false;
    requestAnimationFrame(quadro);
  } catch (e) {
    el.bCam.disabled = false;
    el.erroCam.hidden = false;
    el.erroCam.textContent = 'não deu: ' + (e && e.message ? e.message : e) +
      ' — se foi o MediaPipe, ele vem do CDN e precisa de internet na primeira carga.';
  }
}

// ---------- laço ----------
function quadro() {
  if (!st.rodando) return;
  requestAnimationFrame(quadro);

  const v = el.video;
  if (!v.videoWidth) return;
  if (el.sobrepor.width !== v.videoWidth) {
    el.sobrepor.width = v.videoWidth;
    el.sobrepor.height = v.videoHeight;
  }
  if (v.currentTime === st.ultimoVideoT) { pintar(); return; }
  st.ultimoVideoT = v.currentTime;

  let res = null;
  try { res = st.landmarker.detectForVideo(v, performance.now()); } catch (e) { /* frame ruim */ }

  st.medida = null;
  st.lmTela = null;
  if (res && res.landmarks && res.landmarks.length) {
    st.lmTela = res.landmarks[0];
    /* o MediaPipe devolve cada landmark como OBJETO {x,y,z}; as contas de
       vetor aqui trabalham com ARRAY [x,y,z]. Converter neste limite (e só
       aqui) evita o que já aconteceu: undefined nas contas -> NaN ->
       JSON.stringify grava null, e as amostras saem sem a mão. */
    const world = res.worldLandmarks[0].map((p) => [p.x, p.y, p.z]);
    const base = baseDaMao(world);
    if (base) st.medida = Object.assign({}, base, { world });
    desenharMao(res.landmarks[0]);
  } else {
    ctx.clearRect(0, 0, el.sobrepor.width, el.sobrepor.height);
  }
  atualizarGiro();
  pintar();
}

/* Quanto cada lado girou desde o início da varredura. É a checagem que
   faltava: os dois números têm que andar juntos. Se a mão gira muito mais
   que a IMU, o punho está dobrando e aquela varredura não serve. */
function atualizarGiro() {
  if (!st.gravando || !st.ref || !st.medida || !st.imu.quat) return;
  const gImu = anguloDeMat(matMulT(matDeQuat(st.imu.quat), st.ref.mImu));
  const gReal = anguloDeMat(matMulT(matDaMao(st.medida), st.ref.mReal));
  st.giro.imu = gImu;
  st.giro.real = gReal;
  st.giro.discrep = Math.abs(gImu - gReal);
  st.giro.maxImu = Math.max(st.giro.maxImu, gImu);
  st.giro.piorDiscrep = Math.max(st.giro.piorDiscrep, st.giro.discrep);
}

function desenharMao(lm) {
  const w = el.sobrepor.width, h = el.sobrepor.height;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w, 0); ctx.scale(-1, 1);   // acompanha o vídeo espelhado
  ctx.strokeStyle = st.gravando ? 'rgba(163,230,53,.9)' : 'rgba(34,211,238,.85)';
  ctx.lineWidth = Math.max(2, w / 320);
  ctx.beginPath();
  LIGACOES.forEach(([a, b]) => {
    ctx.moveTo(lm[a].x * w, lm[a].y * h);
    ctx.lineTo(lm[b].x * w, lm[b].y * h);
  });
  ctx.stroke();
  ctx.fillStyle = '#a3e635';
  lm.forEach((p, i) => {
    const r = (i === 0 || i === 5 || i === 17) ? w / 90 : w / 190;  // juntas da medida
    ctx.beginPath(); ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2); ctx.fill();
  });
  ctx.restore();
}

/* Salva um frame da webcam COM o esqueleto e os números desenhados, pra
   auditar a captura depois: olhando as imagens dá pra ver se o punho
   dobrou, se a mão saiu do quadro ou se a pose feita não era a pedida —
   em vez de confiar só nos números. Sem mirror aqui: fica na mesma
   orientação dos worldLandmarks, que é o que a análise usa. */
let cvFrame = null;
function salvarFrame(nome) {
  const v = el.video;
  if (!v.videoWidth) return;
  if (!cvFrame) cvFrame = document.createElement('canvas');
  const L = 480;
  cvFrame.width = L;
  cvFrame.height = Math.round(L * v.videoHeight / v.videoWidth);
  const c = cvFrame.getContext('2d');
  c.drawImage(v, 0, 0, cvFrame.width, cvFrame.height);

  if (st.lmTela) {
    const w = cvFrame.width, h = cvFrame.height;
    c.strokeStyle = 'rgba(163,230,53,.95)';
    c.lineWidth = 2;
    c.beginPath();
    LIGACOES.forEach(([a, b]) => {
      c.moveTo(st.lmTela[a].x * w, st.lmTela[a].y * h);
      c.lineTo(st.lmTela[b].x * w, st.lmTela[b].y * h);
    });
    c.stroke();
    c.fillStyle = '#22d3ee';
    [0, 5, 17].forEach((i) => {
      c.beginPath(); c.arc(st.lmTela[i].x * w, st.lmTela[i].y * h, 4, 0, Math.PI * 2); c.fill();
    });
  }
  const g = st.giro;
  c.fillStyle = 'rgba(0,0,0,.6)';
  c.fillRect(0, 0, cvFrame.width, 34);
  c.fillStyle = '#e9eef8';
  c.font = '12px monospace';
  c.fillText(nome, 6, 14);
  c.fillText(`IMU ${g.imu.toFixed(0)}°  mao ${g.real.toFixed(0)}°  ` +
             `discrep ${g.discrep.toFixed(0)}°`, 6, 28);

  fetch('/api/calibra-frame', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nome, jpeg_base64: cvFrame.toDataURL('image/jpeg', 0.72) }),
  }).catch(() => {});      // não travar a captura por causa de uma imagem
}

function pintar() {
  const m = st.medida;
  el.mMao.innerHTML = m ? '<b style="color:var(--verde)">detectada</b>'
                        : '<b style="color:var(--ambar)">não vejo a mão</b>';
  el.mDedos.textContent = m ? fmt3(m.dedos) : '—';
  el.mPalma.textContent = m ? fmt3(m.palma) : '—';

  const semDado = performance.now() - st.imu.t > 2500;
  marcarPill(el.pPonte, !semDado, semDado ? 'ponte/Myo' : 'ponte/Myo · ' + st.imu.fonte);
  el.mQuat.textContent = (!semDado && st.imu.quat)
    ? st.imu.quat.map((v) => v.toFixed(2).padStart(6)).join(' ') : '—';
  el.mEuler.textContent = (!semDado && st.imu.euler)
    ? st.imu.euler.map((v) => v.toFixed(0) + '°').join(' ') : '—';

  if (st.gravando) {
    const g = st.giro;
    el.mGirouImu.innerHTML = `<b>${g.imu.toFixed(0)}°</b> (máx ${g.maxImu.toFixed(0)}°)`;
    el.mGirouReal.innerHTML = `<b>${g.real.toFixed(0)}°</b>`;
    const cor = g.discrep < 10 ? 'var(--verde)' : g.discrep < 20 ? 'var(--ambar)' : 'var(--vermelho)';
    el.mDiscrep.innerHTML = `<b style="color:${cor}">${g.discrep.toFixed(0)}°</b>` +
      (g.discrep >= 20 ? ' <span style="color:var(--vermelho)">← punho dobrando</span>' : '');
  }

  el.bCapturar.disabled = st.gravando || !m || semDado;
}

function marcarPill(pill, ok, texto) {
  pill.className = 'pill ' + (ok ? 'ok' : 'off');
  pill.querySelector('span').textContent = texto;
}

// ---------- ponte (IMU) ----------
function conectar() {
  let ws;
  try { ws = new WebSocket(WS); } catch (e) { setTimeout(conectar, 4000); return; }
  ws.onclose = () => setTimeout(conectar, 4000);
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch (e) { return; }
    st.imu.t = performance.now();
    st.imu.fonte = d.src === 'myo' ? 'myo' : d.src === 'sim' ? 'sim' : 'ponte';
    if (d.quat && d.quat.length === 4) st.imu.quat = d.quat.map(Number);
    if (d.acc && d.acc.length === 3) st.imu.acc = d.acc.map(Number);
    if (d.euler && d.euler.length >= 3) st.imu.euler = d.euler.map(Number);
  };
}

/* PASSO 1 — a vertical, pela gravidade. Com o braço parado, o acelerômetro
   mede a gravidade no referencial do CORPO; R_imu * acc leva isso pro
   referencial do MUNDO do Myo. Essa é a única direção que a calibração
   precisa alinhar: casada a vertical, todo movimento sai certo e só sobra o
   heading, que sem bússola é indeterminável e a tecla espaço zera.

   Não usa a câmera — foi medido que o MediaPipe com uma câmera só erra
   demais quando a mão fica de perfil (ver calibra.html e o README). */
async function aferirVertical() {
  if (!st.imu.quat || !st.imu.acc) {
    el.vAviso.hidden = false;
    el.vAviso.textContent = 'sem dado da ponte — o feed.py precisa estar no ar ' +
      '(e reiniciado, pra mandar o acelerômetro)';
    return;
  }
  el.bVertical.disabled = true;
  el.vAviso.hidden = false;

  const vs = [];
  const fim = performance.now() + 3000;
  while (performance.now() < fim) {
    await new Promise((r) => setTimeout(r, 50));
    el.vAviso.textContent = `mantenha o braço parado… ${((fim - performance.now()) / 1000).toFixed(1)}s`;
    if (!st.imu.quat || !st.imu.acc) continue;
    const R = matDeQuat(st.imu.quat);
    const a = st.imu.acc;
    vs.push(unit([                        // R * acc, no referencial do mundo
      R[0][0] * a[0] + R[0][1] * a[1] + R[0][2] * a[2],
      R[1][0] * a[0] + R[1][1] * a[1] + R[1][2] * a[2],
      R[2][0] * a[0] + R[2][1] * a[1] + R[2][2] * a[2],
    ]));
  }
  el.bVertical.disabled = false;

  if (vs.length < 10) {
    el.vAviso.textContent = `só ${vs.length} leituras — a ponte está entregando dado?`;
    return;
  }
  const media = unit(vs.reduce((s, v) => [s[0] + v[0], s[1] + v[1], s[2] + v[2]], [0, 0, 0]));
  // espalhamento: pior desvio angular em relação à média (mede se ficou parado)
  const desvios = vs.map((v) => Math.acos(Math.max(-1, Math.min(1,
    v[0] * media[0] + v[1] * media[1] + v[2] * media[2]))) * 180 / Math.PI);
  const espalha = Math.max(...desvios);

  el.vVetor.innerHTML = `<b>[${media.map((x) => x.toFixed(3)).join(', ')}]</b>`;
  const cor = espalha < 5 ? 'var(--verde)' : espalha < 12 ? 'var(--ambar)' : 'var(--vermelho)';
  el.vEspalha.innerHTML = `<b style="color:${cor}">±${espalha.toFixed(1)}°</b> em ${vs.length} leituras`;
  if (espalha > 12) {
    el.vAviso.textContent = 'o braço se moveu muito durante a medida — refaça bem parado';
    return;
  }

  try {
    const r = await fetch('/api/calibra-vertical', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vertical: media, espalhamento_graus: espalha, n: vs.length,
        t: Date.now() / 1000, fonte: st.imu.fonte,
        quat_exemplo: st.imu.quat, acc_exemplo: st.imu.acc,
      }),
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.erro || 'erro ao gravar');
    el.vAviso.textContent = 'vertical gravada em calib/vertical.json — ' +
      'rode "python resolver_calibracao.py".';
  } catch (e) {
    el.vAviso.textContent = 'falhou ao gravar: ' + (e.message || e);
  }
}

// ---------- roteiro ----------
/* Lista clicável: escolher a varredura é seleção, não um ponteiro que só
   avança. A 1a versão usava índice linear e, depois de passar do fim, a
   captura ficava desabilitada sem volta (só recarregando a página) —
   assim dá pra refazer qualquer varredura quantas vezes quiser. */
function renderLista() {
  el.listaPoses.innerHTML = '';
  VARREDURAS.forEach((v, i) => {
    const li = document.createElement('li');
    li.className = (st.feitas[v.id] ? 'feita' : '') + (i === st.sel ? ' atual' : '');
    li.style.cursor = 'pointer';
    const feito = st.feitas[v.id]
      ? ` <span style="color:var(--verde)">· ${st.feitas[v.id]} amostras</span>` : '';
    li.innerHTML = `<div class="nome"><b>${v.nome}</b><span>${v.dica}${feito}</span></div>`;
    li.addEventListener('click', () => {
      if (st.gravando) return;
      st.sel = i;
      renderLista();
    });
    el.listaPoses.appendChild(li);
  });
  // as duas primeiras (no plano da imagem) já bastam pra resolver
  const essenciais = VARREDURAS.slice(0, 2).every((v) => st.feitas[v.id]);
  if (essenciais) {
    el.avisoCaptura.hidden = false;
    el.avisoCaptura.textContent = 'as duas varreduras essenciais estão gravadas — ' +
      'já dá pra rodar "python resolver_calibracao.py". Pode refazer qualquer ' +
      'uma clicando nela.';
  }
}

const finito = (v) => Array.isArray(v) && v.every((x) => Number.isFinite(x));

/* Grava 12 s de movimento: uma amostra a cada ~150 ms, tudo acumulado em
   memória e enviado num POST só no fim (80 POSTs num servidor
   single-thread atrapalhariam a própria captura). */
async function gravarVarredura() {
  const item = VARREDURAS[st.sel];
  if (!item || st.gravando) return;
  if (!st.medida || !st.imu.quat) return;

  st.gravando = true;
  st.ref = { mImu: matDeQuat(st.imu.quat), mReal: matDaMao(st.medida) };
  st.giro = { imu: 0, real: 0, discrep: 0, maxImu: 0, piorDiscrep: 0 };
  el.avisoCaptura.hidden = false;

  const selo = new Date().toISOString().slice(11, 19).replace(/:/g, '');
  const amostras = [];
  const fim = performance.now() + DURACAO_VARREDURA_MS;
  let proxima = 0, proximoFrame = 0, nFrames = 0, frameAtual = null;
  while (performance.now() < fim) {
    await new Promise((r) => requestAnimationFrame(r));
    const restante = (fim - performance.now()) / 1000;
    el.avisoCaptura.textContent = `gravando… ${restante.toFixed(1)}s — ` +
      `${amostras.length} amostras, ${nFrames} imagens`;

    // uma imagem por segundo: material pra auditar a captura depois
    if (performance.now() >= proximoFrame) {
      proximoFrame = performance.now() + INTERVALO_FRAME_MS;
      frameAtual = `${item.id}-${selo}-${String(++nFrames).padStart(3, '0')}`;
      salvarFrame(frameAtual);
    }

    if (performance.now() < proxima) continue;
    proxima = performance.now() + INTERVALO_AMOSTRA_MS;

    const m = st.medida;
    if (!m || !st.imu.quat) continue;
    const r_real = [m.lateral[0], m.palma[0], m.dedos[0],
                    m.lateral[1], m.palma[1], m.dedos[1],
                    m.lateral[2], m.palma[2], m.dedos[2]];
    if (![m.dedos, m.palma, m.lateral, r_real, st.imu.quat].every(finito)) continue;
    amostras.push({
      pose: item.id,
      pose_nome: item.nome,
      t: Date.now() / 1000,
      fonte: st.imu.fonte,
      quat_imu: st.imu.quat,        // (w,x,y,z) cru, normalizado no feed.py
      euler_imu: st.imu.euler,      // só referência/telemetria
      r_real,                       // colunas: x=lateral, y=palma, z=dedos
      dedos: m.dedos, palma: m.palma, lateral: m.lateral,
      world_landmarks: m.world,
      giro_imu: st.giro.imu, giro_real: st.giro.real,
      frame: frameAtual,      // liga a amostra à imagem salva daquele instante
    });
  }

  st.gravando = false;
  const g = st.giro;
  if (amostras.length < 20) {
    el.avisoCaptura.textContent = `só ${amostras.length} amostras válidas — ` +
      'a mão saiu do quadro? não gravei; tente de novo';
    return;
  }
  if (g.maxImu < GIRO_ALVO_GRAUS * 0.5) {
    el.avisoCaptura.textContent = `o braço girou pouco (máx ${g.maxImu.toFixed(0)}°, ` +
      `queria ~${GIRO_ALVO_GRAUS}°) — não gravei; faça o movimento mais amplo`;
    return;
  }

  try {
    const r = await fetch('/api/calibra', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(amostras),
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.erro || 'erro ao gravar');
    st.feitas[item.id] = (st.feitas[item.id] || 0) + j.gravadas;
    st.nSessao += j.gravadas;
    el.nSessao.textContent = st.nSessao;
    el.nArquivo.textContent = j.total;
    const alerta = g.piorDiscrep >= 20
      ? ` ⚠ discrepância chegou a ${g.piorDiscrep.toFixed(0)}° (punho dobrou) — ` +
        'clique nesta varredura pra refazer'
      : '';
    el.avisoCaptura.textContent = `“${item.nome}”: ${j.gravadas} amostras, ` +
      `${nFrames} imagens, amplitude ${g.maxImu.toFixed(0)}°.${alerta}`;
    // avança pra próxima que ainda não foi feita (sem nunca trancar)
    const prox = VARREDURAS.findIndex((v) => !st.feitas[v.id]);
    if (prox >= 0) st.sel = prox;
    renderLista();
  } catch (e) {
    el.avisoCaptura.textContent = 'falhou ao gravar: ' + (e.message || e);
  }
}

// ---------- botões ----------
el.bVertical.addEventListener('click', aferirVertical);
el.bCam.addEventListener('click', ligarWebcam);
el.bCapturar.addEventListener('click', gravarVarredura);
el.bPular.addEventListener('click', () => {
  st.sel = (st.sel + 1) % VARREDURAS.length;    // cicla, nunca passa do fim
  renderLista();
});

fetch('/api/calibra-status').then((r) => r.json())
  .then((j) => { el.nArquivo.textContent = j.total; })
  .catch(() => { el.nArquivo.textContent = '—'; });

renderLista();
conectar();
setInterval(pintar, 200);   // mantém os pills/telemetria vivos antes da webcam
