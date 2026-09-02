/* ============================================================
   dashboard.js — painel que engloba a mão 3D.

   A mão 3D (index.html/hand.js) continua exatamente como está: este
   painel abre uma SEGUNDA conexão com a mesma ponte (bridge.py aceita
   vários navegadores ao mesmo tempo, ver README.md) e desenha, ao
   redor da mão embutida num <iframe>, as ondas de EMG cru, o cubo da
   IMU e o resto da telemetria.

   O <iframe> de index.html fica com largura fixa abaixo de 900px (ver
   main{grid-template-columns} em dashboard.html) de propósito: é o
   mesmo ponto de quebra que index.html já usa (@media max-width:900px)
   para escoder o painel lateral dela sozinha — sem precisar tocar
   naquele arquivo, só a mão aparece aqui dentro.
   ============================================================ */
(function () {
  'use strict';

  var WS = 'ws://127.0.0.1:8765';
  var CORES = ['#22d3ee', '#2dd4bf', '#a3e635', '#fbbf24',
               '#fb923c', '#fb7185', '#f472b6', '#a78bfa'];
  var N_CANAIS = 8;
  var JANELA = 640;              // amostras guardadas por canal (~13s a 50Hz)

  /* Orientação do cubo da IMU: pelo QUATERNION cru, como no web/hand.js.
     Antes o cubo montava a rotação com ângulos de Euler e um mapeamento de
     eixos feito à mão ('rotateX(-pitch) rotateY(yaw) rotateZ(roll)'), que é
     a mesma armadilha que a mão 3D tinha — Euler de uma convenção Z-up
     remontado noutra ordem — e o cubo saía com roll trocado com pitch
     mesmo depois da mão já estar certa.

     Q_IMU_CENA_BASE e HEADING_GRAUS são CÓPIAS de web/hand.js, que é a
     fonte da verdade (medidos pela gravidade por resolver_calibracao.py).
     Se mudarem lá, mudam aqui — não há como o dashboard.js ler a constante
     de dentro do <iframe> sem acoplar as duas páginas. */
  var Q_IMU_CENA_BASE = [-0.706714, 0.000000, 0.001382, 0.707498];  // (x,y,z,w)
  var HEADING_GRAUS = 90;

  var st = {
    ws: null, ultimoDado: 0, ultimoPacote: 0,
    fonte: 'off', fs: 0, gesture: null, nome: '—',
    euler: [0, 0, 0], eulerAlvo: [0, 0, 0],
    quat: null,                            // (x,y,z,w) cru do pacote
    quatCubo: [0, 0, 0, 1],                // suavizado, é o que o cubo mostra
    rms: new Array(N_CANAIS).fill(0),
    inicio: performance.now(),
    totalAmostras: 0, pico: 0,
    ultimoLote: 0,
    pacotesJanela: [],
  };

  // ring buffer por canal + escala automática (decai lentamente, sobe rápido)
  var buf = [], escreverEm = [], escala = [];
  for (var c = 0; c < N_CANAIS; c++) {
    buf.push(new Float32Array(JANELA));
    escreverEm.push(0);
    escala.push(120);
  }

  // ---------- quaternions (o mínimo, sem three.js nesta página) ----------
  function qMul(a, b) {                    // a*b: aplica b, depois a
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
  }
  function qEixo(eixo, graus) {
    var r = graus * Math.PI / 360, s = Math.sin(r);
    return [eixo[0] * s, eixo[1] * s, eixo[2] * s, Math.cos(r)];
  }
  // interpolação com correção de sinal: sem ela, o cubo dá meia-volta pelo
  // lado longo quando o quaternion troca de hemisfério
  function qSuavizar(atual, alvo, k) {
    var d = atual[0] * alvo[0] + atual[1] * alvo[1] + atual[2] * alvo[2] + atual[3] * alvo[3];
    var sinal = d < 0 ? -1 : 1;
    var q = [0, 0, 0, 0], n = 0;
    for (var i = 0; i < 4; i++) {
      q[i] = atual[i] + (alvo[i] * sinal - atual[i]) * k;
      n += q[i] * q[i];
    }
    n = Math.sqrt(n) || 1;
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  }
  /* Quaternion (cena, Y pra cima) -> matrix3d do CSS (Y pra BAIXO).
     A conversão é R_css = F * R * F, com F = diag(1,-1,1): conjugar por essa
     reflexão troca o sentido do eixo Y e devolve uma rotação válida. Usar
     matrix3d evita decompor em rotateX/Y/Z — foi decompor que trouxe o bug
     de ordem de eixos aqui. matrix3d é COLUNA-maior. */
  function cssDeQuat(q) {
    var x = q[0], y = q[1], z = q[2], w = q[3];
    var m = [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ];
    var f = [1, -1, 1];
    var r = [];
    for (var i = 0; i < 3; i++) {
      r.push([]);
      for (var j = 0; j < 3; j++) r[i].push(f[i] * m[i][j] * f[j]);
    }
    return 'matrix3d(' + [
      r[0][0], r[1][0], r[2][0], 0,
      r[0][1], r[1][1], r[2][1], 0,
      r[0][2], r[1][2], r[2][2], 0,
      0, 0, 0, 1,
    ].map(function (v) { return v.toFixed(5); }).join(',') + ')';
  }

  // IMU -> cena, já com o heading (mesma composição do hand.js). Normaliza
  // porque a constante está escrita com 6 casas e não sai exatamente
  // unitária — sem isso a matriz do CSS acumula ~1e-8 de erro de escala.
  var Q_IMU_CENA = (function () {
    var q = HEADING_GRAUS
      ? qMul(qEixo([0, 1, 0], HEADING_GRAUS), Q_IMU_CENA_BASE)
      : Q_IMU_CENA_BASE.slice();
    var n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) || 1;
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  })();

  function empilhar(canal, valor) {
    buf[canal][escreverEm[canal] % JANELA] = valor;
    escreverEm[canal]++;
    var abs = Math.abs(valor);
    escala[canal] = abs > escala[canal] ? abs : escala[canal] * 0.997 + 40 * 0.003;
  }

  // ---------- elementos ----------
  var el = {};
  ['pFonte', 'relogio', 'emgInfo', 'emgLegenda', 'sRotulo', 'sFs', 'sLat', 'sLote',
   'gNome', 'gClasse', 'gBarras', 'iRoll', 'iPitch', 'iYaw', 'imuCubo',
   'stTempo', 'stAmostras', 'stPacotes', 'stPico'].forEach(function (id) {
    el[id] = document.getElementById(id);
  });

  // legenda dos canais
  CORES.forEach(function (cor, i) {
    var span = document.createElement('span');
    span.innerHTML = '<i style="background:' + cor + '"></i>c' + (i + 1);
    el.emgLegenda.appendChild(span);
  });

  // barrinhas de atividade por canal, no cartão de gesto
  var barrinhas = [];
  for (c = 0; c < N_CANAIS; c++) {
    var b = document.createElement('i');
    el.gBarras.appendChild(b);
    barrinhas.push(b);
  }

  // ---------- WebSocket ----------
  function conectar() {
    if (!('WebSocket' in window)) return;
    if (st.ws && (st.ws.readyState === 0 || st.ws.readyState === 1)) return;
    var ws;
    try { ws = new WebSocket(WS); } catch (e) { setTimeout(conectar, 4000); return; }
    st.ws = ws;
    ws.onclose = function () { setTimeout(conectar, 4000); };
    ws.onerror = function () {};
    ws.onmessage = function (ev) {
      var d;
      try { d = JSON.parse(ev.data); } catch (e) { return; }
      var agora = performance.now();
      var dt = st.ultimoDado ? agora - st.ultimoDado : 0;
      st.ultimoDado = agora;
      st.ultimoPacote = dt;
      st.pacotesJanela.push(agora);

      st.fonte = d.src === 'myo' ? 'myo' : d.src === 'sim' ? 'sim' : 'ponte';
      if (d.fs) st.fs = +d.fs;
      if (d.gesture !== undefined) st.gesture = +d.gesture;
      if (d.name) st.nome = String(d.name);
      if (d.euler && d.euler.length >= 3) {
        st.eulerAlvo = [+d.euler[0], +d.euler[1], +d.euler[2]];
      }
      // quat cru: (w,x,y,z) no pacote, (x,y,z,w) aqui
      if (d.quat && d.quat.length === 4) {
        st.quat = [+d.quat[1], +d.quat[2], +d.quat[3], +d.quat[0]];
      }
      if (d.rms && d.rms.length) st.rms = d.rms.map(Number);
      if (d.emg && d.emg.length) {
        st.ultimoLote = d.emg.length;
        d.emg.forEach(function (amostra) {
          if (!Array.isArray(amostra)) return;
          for (var i = 0; i < N_CANAIS && i < amostra.length; i++) {
            var v = +amostra[i];
            empilhar(i, v);
            st.totalAmostras++;
            if (v > st.pico) st.pico = v;
          }
        });
        // aba de treinamento (treino.js): repassa o lote bruto pra quem
        // quiser gravar amostras, sem abrir uma segunda conexao WebSocket.
        if (window.onLoteEmg) window.onLoteEmg(d.emg);
      }
    };
  }

  // reusado por treino.js -- mesma heuristica de "sem dado" de pintar()/
  // desenharEmg() abaixo, pra nao duplicar a deteccao de conexao.
  window.getEstadoPonte = function () {
    return { semDado: performance.now() - st.ultimoDado > 2500, fs: st.fs };
  };

  // ---------- canvas de EMG ----------
  var canvas = document.getElementById('emgCanvas');
  var ctx = canvas.getContext('2d');

  function redimensionarCanvas() {
    var r = canvas.parentElement.getBoundingClientRect();
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
  }
  window.addEventListener('resize', redimensionarCanvas);

  function desenharEmg() {
    var w = canvas.width, h = canvas.height;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);

    var faixa = h / N_CANAIS;
    var semDado = performance.now() - st.ultimoDado > 2500;

    for (var c = 0; c < N_CANAIS; c++) {
      var y0 = c * faixa, meio = y0 + faixa / 2;

      // linha de base da faixa
      ctx.strokeStyle = 'rgba(150,180,230,.10)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, meio); ctx.lineTo(w, meio);
      ctx.stroke();

      // rótulo do canal
      ctx.fillStyle = 'rgba(125,139,165,.85)';
      ctx.font = (11 * (window.devicePixelRatio || 1)) + 'px "Cascadia Code","JetBrains Mono",monospace';
      ctx.fillText('c' + (c + 1), 6, y0 + 13 * (window.devicePixelRatio || 1));

      if (semDado) continue;

      ctx.strokeStyle = CORES[c];
      ctx.lineWidth = 1.6 * (window.devicePixelRatio || 1);
      ctx.globalAlpha = 0.92;
      ctx.beginPath();
      var amp = (faixa * 0.42) / (escala[c] || 1);
      var n = Math.min(JANELA, escreverEm[c]);
      for (var i = 0; i < n; i++) {
        var idx = (escreverEm[c] - n + i) % JANELA;
        var x = (i / (JANELA - 1)) * w;
        var v = meio - buf[c][idx] * amp;
        if (i === 0) ctx.moveTo(x, v); else ctx.lineTo(x, v);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (semDado) {
      ctx.fillStyle = 'rgba(125,139,165,.9)';
      ctx.font = (13 * (window.devicePixelRatio || 1)) + 'px "Segoe UI",sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('sem dado da ponte — rode python run.py em hand3d/', w / 2, h / 2);
      ctx.textAlign = 'left';
    }
  }

  // ---------- painel de texto / cartões ----------
  var relogioT0 = Date.now();
  function fmtRelogio(d) {
    return [d.getHours(), d.getMinutes(), d.getSeconds()]
      .map(function (v) { return String(v).padStart(2, '0'); }).join(':');
  }
  function fmtTempo(s) {
    var m = Math.floor(s / 60), sg = Math.floor(s % 60);
    return String(m).padStart(2, '0') + ':' + String(sg).padStart(2, '0');
  }

  function pintar() {
    var vivo = st.fonte === 'myo', sim = st.fonte === 'sim';
    var semDado = performance.now() - st.ultimoDado > 2500;
    var rotulo = semDado ? 'sem ponte'
      : vivo ? 'Myo ao vivo' : sim ? 'ponte em simulação' : 'ponte no ar, sem bracelete';

    el.pFonte.className = 'pill ' + (semDado ? 'off' : vivo ? 'vivo' : sim ? 'sim' : 'off');
    el.pFonte.querySelector('span').textContent = rotulo;
    el.sRotulo.textContent = rotulo;
    el.sFs.textContent = st.fs ? st.fs.toFixed(0) + ' Hz' : '—';
    el.sLat.textContent = st.ultimoPacote ? st.ultimoPacote.toFixed(0) + ' ms' : '—';
    el.sLote.textContent = st.ultimoLote ? st.ultimoLote + ' am.' : '—';
    el.emgInfo.textContent = st.fs ? st.fs.toFixed(0) + ' Hz' : '— Hz';

    el.gNome.textContent = semDado ? '—' : st.nome;
    el.gClasse.textContent = 'classe ' + (st.gesture === null || semDado ? '—' : st.gesture);
    for (var i = 0; i < N_CANAIS; i++) {
      var v = semDado ? 0 : Math.max(0, Math.min(1, st.rms[i] || 0));
      barrinhas[i].style.height = Math.max(2, v * 22) + 'px';
      barrinhas[i].style.background = v > 0.05 ? CORES[i] : 'var(--linha2)';
    }

    el.iRoll.textContent = st.euler[0].toFixed(0) + '°';
    el.iPitch.textContent = st.euler[1].toFixed(0) + '°';
    el.iYaw.textContent = st.euler[2].toFixed(0) + '°';

    el.relogio.textContent = fmtRelogio(new Date());
    el.stTempo.textContent = fmtTempo((Date.now() - relogioT0) / 1000);
    el.stAmostras.textContent = st.totalAmostras.toLocaleString('pt-BR');
    el.stPico.textContent = st.pico.toFixed(0);

    var agora = performance.now();
    st.pacotesJanela = st.pacotesJanela.filter(function (t) { return agora - t < 1000; });
    el.stPacotes.textContent = st.pacotesJanela.length;
  }

  // ---------- laço ----------
  var ultimoQuadro = 0;
  function quadro() {
    requestAnimationFrame(quadro);
    var agora = performance.now();
    var dt = ultimoQuadro ? Math.min(0.05, (agora - ultimoQuadro) / 1000) : 0.016;
    ultimoQuadro = agora;

    var k = 1 - Math.exp(-dt / 0.12);
    for (var i = 0; i < 3; i++) {
      /* Só a LEITURA numérica (roll/pitch/yaw) continua vindo do Euler —
         ali ele é o dado certo, é o que o feed.py mede. O caminho mais
         curto evita o salto quando o ângulo embrulha em ±180. */
      var delta = st.eulerAlvo[i] - st.euler[i];
      delta -= Math.round(delta / 360) * 360;
      st.euler[i] += delta * k;
    }

    /* O CUBO vem do quaternion, não do Euler: mesma cadeia do hand.js
       (Q_IMU_CENA * q_imu), depois convertida pro CSS. Sem calibração de
       sessão de propósito — este painel mostra o SENSOR, não a pose da mão;
       o que importa aqui é que os eixos sejam fisicamente corretos. */
    if (st.quat) {
      st.quatCubo = qSuavizar(st.quatCubo, qMul(Q_IMU_CENA, st.quat), k);
      el.imuCubo.style.transform = cssDeQuat(st.quatCubo);
    }

    desenharEmg();
    pintar();
  }

  redimensionarCanvas();
  conectar();
  quadro();
})();
