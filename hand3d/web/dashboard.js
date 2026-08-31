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

  var st = {
    ws: null, ultimoDado: 0, ultimoPacote: 0,
    fonte: 'off', fs: 0, gesture: null, nome: '—',
    euler: [0, 0, 0], eulerAlvo: [0, 0, 0],
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
      // caminho mais curto: evita o "salto" quando o angulo embrulha em
      // +-180 (mesmo ajuste de hand.js — ver feed.py:Desembrulhador)
      var delta = st.eulerAlvo[i] - st.euler[i];
      delta -= Math.round(delta / 360) * 360;
      st.euler[i] += delta * k;
    }
    // mesma ordem de eixos usada em hand.js: roll, yaw(Y), pitch(X)
    el.imuCubo.style.transform =
      'rotateX(' + (-st.euler[1]).toFixed(1) + 'deg) ' +
      'rotateY(' + st.euler[2].toFixed(1) + 'deg) ' +
      'rotateZ(' + st.euler[0].toFixed(1) + 'deg)';

    desenharEmg();
    pintar();
  }

  redimensionarCanvas();
  conectar();
  quadro();
})();
