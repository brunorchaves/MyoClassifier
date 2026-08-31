/* ============================================================
   treino.js — aba "treinamento" do painel (dashboard.html).

   Não abre uma segunda conexão com a ponte: usa o hook window.onLoteEmg
   (chamado por dashboard.js a cada lote de EMG recebido) e o hook
   window.getEstadoPonte (mesma heurística de "sem dado" de dashboard.js).

   Fluxo: escolha do gesto -> instrução -> contagem regressiva -> gravação
   (8s) -> revisão -> salvar (POST /api/gravar em serve.py) -> confirmação.
   Nada é gravado em disco até o usuário confirmar em "revisão".
   ============================================================ */
(function () {
  'use strict';

  var CLASSES = [0, 1, 2, 3, 4];
  var REPOUSO_FIXO = { classe: 0, nome: 'Repouso', cor: '#7d8ba5' };
  var DURACAO_GRAVACAO_S = 8;
  var MIN_AMOSTRAS_SESSAO = 150;
  var JANELA_SOMA = 300;          // ~6s a 50Hz

  var st = {
    fase: 'escolha',
    gestos: null,               // {0:{classe,nome,cor}, 1:{...}, ...}
    statusPorClasse: {},        // {0: n amostras, ...} de /api/dataset-status
    classeSel: null,
    amostrasSessao: [],
    inicioGravacao: 0,
    tickGravacao: null,
    tickContagem: null,
    salvando: false,
  };

  // ring buffer da soma dos 8 canais (1 "canal virtual"), mesma ideia de
  // empilhar()/escala automática em dashboard.js.
  var somaBuf = new Float32Array(JANELA_SOMA);
  var somaEscreverEm = 0;
  var somaEscala = 400;

  function empilharSoma(v) {
    somaBuf[somaEscreverEm % JANELA_SOMA] = v;
    somaEscreverEm++;
    somaEscala = v > somaEscala ? v : somaEscala * 0.995 + 400 * 0.005;
  }

  // ---------- elementos ----------
  var el = {};
  [
    'treinoView', 'treinoInfoPonte',
    'seletorBotao', 'seletorRotulo', 'seletorLista',
    'datasetLinhas', 'datasetTotal',
    'prepNome', 'somaCanvas', 'medidorPreenchimento', 'prepSemSinal', 'prepVoltar', 'prepComecar',
    'contagemNumero', 'contagemCancelar',
    'gravNormal', 'gravTempo', 'gravProgresso', 'somaCanvasGrande', 'medidorPreenchimentoGrande',
    'gravContador', 'gravParar', 'gravErro', 'gravErroVoltar',
    'revCanvas', 'revResumo', 'revAviso', 'revRegravar', 'revOutro', 'revSalvar',
    'salvoOk', 'salvoResumo', 'salvoMais', 'salvoOutro',
    'salvoErro', 'salvoErroMsg', 'salvoTentarDeNovo',
  ].forEach(function (id) { el[id] = document.getElementById(id); });

  var ctxSoma = el.somaCanvas.getContext('2d');
  var ctxSomaGrande = el.somaCanvasGrande.getContext('2d');
  var ctxRev = el.revCanvas.getContext('2d');

  // ---------- dados: nomes/cores dos gestos + status do dataset ----------
  function carregarGestos(cb) {
    fetch('/gestos.json').then(function (r) { return r.json(); }).then(function (j) {
      var mapa = {};
      mapa[0] = REPOUSO_FIXO;
      (j.ordem || []).forEach(function (g) {
        if (CLASSES.indexOf(g.classe) !== -1) mapa[g.classe] = g;
      });
      st.gestos = mapa;
      cb();
    }).catch(function () {
      var mapa = {};
      CLASSES.forEach(function (c) { mapa[c] = { classe: c, nome: 'gesto ' + c, cor: '#7d8ba5' }; });
      st.gestos = mapa;
      cb();
    });
  }

  function carregarStatus(cb) {
    fetch('/api/dataset-status').then(function (r) { return r.json(); }).then(function (j) {
      st.statusPorClasse = {};
      (j.classes || []).forEach(function (c) { st.statusPorClasse[c.classe] = c.amostras; });
      if (cb) cb();
    }).catch(function () { if (cb) cb(); });
  }

  function nomeGesto(classe) {
    var g = (st.gestos && st.gestos[classe]) || {};
    return g.nome || ('gesto ' + classe);
  }

  // ---------- fase: escolha (seletor tipo dropdown) ----------
  function renderSeletor() {
    el.seletorLista.innerHTML = '';
    CLASSES.forEach(function (c) {
      var g = (st.gestos && st.gestos[c]) || { nome: 'gesto ' + c, cor: '#7d8ba5' };
      var n = st.statusPorClasse[c] || 0;
      var linha = document.createElement('button');
      linha.type = 'button';
      linha.className = 'linha-gesto' + (n === 0 ? ' vazio' : '');
      linha.innerHTML =
        '<span class="pontinho" style="background:' + g.cor + '"></span>' +
        '<span class="nome">' + g.nome + '</span>' +
        '<span class="contagem">' + (n === 0 ? 'sem dados' : n.toLocaleString('pt-BR') + ' am.') + '</span>';
      linha.addEventListener('click', function () {
        st.classeSel = c;
        fecharSeletor();
        atualizarRotuloSeletor();
        renderPainelDataset();
        irPara('preparar');
      });
      el.seletorLista.appendChild(linha);
    });
  }

  function atualizarRotuloSeletor() {
    if (st.classeSel === null) {
      el.seletorRotulo.innerHTML = '<span class="placeholder">escolher gesto</span>';
      return;
    }
    var g = (st.gestos && st.gestos[st.classeSel]) || { nome: 'gesto ' + st.classeSel, cor: '#7d8ba5' };
    el.seletorRotulo.innerHTML = '<span class="pontinho" style="background:' + g.cor + '"></span>' + g.nome;
  }

  function abrirSeletor() { el.seletorLista.hidden = false; el.seletorBotao.classList.add('aberto'); }
  function fecharSeletor() { el.seletorLista.hidden = true; el.seletorBotao.classList.remove('aberto'); }
  el.seletorBotao.addEventListener('click', function (e) {
    e.stopPropagation();
    if (el.seletorLista.hidden) abrirSeletor(); else fecharSeletor();
  });
  document.addEventListener('click', function (e) {
    if (!el.seletorLista.hidden && e.target !== el.seletorBotao && !el.seletorLista.contains(e.target)) {
      fecharSeletor();
    }
  });

  // ---------- painel do dataset (coluna direita, sempre visivel) ----------
  function renderPainelDataset() {
    el.datasetLinhas.innerHTML = '';
    var total = 0;
    var maxN = 1;
    CLASSES.forEach(function (c) { maxN = Math.max(maxN, st.statusPorClasse[c] || 0); });
    CLASSES.forEach(function (c) {
      var g = (st.gestos && st.gestos[c]) || { nome: 'gesto ' + c, cor: '#7d8ba5' };
      var n = st.statusPorClasse[c] || 0;
      total += n;
      var pct = Math.round((n / maxN) * 100);
      var linha = document.createElement('div');
      linha.className = 'dataset-linha' + (n === 0 ? ' vazio' : '') + (c === st.classeSel ? ' atual' : '');
      linha.innerHTML =
        '<span class="pontinho" style="background:' + g.cor + '"></span>' +
        '<div class="info"><div class="nome">' + g.nome + '</div>' +
        '<div class="barra"><div class="preenchimento" style="width:' + pct + '%;background:' + g.cor + '"></div></div></div>' +
        '<div class="numero">' + n.toLocaleString('pt-BR') + '</div>';
      el.datasetLinhas.appendChild(linha);
    });
    el.datasetTotal.textContent = total.toLocaleString('pt-BR');
  }

  // ---------- máquina de estados ----------
  function irPara(fase) {
    st.fase = fase;
    document.querySelectorAll('.fase-treino').forEach(function (elFase) {
      elFase.hidden = elFase.dataset.fase !== fase;
    });
    renderPainelDataset();     // atualiza o destaque da linha "atual" a cada troca de fase
    if (fase === 'escolha') carregarStatus(function () { renderSeletor(); renderPainelDataset(); });
    else if (fase === 'preparar') entrarPreparar();
    else if (fase === 'contagem') entrarContagem();
    else if (fase === 'gravando') entrarGravando();
    else if (fase === 'revisao') entrarRevisao();
  }

  function entrarPreparar() {
    el.prepNome.textContent = nomeGesto(st.classeSel);
    atualizarBloqueioPreparar();
  }

  function atualizarBloqueioPreparar() {
    if (st.fase !== 'preparar') return;
    var estado = window.getEstadoPonte ? window.getEstadoPonte() : { semDado: true };
    el.prepSemSinal.hidden = !estado.semDado;
    el.prepComecar.disabled = estado.semDado;
  }

  function entrarContagem() {
    var n = 3;
    el.contagemNumero.textContent = String(n);
    st.tickContagem = setInterval(function () {
      n--;
      if (n <= 0) {
        clearInterval(st.tickContagem);
        st.tickContagem = null;
        irPara('gravando');
        return;
      }
      el.contagemNumero.textContent = String(n);
    }, 800);
  }

  function entrarGravando() {
    el.gravNormal.hidden = false;
    el.gravErro.hidden = true;
    st.amostrasSessao = [];
    st.inicioGravacao = performance.now();
    el.gravContador.textContent = '0';
    el.gravProgresso.style.width = '0%';
    el.gravTempo.textContent = DURACAO_GRAVACAO_S.toFixed(1) + 's';

    st.tickGravacao = setInterval(function () {
      var passado = (performance.now() - st.inicioGravacao) / 1000;
      var restante = Math.max(0, DURACAO_GRAVACAO_S - passado);
      el.gravTempo.textContent = restante.toFixed(1) + 's';
      el.gravProgresso.style.width = Math.min(100, (passado / DURACAO_GRAVACAO_S) * 100) + '%';

      var estado = window.getEstadoPonte ? window.getEstadoPonte() : { semDado: true };
      if (estado.semDado) {
        pararGravacao();
        el.gravNormal.hidden = true;
        el.gravErro.hidden = false;
        return;
      }
      if (passado >= DURACAO_GRAVACAO_S) {
        pararGravacao();
        irPara('revisao');
      }
    }, 100);
  }

  function pararGravacao() {
    if (st.tickGravacao) { clearInterval(st.tickGravacao); st.tickGravacao = null; }
  }

  function entrarRevisao() {
    var n = st.amostrasSessao.length;
    var fs = (window.getEstadoPonte && window.getEstadoPonte().fs) || 50;
    var dur = fs ? n / fs : 0;
    var somas = st.amostrasSessao.map(function (a) { return a.reduce(function (x, y) { return x + y; }, 0); });
    var pico = somas.length ? Math.max.apply(null, somas) : 0;
    var media = somas.length ? somas.reduce(function (x, y) { return x + y; }, 0) / somas.length : 0;

    el.revResumo.textContent = n.toLocaleString('pt-BR') + ' amostras capturadas (~' + dur.toFixed(1) +
      's) · pico ' + pico.toFixed(0) + ' · média ' + media.toFixed(0);
    el.revAviso.hidden = !(n > 0 && n < MIN_AMOSTRAS_SESSAO);
    el.revSalvar.disabled = n === 0;
    desenharRevisao(somas);
  }

  function desenharRevisao(somas) {
    var canvas = el.revCanvas;
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    var w = canvas.width, h = canvas.height;
    ctxRev.clearRect(0, 0, w, h);
    if (!somas.length) return;
    var max = Math.max.apply(null, somas.concat([1]));
    ctxRev.strokeStyle = '#22d3ee';
    ctxRev.lineWidth = 1.6 * dpr;
    ctxRev.beginPath();
    somas.forEach(function (v, i) {
      var x = (i / (somas.length - 1 || 1)) * w;
      var y = h - (v / max) * h * 0.9 - h * 0.05;
      if (i === 0) ctxRev.moveTo(x, y); else ctxRev.lineTo(x, y);
    });
    ctxRev.stroke();
  }

  // ---------- captura ao vivo (hook chamado por dashboard.js) ----------
  window.onLoteEmg = function (lote) {
    lote.forEach(function (amostra) {
      if (!Array.isArray(amostra) || amostra.length < 8) return;
      var oito = amostra.slice(0, 8).map(function (v) {
        return Math.max(0, Math.min(65535, Math.round(+v) || 0));
      });
      empilharSoma(oito.reduce(function (a, b) { return a + b; }, 0));
      if (st.fase === 'gravando' && st.tickGravacao) {
        st.amostrasSessao.push(oito);
        el.gravContador.textContent = st.amostrasSessao.length.toLocaleString('pt-BR');
      }
    });
  };

  // ---------- desenho ao vivo da soma (fases preparar/gravando) ----------
  function desenharSomaAoVivo() {
    if (el.treinoView.hidden) return;
    var canvas = null, medidor = null;
    if (st.fase === 'preparar') { canvas = el.somaCanvas; medidor = el.medidorPreenchimento; }
    else if (st.fase === 'gravando') { canvas = el.somaCanvasGrande; medidor = el.medidorPreenchimentoGrande; }
    if (!canvas) return;

    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var r = canvas.getBoundingClientRect();
    var wPx = Math.max(1, Math.round(r.width * dpr)), hPx = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== wPx) canvas.width = wPx;
    if (canvas.height !== hPx) canvas.height = hPx;
    var ctx = canvas === el.somaCanvas ? ctxSoma : ctxSomaGrande;
    var w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    var n = Math.min(JANELA_SOMA, somaEscreverEm);
    var ultimo = 0;
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (var i = 0; i < n; i++) {
      var idx = (somaEscreverEm - n + i) % JANELA_SOMA;
      var v = somaBuf[idx];
      if (i === n - 1) ultimo = v;
      var x = (i / (JANELA_SOMA - 1)) * w;
      var y = h - Math.min(1, v / (somaEscala || 1)) * h * 0.88 - h * 0.04;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (medidor) {
      var frac = Math.max(0, Math.min(1, ultimo / (somaEscala * 1.2 || 1)));
      medidor.style.height = (frac * 100) + '%';
      medidor.style.background = frac < 0.3 ? '#a3e635' : frac < 0.7 ? '#fbbf24' : '#fb7185';
    }
  }

  // ---------- salvar ----------
  function salvar() {
    if (st.salvando) return;
    st.salvando = true;
    el.revSalvar.disabled = true;
    irPara('salvando');
    fetch('/api/gravar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classe: st.classeSel, amostras: st.amostrasSessao }),
    }).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, j: j }; });
    }).then(function (res) {
      st.salvando = false;
      if (!res.ok || !res.j.ok) {
        mostrarErroSalvar((res.j && res.j.erro) || 'erro desconhecido ao salvar');
        return;
      }
      st.statusPorClasse[st.classeSel] = res.j.total_amostras;
      renderPainelDataset();
      el.salvoOk.hidden = false;
      el.salvoErro.hidden = true;
      el.salvoResumo.textContent = res.j.gravadas.toLocaleString('pt-BR') + ' amostras salvas em "' +
        nomeGesto(st.classeSel) + '" · total agora: ' + res.j.total_amostras.toLocaleString('pt-BR');
      irPara('salvo');
    }).catch(function () {
      st.salvando = false;
      mostrarErroSalvar('falha de rede ao salvar — tente de novo');
    });
  }

  function mostrarErroSalvar(msg) {
    el.salvoOk.hidden = true;
    el.salvoErro.hidden = false;
    el.salvoErroMsg.textContent = msg;
    irPara('salvo');
  }

  // ---------- botões ----------
  el.prepVoltar.addEventListener('click', function () { irPara('escolha'); });
  el.prepComecar.addEventListener('click', function () { if (!el.prepComecar.disabled) irPara('contagem'); });
  el.contagemCancelar.addEventListener('click', function () {
    if (st.tickContagem) { clearInterval(st.tickContagem); st.tickContagem = null; }
    irPara('preparar');
  });
  el.gravParar.addEventListener('click', function () { pararGravacao(); irPara('revisao'); });
  el.gravErroVoltar.addEventListener('click', function () { irPara('preparar'); });
  el.revRegravar.addEventListener('click', function () { irPara('preparar'); });
  el.revOutro.addEventListener('click', function () { irPara('escolha'); });
  el.revSalvar.addEventListener('click', salvar);
  el.salvoMais.addEventListener('click', function () { irPara('preparar'); });
  el.salvoOutro.addEventListener('click', function () { irPara('escolha'); });
  el.salvoTentarDeNovo.addEventListener('click', salvar);

  var abas = document.getElementById('abas');
  if (abas) {
    abas.addEventListener('click', function (e) {
      var btn = e.target.closest('button.aba');
      if (!btn) return;
      var alvo = btn.dataset.aba;
      abas.querySelectorAll('button').forEach(function (b) { b.classList.toggle('on', b === btn); });
      document.getElementById('mainView').hidden = alvo !== 'visualizacao';
      document.getElementById('footerView').hidden = alvo !== 'visualizacao';
      el.treinoView.hidden = alvo !== 'treinamento';
    });
  }

  // ---------- laço de desenho ----------
  function atualizarInfoPonte() {
    if (el.treinoView.hidden) return;
    var estado = window.getEstadoPonte ? window.getEstadoPonte() : { semDado: true };
    el.treinoInfoPonte.textContent = estado.semDado ? 'sem ponte' : (estado.fs ? estado.fs.toFixed(0) + ' Hz' : 'conectado');
  }

  function quadro() {
    requestAnimationFrame(quadro);
    desenharSomaAoVivo();
    atualizarInfoPonte();
    if (st.fase === 'preparar') atualizarBloqueioPreparar();
  }

  carregarGestos(function () {
    atualizarRotuloSeletor();
    carregarStatus(function () { renderSeletor(); renderPainelDataset(); });
  });
  quadro();
})();
