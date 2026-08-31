/* ============================================================
   hand.js — a mão do MyoClassifier renderizada no navegador.

   Substitui o projeto Unity (3DORientaion_test) por completo:

     myListener.cs      TCP 25001 -> transform.rotation
       agora:           WebSocket -> pivot.quaternion

     handController.cs  Animator com blend tree Grip x Trigger,
                        alimentado por TECLAS EMULADAS (1-4)
       agora:           AnimationMixer com um peso por clipe,
                        alimentado pelo campo "gesture" do JSON

   O FBX traz as quatro poses (Relaxed, fist, spock, Pointing) como
   clipes de um quadro — que é exatamente o que um blend tree mistura.
   A transição suave sai de interpolar peso, não de animação gravada.

   Sem a ponte no ar, os botões da lateral comandam a pose.
   ============================================================ */
(function () {
  'use strict';

  var WS = 'ws://127.0.0.1:8765';
  var MODELO = 'model/hand.fbx';

  // o FBX vem deitado (Blender é Z-up); isto põe a palma de frente
  var ORIENT = { rx: -90, ry: 0, rz: 90 };
  var CAM = { az: 32, el: 8, zoom: 1.55 };

  /* classe do classificador -> clipe do FBX -> rótulo.
     É o mesmo mapa do GESTOS no bridge.py; mantenha os dois juntos. */
  var POSES = [
    { clip: 'Relaxed',  nome: 'mão aberta',      classe: 1, cor: '#a3e635' },
    { clip: 'fist',     nome: 'punho fechado',   classe: 2, cor: '#22d3ee' },
    { clip: 'spock',    nome: 'dedos separados', classe: 3, cor: '#a78bfa' },
    { clip: 'Pointing', nome: 'apontando',       classe: 4, cor: '#fbbf24' }
  ];

  var CORES_CANAL = ['#22d3ee', '#2dd4bf', '#a3e635', '#fbbf24',
                     '#fb923c', '#fb7185', '#f472b6', '#a78bfa'];

  // ---------- estado ----------
  var st = {
    peso: [1, 0, 0, 0], alvo: [1, 0, 0, 0], idx: 0,
    euler: [0, 0, 0], eulerAlvo: [0, 0, 0],
    rms: null, fonte: 'off', fs: 0, classe: null,
    seguirImu: true, girar: false,
    ws: null, ultimoDado: 0, ultimoQuadro: 0
  };

  var el = {};
  ['fonte', 'gesto', 'clipe', 'euler', 'erro', 'botoes', 'barras',
   'l-fonte', 'l-fs', 'l-classe', 'l-modelo', 'dicaponte'].forEach(function (id) {
    el[id] = document.getElementById(id);
  });

  function falhar(msg) {
    el.erro.style.display = 'block';
    el.erro.innerHTML = msg;
  }

  // ---------- cena ----------
  if (!window.THREE) { falhar('three.js não carregou (vendor/three.min.js).'); return; }
  if (!THREE.FBXLoader) { falhar('FBXLoader não carregou (vendor/FBXLoader.js).'); return; }

  var palco = document.getElementById('palco');
  var tela = document.getElementById('tela');
  var cena = new THREE.Scene();
  var cam = new THREE.PerspectiveCamera(38, 1, 0.01, 8000);
  var ren = new THREE.WebGLRenderer({ canvas: tela, antialias: true, alpha: true });
  ren.setClearColor(0x000000, 0);
  ren.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

  cena.add(new THREE.HemisphereLight(0xbcd4ff, 0x141a26, 0.9));
  var luz = new THREE.DirectionalLight(0xffffff, 0.9); luz.position.set(3, 5, 4);
  var borda = new THREE.DirectionalLight(0x22d3ee, 0.7); borda.position.set(-4, 0.5, -3);
  var preenche = new THREE.DirectionalLight(0xf472b6, 0.18); preenche.position.set(2, -3, -2);
  cena.add(luz, borda, preenche);

  var orbita = new THREE.OrbitControls(cam, ren.domElement);
  orbita.enableDamping = true;
  orbita.dampingFactor = 0.09;

  // pivot = o que a IMU gira. orient = correção de eixo, uma vez só.
  var pivot = new THREE.Group(); cena.add(pivot);
  var orient = new THREE.Group(); pivot.add(orient);
  var d2r = Math.PI / 180;
  orient.rotation.set(ORIENT.rx * d2r, ORIENT.ry * d2r, ORIENT.rz * d2r);

  var modelo = null, mixer = null, acoes = [], esqueleto = null, maxDim = 1;

  function redimensionar() {
    var w = palco.clientWidth, h = palco.clientHeight;
    if (!w || !h) return;
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    ren.setSize(w, h, false);
  }
  window.addEventListener('resize', redimensionar);

  function enquadrar() {
    if (!modelo) return;
    var dist = maxDim / (2 * Math.tan(cam.fov * Math.PI / 360)) * CAM.zoom;
    var a = CAM.az * d2r, e = CAM.el * d2r;
    cam.position.set(dist * Math.cos(e) * Math.sin(a),
                     dist * Math.sin(e),
                     dist * Math.cos(e) * Math.cos(a));
    // mira um pouco acima do centro: a mao desce no quadro e sobra menos
    // vazio embaixo
    orbita.target.set(0, maxDim * 0.07, 0);
    orbita.update();
  }

  // ---------- carrega o modelo ----------
  new THREE.FBXLoader().load(MODELO, function (obj) {
    var ossos = 0;
    obj.traverse(function (n) {
      if (n.isBone) ossos++;
      if (n.isMesh) {
        n.material = new THREE.MeshStandardMaterial({
          color: 0xdde4ee, roughness: 0.46, metalness: 0.16
        });
      }
    });
    orient.add(obj);
    modelo = obj;

    var caixa = new THREE.Box3().setFromObject(obj);
    var tam = caixa.getSize(new THREE.Vector3());
    obj.position.sub(caixa.getCenter(new THREE.Vector3()));
    maxDim = Math.max(tam.x, tam.y, tam.z) || 1;
    enquadrar();

    var clipes = obj.animations || [];
    mixer = new THREE.AnimationMixer(obj);
    var faltando = [];
    acoes = POSES.map(function (p) {
      var c = null;
      for (var i = 0; i < clipes.length; i++) {
        if (clipes[i].name.toLowerCase().indexOf(p.clip.toLowerCase()) >= 0) {
          c = clipes[i]; break;
        }
      }
      if (!c) { faltando.push(p.clip); return null; }
      var a = mixer.clipAction(c);
      a.play();
      a.time = c.duration * 0.999;   // clipe de um quadro: amostra o fim
      a.setEffectiveWeight(0);
      a.paused = true;
      return a;
    });
    if (faltando.length) {
      falhar('O FBX não tem o(s) clipe(s): <b>' + faltando.join(', ') + '</b>.<br>' +
             'Clipes encontrados: ' + (clipes.map(function (c) { return c.name; })
               .join(', ') || 'nenhum') + '.');
    }
    el['l-modelo'].textContent = ossos + ' / ' + clipes.length;
    porPose(0, false);
    conectar();
  }, null, function () {
    falhar('Não consegui carregar <code>' + MODELO + '</code>.<br>' +
           'Rode pelo <code>serve.py</code> (abrir o arquivo direto não funciona: ' +
           'o navegador bloqueia a leitura local).');
  });

  // ---------- poses ----------
  function porPose(i, manual) {
    if (i < 0 || i >= POSES.length) return;
    st.idx = i;
    st.alvo = POSES.map(function (_, k) { return k === i ? 1 : 0; });
    if (manual) st.fonte = st.fonte === 'off' ? 'off' : 'manual';
    var p = POSES[i];
    el.gesto.textContent = p.nome;
    el.gesto.style.color = p.cor;
    el.clipe.textContent = 'clipe "' + p.clip + '" · classe ' + p.classe;
    Array.prototype.forEach.call(el.botoes.children, function (b, k) {
      b.classList.toggle('on', k === i);
    });
  }

  POSES.forEach(function (p, i) {
    var b = document.createElement('button');
    b.textContent = p.nome;
    b.onclick = function () { porPose(i, true); };
    el.botoes.appendChild(b);
  });

  document.addEventListener('keydown', function (ev) {
    var n = parseInt(ev.key, 10);
    if (n >= 1 && n <= POSES.length) porPose(n - 1, true);
  });

  // ---------- barras de EMG ----------
  var cheios = [];
  for (var c = 0; c < 8; c++) {
    var linha = document.createElement('div');
    linha.className = 'barra';
    linha.innerHTML = '<span>c' + (c + 1) + '</span>' +
      '<div class="trilho"><div class="cheio" style="width:0;background:' +
      CORES_CANAL[c] + '"></div></div><span class="v">0</span>';
    el.barras.appendChild(linha);
    cheios.push([linha.querySelector('.cheio'), linha.querySelector('.v')]);
  }

  // ---------- ponte ----------
  function conectar() {
    if (!('WebSocket' in window)) return;
    if (st.ws && (st.ws.readyState === 0 || st.ws.readyState === 1)) return;
    var ws;
    try { ws = new WebSocket(WS); } catch (e) { setTimeout(conectar, 4000); return; }
    st.ws = ws;
    ws.onclose = function () { setTimeout(conectar, 4000); };
    ws.onmessage = function (ev) {
      var d;
      try { d = JSON.parse(ev.data); } catch (e) { return; }
      st.ultimoDado = performance.now();
      // a ponte pode estar no ar SEM bracelete: nesse caso src vem "—".
      // Tratar tudo que nao e 'sim' como 'myo' faria a tela mentir.
      st.fonte = d.src === 'myo' ? 'myo' : d.src === 'sim' ? 'sim' : 'ponte';
      if (d.fs) st.fs = +d.fs;
      if (d.gesture !== undefined) st.classe = +d.gesture;
      if (d.euler && d.euler.length >= 3) {
        st.eulerAlvo = [+d.euler[0], +d.euler[1], +d.euler[2]];
      }
      if (d.rms && d.rms.length) st.rms = d.rms.map(Number);
      if (d.name) {
        for (var i = 0; i < POSES.length; i++) {
          if (POSES[i].clip.toLowerCase() === String(d.name).toLowerCase()) {
            if (i !== st.idx) porPose(i, false);
            break;
          }
        }
      }
    };
  }

  // ---------- botões visuais ----------
  document.getElementById('b-imu').onclick = function () {
    st.seguirImu = !st.seguirImu;
    this.classList.toggle('on', st.seguirImu);
  };
  document.getElementById('b-girar').onclick = function () {
    st.girar = !st.girar;
    this.classList.toggle('on', st.girar);
  };
  document.getElementById('b-osso').onclick = function () {
    if (!modelo) return;
    if (esqueleto) { cena.remove(esqueleto); esqueleto = null; this.classList.remove('on'); return; }
    esqueleto = new THREE.SkeletonHelper(modelo);
    cena.add(esqueleto);
    this.classList.add('on');
  };
  document.getElementById('b-fio').onclick = function () {
    if (!modelo) return;
    var ligado = false;
    modelo.traverse(function (n) {
      if (n.isMesh) { n.material.wireframe = !n.material.wireframe; ligado = n.material.wireframe; }
    });
    this.classList.toggle('on', ligado);
  };
  document.getElementById('b-camera').onclick = enquadrar;

  // ---------- laço ----------
  function pintarPainel() {
    var vivo = st.fonte === 'myo', sim = st.fonte === 'sim';
    var rot = vivo ? 'Myo ao vivo'
      : sim ? 'ponte em simulação'
        : st.fonte === 'ponte' ? 'ponte no ar, sem dado do bracelete'
          : st.fonte === 'manual' ? 'botões desta página'
            : 'sem ponte';
    el.fonte.className = 'fonte ' + (vivo ? 'vivo' : sim ? 'sim' : 'off');
    el.fonte.querySelector('span').textContent = rot;
    el['l-fonte'].textContent = rot;
    el['l-fs'].textContent = st.fs ? st.fs.toFixed(0) + ' Hz' : '—';
    el['l-classe'].textContent = st.classe === null ? '—' : st.classe;
    el.euler.textContent = st.euler.map(function (v) { return v.toFixed(0) + '°'; }).join(' ');
    el.dicaponte.style.display = (vivo || sim) ? 'none' : 'block';
    if (st.rms) {
      for (var i = 0; i < 8; i++) {
        var v = Math.max(0, Math.min(1, st.rms[i] || 0));
        cheios[i][0].style.width = (v * 100).toFixed(0) + '%';
        cheios[i][1].textContent = (v * 100).toFixed(0);
      }
    }
  }

  var ultimoPainel = 0;
  function quadro() {
    requestAnimationFrame(quadro);
    var agora = performance.now();
    var dt = st.ultimoQuadro ? Math.min(0.05, (agora - st.ultimoQuadro) / 1000) : 0.016;
    st.ultimoQuadro = agora;

    // 2,5 s sem dado: volta pro manual (aguenta um engasgo do BLE)
    if (st.fonte !== 'off' && st.fonte !== 'manual' && agora - st.ultimoDado > 2500) {
      st.fonte = 'off';
    }

    if (tela.width !== palco.clientWidth || tela.height !== palco.clientHeight) {
      redimensionar();
    }

    // pesos caminham suavemente: é o blend do Animator
    var k = 1 - Math.exp(-dt / 0.10);
    for (var i = 0; i < st.peso.length; i++) {
      st.peso[i] += (st.alvo[i] - st.peso[i]) * k;
      if (acoes[i]) acoes[i].setEffectiveWeight(st.peso[i]);
    }
    if (mixer) mixer.update(0);        // pose estática: não avança o tempo

    var ke = 1 - Math.exp(-dt / 0.12);
    var segue = st.seguirImu && (st.fonte === 'myo' || st.fonte === 'sim');
    for (i = 0; i < 3; i++) {
      st.euler[i] += ((segue ? st.eulerAlvo[i] : 0) - st.euler[i]) * ke;
    }
    // ordem roll, yaw, pitch: mesma que o myListener.cs aplicava
    pivot.rotation.set(st.euler[0] * d2r, st.euler[2] * d2r, st.euler[1] * d2r);
    if (st.girar) pivot.rotation.y += dt * 0.55;

    if (agora - ultimoPainel > 120) { ultimoPainel = agora; pintarPainel(); }

    orbita.update();
    ren.render(cena, cam);
  }

  redimensionar();
  conectar();
  quadro();
})();
