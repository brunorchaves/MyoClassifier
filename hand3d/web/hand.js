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

  /* classe do classificador -> clipe do FBX -> rótulo. Vem de gestos.json,
     fonte única (bridge.py e desktop.py leem o mesmo arquivo). Preenchido
     pelo fetch no fim deste arquivo, antes de qualquer coisa que precise
     dele (botões, teclado, FBXLoader). */
  var POSES = null;
  /* poses desenhadas por osso (nao clipes do FBX) — gestos.json:extras.
     So entram no cache (exportarPoses), nunca nos botoes/teclado da pagina
     — ver README.md, "Creating new poses". */
  var EXTRAS = [];

  var CORES_CANAL = ['#22d3ee', '#2dd4bf', '#a3e635', '#fbbf24',
                     '#fb923c', '#fb7185', '#f472b6', '#a78bfa'];

  // ---------- estado ----------
  /* pose "inicial" (roll,pitch,yaw): dorso da mao pra cima / palma pra
     baixo, dedos esticados pra longe da camera — medido tentando varias
     combinacoes com os eixos de mundo desenhados (AxesHelper) e olhando o
     "apontando" (o dedo reto deixa a direcao inequivoca) e o "punho
     fechado" (os nos dos dedos pra cima confirmam palma pra baixo). Tecla
     espaco recalibra pra isto — ver ESPACO abaixo. */
  var EULER_INICIAL = [0, -90, 0];

  var st = {
    peso: [1, 0, 0, 0], alvo: [1, 0, 0, 0], idx: 0,
    euler: EULER_INICIAL.slice(), eulerAlvo: EULER_INICIAL.slice(),
    eulerBruto: [0, 0, 0], offsetCalib: [0, 0, 0],
    rms: null, fonte: 'off', fs: 0, classe: null,
    seguirImu: true, girar: false,
    ws: null, ultimoDado: 0, ultimoQuadro: 0,
    previsualizando: false   // previsualizarPoseCustom(): pausa a mistura normal
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
  // onde a camera mira, em coordenadas de mundo (fica em (0,0,0) ate o
  // modelo carregar e recalcular a partir do novo pivo do pulso)
  var foco = new THREE.Vector3();

  /* ---------- exportar poses (modo desktop, rota B) ----------
     O assimp-py nao expoe ossos/pesos/animacoes deste FBX (so malha estatica
     — ver hand3d/PLANO-desktop.md). A fonte dos dados passa a ser o proprio
     three.js: para cada uma das 4 poses, calcula a posicao e a normal JA
     SKINADAS por vertice (a mesma formula do skinning_vertex.glsl.js do
     three.js: skinned = bindMatrixInverse * Sum(peso_i * boneMatriz_i * bindMatrix
     * vertice)) e devolve tudo em base64. O modo desktop so interpola essas
     4 posicoes por vertice — perde a mistura em espaco de osso, ganha
     simplicidade (nao precisa reimplementar leitura de FBX em Python). */
  /* dedo -> cadeia de ossos, da raiz (metacarpo) a ponta. Identificado por
     investigacao (ver conversa sobre poses novas): indicador e polegar por
     comportamento nas 4 poses conhecidas (indicador fica reto em
     "Pointing"; polegar e o unico que fica quase reto em "spock" e dobra
     em "fist" E em "Pointing"); medio/anelar/minimo por comprimento total
     da cadeia na bind pose (anatomia: medio > anelar > minimo). */
  var DEDOS = {
    polegar: ['Bone005', 'Bone006', 'Bone019'],
    indicador: ['Bone004', 'Bone016', 'Bone017', 'Bone018'],
    medio: ['Bone003', 'Bone007', 'Bone008', 'Bone009'],
    anelar: ['Bone002', 'Bone010', 'Bone011', 'Bone012'],
    minimo: ['Bone001', 'Bone013', 'Bone014', 'Bone015']
  };

  function b64DeFloat32(arr) {
    var bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    var bin = '', CH = 0x8000;
    for (var i = 0; i < bytes.length; i += CH) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    }
    return btoa(bin);
  }

  /* Skina (three.js: skinning_vertex.glsl.js/skinnormal_vertex.glsl.js) a
     malha no estado ATUAL do esqueleto (chame malha.skeleton.update() antes)
     e leva pro espaco "mundo com pivot zerado" via matrizMundo (ver
     acharMatrizMundo). Reutilizado por exportarPoses() e por
     criarPoseCustom(). */
  function skinarPoseAtual(malha, matrizMundo, empurrar) {
    var geo = malha.geometry;
    var pos = geo.attributes.position;
    var nrm = geo.attributes.normal;
    var skinIdx = geo.attributes.skinIndex;
    var skinWt = geo.attributes.skinWeight;
    var N = pos.count;
    var bm = malha.skeleton.boneMatrices;

    // empurrar: [{indicesOsso: [i,...], vetor: Vector3}] — desloca depois
    // do skinning normal, proporcional ao peso total nesses ossos (0 na
    // base do dedo, ate 1 na ponta). Nao gira nada, so translada — sem
    // costura de junta, porque nao depende de peso pintado pra flexao.
    empurrar = empurrar || [];

    var bindPos = new THREE.Vector4();
    var bindNrm = new THREE.Vector4();
    var boneMat = new THREE.Matrix4();
    var acumPos = new THREE.Vector4();
    var acumNrm = new THREE.Vector4();
    var cp = new THREE.Vector4();
    var cn = new THREE.Vector4();
    var getComp = ['getX', 'getY', 'getZ', 'getW'];
    function comp(attr, v, j) { return attr[getComp[j]](v); }

    var positions = new Float32Array(N * 3);
    var normals = new Float32Array(N * 3);
    for (var v = 0; v < N; v++) {
      bindPos.set(pos.getX(v), pos.getY(v), pos.getZ(v), 1).applyMatrix4(malha.bindMatrix);
      bindNrm.set(nrm.getX(v), nrm.getY(v), nrm.getZ(v), 0).applyMatrix4(malha.bindMatrix);
      acumPos.set(0, 0, 0, 0);
      acumNrm.set(0, 0, 0, 0);
      for (var j = 0; j < 4; j++) {
        var w = comp(skinWt, v, j);
        if (!w) continue;
        var bi = comp(skinIdx, v, j);
        boneMat.fromArray(bm, bi * 16);
        cp.copy(bindPos).applyMatrix4(boneMat);
        cn.copy(bindNrm).applyMatrix4(boneMat);
        acumPos.x += cp.x * w; acumPos.y += cp.y * w; acumPos.z += cp.z * w;
        acumNrm.x += cn.x * w; acumNrm.y += cn.y * w; acumNrm.z += cn.z * w;
      }
      acumPos.applyMatrix4(malha.bindMatrixInverse);
      acumNrm.applyMatrix4(malha.bindMatrixInverse);
      // leva pro espaco "de mundo com pivot zerado": ja inclui a
      // centralizacao e o ORIENT, fixos, calculados uma vez em acharMatrizMundo
      acumPos.applyMatrix4(matrizMundo);
      acumNrm.applyMatrix4(matrizMundo);
      for (var e = 0; e < empurrar.length; e++) {
        var pesoDedo = 0;
        for (var j2 = 0; j2 < 4; j2++) {
          if (empurrar[e].indicesOsso.indexOf(comp(skinIdx, v, j2)) >= 0) {
            pesoDedo += comp(skinWt, v, j2);
          }
        }
        if (pesoDedo) {
          acumPos.x += empurrar[e].vetor.x * pesoDedo;
          acumPos.y += empurrar[e].vetor.y * pesoDedo;
          acumPos.z += empurrar[e].vetor.z * pesoDedo;
        }
      }
      var nlen = Math.hypot(acumNrm.x, acumNrm.y, acumNrm.z) || 1;
      positions[v * 3] = acumPos.x; positions[v * 3 + 1] = acumPos.y; positions[v * 3 + 2] = acumPos.z;
      normals[v * 3] = acumNrm.x / nlen; normals[v * 3 + 1] = acumNrm.y / nlen; normals[v * 3 + 2] = acumNrm.z / nlen;
    }
    return { position_f32: b64DeFloat32(positions), normal_f32: b64DeFloat32(normals) };
  }

  function acharMalha() {
    var malha = null;
    modelo.traverse(function (n) { if (n.isSkinnedMesh && !malha) malha = n; });
    return malha;
  }

  /* pivot em zero: o export ja inclui a centralizacao (obj.position) e a
     correcao de eixo (ORIENT), via malha.matrixWorld — assim o desktop.py
     so precisa aplicar a rotacao vinda da IMU em cima disto, sem reimplementar
     ORIENT nem a conta de centralizacao. Chame com o pivot original salvo;
     devolve a matriz (calculada uma vez, nao muda entre poses). */
  function acharMatrizMundo(malha) {
    var original = pivot.rotation.clone();
    pivot.rotation.set(0, 0, 0);
    cena.updateMatrixWorld(true);
    var m = malha.matrixWorld.clone();
    pivot.rotation.copy(original);
    return m;
  }

  function exportarPoses() {
    if (!modelo || !acoes.length || !POSES) return { erro: 'modelo ainda nao carregou' };
    var malha = acharMalha();
    if (!malha) return { erro: 'nenhum SkinnedMesh no modelo' };

    var geo = malha.geometry;
    var idxArr = geo.index ? geo.index.array : null;
    var pesosOriginais = acoes.map(function (a) { return a ? a.getEffectiveWeight() : 0; });
    var idxOriginal = st.idx;
    var matrizMundo = acharMatrizMundo(malha);

    var out = {
      n_vertices: geo.attributes.position.count,
      max_dim: maxDim,
      indices_u32: idxArr ? b64DeFloat32(Uint32Array.from(idxArr)) : null,
      poses: {}
    };

    POSES.forEach(function (p, i) {
      acoes.forEach(function (a, k) { if (a) a.setEffectiveWeight(k === i ? 1 : 0); });
      if (mixer) mixer.update(0);
      cena.updateMatrixWorld(true);
      malha.skeleton.update();
      out.poses[p.clip] = skinarPoseAtual(malha, matrizMundo);
    });

    // extras (gestos.json:extras) — desenhadas por osso, nao clipes do FBX;
    // entram no mesmo cache pelo mesmo formato, so nao passam por 'acoes'
    var erroExtra = null;
    EXTRAS.forEach(function (ex) {
      if (erroExtra) return;
      try {
        var empurrar = montarPoseCustom(malha, ex.curvas, ex.poseAberta || 'Relaxed');
        out.poses[ex.clip] = skinarPoseAtual(malha, matrizMundo, empurrar);
      } catch (e) {
        erroExtra = { erro: 'extra "' + ex.clip + '": ' + e.message };
      }
    });

    // devolve o mixer ao estado visual de antes de exportar
    acoes.forEach(function (a, k) { if (a) a.setEffectiveWeight(pesosOriginais[k]); });
    if (mixer) mixer.update(0);
    porPose(idxOriginal, false);

    return erroExtra || out;
  }
  window.exportarPoses = exportarPoses;

  /* Monta no esqueleto (bone.quaternion) uma pose descrita por quanto cada
     dedo dobra. curvas: { polegar: t, indicador: t, medio: t, anelar: t,
     minimo: t }, cada t em [0,1] — 0 fica com a rotacao do osso em
     `poseAberta` (padrao "Relaxed"), 1 fica com a de "fist", fracao faz
     slerp de quaternion entre as duas. Dedo omitido fica em `poseAberta`.
     Pra usar uma pose de referencia diferente so num dedo (ex: polegar
     esticado de verdade fica mais parecido com "spock" que com "Relaxed"),
     passe um objeto em vez de numero: { de: 'spock', para: 'fist', t: 0 }.
     Ossos que nao sao de dedo (metacarpos etc.) ficam em `poseAberta` —
     eles nunca mudam entre as 4 poses conhecidas mesmo.

     `empurrar: [x,y,z]` no spec de um dedo desloca ele pro lado depois do
     skinning (nao gira osso nenhum) — devolvido separado, pra
     skinarPoseAtual aplicar. Girar um osso pra separar dedo do vizinho
     (abducao) sai com costura feia: o skinning so foi pintado pra flexao,
     nunca pra esse eixo, em nenhuma das 4 poses reais. Empurrar os
     vertices depois evita isso — e so translacao, sem depender de peso
     pintado pra nada. Devolve os ossos da cadeia como INDICE (a mesma
     ordem de malha.skeleton.bones, que e o que skinIndex guarda). */
  function montarPoseCustom(malha, curvas, poseAberta) {
    var ossos = malha.skeleton.bones;
    var porNome = {};
    var indicePorNome = {};
    ossos.forEach(function (b, i) { porNome[b.name] = b; indicePorNome[b.name] = i; });

    var cacheQuats = {};
    function capturarPose(nomeClipe) {
      if (cacheQuats[nomeClipe]) return cacheQuats[nomeClipe];
      var i = POSES.findIndex(function (p) { return p.clip === nomeClipe; });
      if (i < 0) throw new Error('pose de referencia desconhecida: ' + nomeClipe);
      acoes.forEach(function (a, k) { if (a) a.setEffectiveWeight(k === i ? 1 : 0); });
      if (mixer) mixer.update(0);
      var quats = {};
      ossos.forEach(function (b) { quats[b.name] = b.quaternion.clone(); });
      cacheQuats[nomeClipe] = quats;
      return quats;
    }

    var base = capturarPose(poseAberta);
    ossos.forEach(function (b) { b.quaternion.copy(base[b.name]); });

    var empurrar = [];
    Object.keys(curvas).forEach(function (dedo) {
      var cadeia = DEDOS[dedo];
      if (!cadeia) throw new Error('dedo desconhecido: ' + dedo + ' (use ' + Object.keys(DEDOS).join(', ') + ')');
      var spec = curvas[dedo];
      var de = typeof spec === 'object' ? (spec.de || poseAberta) : poseAberta;
      var para = typeof spec === 'object' ? (spec.para || 'fist') : 'fist';
      var t = typeof spec === 'object' ? spec.t : spec;
      var qDe = capturarPose(de), qPara = capturarPose(para);
      cadeia.forEach(function (nomeOsso) {
        var b = porNome[nomeOsso];
        if (!b) return;
        b.quaternion.copy(qDe[nomeOsso]).slerp(qPara[nomeOsso], t);
      });
      if (typeof spec === 'object' && spec.empurrar) {
        empurrar.push({
          indicesOsso: cadeia.map(function (n) { return indicePorNome[n]; }),
          vetor: new THREE.Vector3(spec.empurrar[0], spec.empurrar[1], spec.empurrar[2])
        });
      }
    });

    // ORDEM IMPORTA: updateMatrixWorld tem que vir antes de skeleton.update().
    // Skeleton.update() so LE bone.matrixWorld (nao recalcula nada) — sem
    // isto aqui antes, ele usa a matrizWorld de QUALQUER pose anterior que
    // tenha rodado por ultimo, nao os quaternions que acabamos de setar.
    // (bug real: sem isto, cada pose custom saia com a forma da anterior —
    // ThumbsUp saia igual a Pointing, Peace igual ao ThumbsUp certo, etc.)
    cena.updateMatrixWorld(true);
    malha.skeleton.update();
    return empurrar;
  }

  /* Calcula {position_f32, normal_f32} de uma pose custom, pronto pra
     entrar no cache do jeito que uma pose de exportarPoses() entra. NAO
     deixa a pose aplicada na tela depois — so calcula e devolve. Ver
     montarPoseCustom() pro formato de `curvas`. */
  function criarPoseCustom(curvas, poseAberta) {
    if (!modelo || !acoes.length || !POSES) return { erro: 'modelo ainda nao carregou' };
    var malha = acharMalha();
    if (!malha) return { erro: 'nenhum SkinnedMesh no modelo' };

    var pesosOriginais = acoes.map(function (a) { return a ? a.getEffectiveWeight() : 0; });
    var idxOriginal = st.idx;
    var matrizMundo = acharMatrizMundo(malha);

    var empurrar = montarPoseCustom(malha, curvas, poseAberta || 'Relaxed');
    var resultado = skinarPoseAtual(malha, matrizMundo, empurrar);

    // devolve o mixer ao estado visual de antes (o loop de render volta a
    // escrever nos ossos a cada quadro, entao isto so evita 1 quadro torto)
    acoes.forEach(function (a, k) { if (a) a.setEffectiveWeight(pesosOriginais[k]); });
    if (mixer) mixer.update(0);
    porPose(idxOriginal, false);

    return resultado;
  }
  window.criarPoseCustom = criarPoseCustom;
  window.DEDOS = DEDOS;

  /* Aplica uma pose custom e MANTEM na tela — pausa o laco de mistura
     (senão o quadro seguinte sobrescreve os ossos com a pose manual/da
     ponte de novo) — pra pré-visualizar um gesto novo antes de decidir os
     numeros finais. window.pararPrevisualizacao() volta ao normal. */
  function previsualizarPoseCustom(curvas, poseAberta) {
    if (!modelo || !acoes.length || !POSES) return { erro: 'modelo ainda nao carregou' };
    var malha = acharMalha();
    if (!malha) return { erro: 'nenhum SkinnedMesh no modelo' };
    montarPoseCustom(malha, curvas, poseAberta || 'Relaxed');
    st.previsualizando = true;
    return { ok: true };
  }
  window.previsualizarPoseCustom = previsualizarPoseCustom;
  window.pararPrevisualizacao = function () { st.previsualizando = false; };

  function exportarPosesEBaixar() {
    var out = exportarPoses();
    if (out.erro) { falhar('Exportar poses: ' + out.erro); return; }
    var blob = new Blob([JSON.stringify(out)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'hand_poses.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

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
    // mira o corpo da mao (foco), nao o pivo do pulso — e um pouco acima
    // disso, pra sobrar menos vazio embaixo no quadro
    orbita.target.copy(foco).addScaledVector(new THREE.Vector3(0, 1, 0), maxDim * 0.07);
    orbita.update();
  }

  // ---------- carrega o mapa de gestos, depois o modelo ----------
  function iniciarComPoses(poses) {
    POSES = poses;
    carregarModelo();
    POSES.forEach(function (p, i) {
      var b = document.createElement('button');
      b.textContent = p.nome;
      b.onclick = function () { porPose(i, true); };
      el.botoes.appendChild(b);
    });
    document.addEventListener('keydown', function (ev) {
      if (ev.key === ' ' || ev.code === 'Space') {
        ev.preventDefault();
        recalibrar();
        return;
      }
      var n = parseInt(ev.key, 10);
      if (n >= 1 && n <= POSES.length) porPose(n - 1, true);
    });
  }

  /* espaco: a orientacao ATUAL do braço passa a valer como EULER_INICIAL
     (dorso pra cima / palma pra baixo / dedos pra longe da camera) — a
     mao pula pra essa pose na hora e, dali em diante, os movimentos do
     braço continuam sendo lidos normalmente, só que relativos a este novo
     zero. Existe porque o Myo nao tem bussola: o "zero" dele é arbitrário
     a cada conexão nova, então sem isto toda sessão começa numa pose
     estranha diferente. */
  function recalibrar() {
    st.offsetCalib = st.eulerBruto.map(function (v, i) { return v - EULER_INICIAL[i]; });
    st.eulerAlvo = EULER_INICIAL.slice();
    st.euler = EULER_INICIAL.slice();
  }

  fetch('gestos.json').then(function (r) {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(function (d) {
    EXTRAS = d.extras || [];
    iniciarComPoses(d.ordem.map(function (item) {
      return { clip: item.clip, nome: item.nome, classe: item.classe, cor: item.cor };
    }));
  }).catch(function (e) {
    falhar('Não consegui carregar <code>gestos.json</code> (' + e + ').');
  });

  function carregarModelo() {
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
    var centro = caixa.getCenter(new THREE.Vector3());
    /* pivo do pulso, nao o centro geometrico da mao. A pulseira do Myo, no
       braço de verdade, fica la no antebraco — bem mais pra tras do que o
       meio da mao. Girar em torno do centro faz a mao "girar em si mesma";
       girar a partir do limite oposto aos dedos e o que reproduz o pulso
       balançando de verdade (ver conversa sobre o salto de angulo/pivo).
       Medido uma vez na bind pose deste FBX (mao aberta): os dedos se
       espalham para +Z local e o osso raiz do esqueleto (o mais proximo da
       pulseira real) fica quase no limite -Z — por isso Z e o eixo
       pulso<->dedos aqui, e caixa.min.z e o limite do lado do pulso. Isto e
       especifico deste modelo; se o FBX for trocado, meça de novo. */
    var pivoPulso = new THREE.Vector3(centro.x, centro.y, caixa.min.z);
    obj.position.sub(pivoPulso);
    foco.copy(centro).sub(pivoPulso).applyEuler(orient.rotation);
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
  }

  // ---------- poses ----------
  function porPose(i, manual) {
    if (!POSES || i < 0 || i >= POSES.length) return;
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
        st.eulerBruto = [+d.euler[0], +d.euler[1], +d.euler[2]];
        st.eulerAlvo = st.eulerBruto.map(function (v, i) { return v - st.offsetCalib[i]; });
      }
      if (d.rms && d.rms.length) st.rms = d.rms.map(Number);
      if (d.name && POSES) {
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
  var bExportar = document.getElementById('b-exportar');
  if (bExportar) bExportar.onclick = exportarPosesEBaixar;

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

    // pesos caminham suavemente: é o blend do Animator (pausado durante
    // previsualizarPoseCustom, senão este laco sobrescreve a pose manual)
    if (!st.previsualizando) {
      var k = 1 - Math.exp(-dt / 0.10);
      for (var i = 0; i < st.peso.length; i++) {
        st.peso[i] += (st.alvo[i] - st.peso[i]) * k;
        if (acoes[i]) acoes[i].setEffectiveWeight(st.peso[i]);
      }
      if (mixer) mixer.update(0);        // pose estática: não avança o tempo
    }

    var ke = 1 - Math.exp(-dt / 0.12);
    var segue = st.seguirImu && (st.fonte === 'myo' || st.fonte === 'sim');
    for (i = 0; i < 3; i++) {
      var alvoEuler = segue ? st.eulerAlvo[i] : EULER_INICIAL[i];
      var deltaEuler = alvoEuler - st.euler[i];
      // caminho mais curto: sem isto, um alvo que embrulhou em ±180° (ou
      // que a ponte manda sem o acumulador do feed.py) faz a mao girar pelo
      // lado errado e "saltar" em vez de suavizar — ver conversa sobre o
      // salto de angulo na IMU
      deltaEuler -= Math.round(deltaEuler / 360) * 360;
      st.euler[i] += deltaEuler * ke;
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
