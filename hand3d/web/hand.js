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
  /* Não existe mais "EULER_INICIAL". A pose de repouso virou Q_REF (abaixo)
     e a orientação vem do quaternion da IMU. Os valores antigos ([0,-90,0],
     [180,-90,-CAM.az], [180,90,-CAM.az]…) foram todos achados por tentativa
     contra uma montagem de Euler que estava errada de origem — cada um
     acertava um eixo e desalinhava outro. Histórico completo no comentário
     de Q_IMU_CENA e em hand3d/README.md. */

  /* ---------- orientação: quaternion, não Euler ----------
     A ORIGEM DE TODO O PROBLEMA era converter a IMU pra Euler e remontar
     na cena: o feed.py extrai roll/pitch/yaw na convenção de aeronáutica
     (R = Rz*Ry*Rx, mundo Z-up) e o three.js remontava com outra ordem num
     mundo Y-up. Cada tentativa de acertar um eixo desalinhava outro
     ("palma certa mas dedos pra trás", "levanto o braço e vai na
     diagonal"). Agora o quat CRU da IMU vem no pacote (feed.py) e é usado
     direto, sem nunca passar por Euler.

     Q_IMU_CENA: a rotação constante IMU -> cena, medida (não chutada) por
     resolver_calibracao.py a partir da GRAVIDADE — o acelerômetro parado
     dá a vertical do mundo do Myo, que saiu +z com estabilidade de ±0,7°,
     conferida contra o eixo do movimento de varredura lateral (1,6° de
     diferença). O resultado é essencialmente "Z-up -> Y-up".

     Só a vertical precisa casar: com ela alinhada, subir o braço sobe a
     mão e varrer pro lado varre pro lado. O que sobra é o heading, que um
     IMU sem bússola não sabe — vive em Q_MONTAGEM, junto com "como o
     bracelete está vestido no antebraço". Esse fator entra pela DIREITA
     (multiply), e por isso não distorce movimento nenhum: em rotações
     relativas ele cancela. É o espaço/auto-calibração que o define. */
  // construídos depois da checagem de que o three.js carregou (logo abaixo):
  // usar THREE aqui em cima trocaria a mensagem de erro amigável por um
  // ReferenceError cru.
  var Q_IMU_CENA = [-0.706714, 0.000000, 0.001382, 0.707498];   // (x,y,z,w)
  var Q_REF = null;   // pose de repouso = pivot identidade (pose do ORIENT)

  var st = {
    peso: [1, 0, 0, 0], alvo: [1, 0, 0, 0], idx: 0,
    euler: [0, 0, 0],                    // só leitura do painel
    qImu: null, qAlvo: null, qAtual: null, qMontagem: null,
    temQuat: false, calibrouAuto: false, giroExtra: 0,
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

  // agora dá pra construir os quaternions (ver comentario da orientacao acima)
  Q_IMU_CENA = new THREE.Quaternion(Q_IMU_CENA[0], Q_IMU_CENA[1], Q_IMU_CENA[2], Q_IMU_CENA[3]);
  Q_REF = new THREE.Quaternion();
  st.qImu = new THREE.Quaternion();
  st.qAlvo = new THREE.Quaternion();
  st.qAtual = new THREE.Quaternion();
  st.qMontagem = new THREE.Quaternion();

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
  /* A orientação é escrita em pivot.QUATERNION (ver o bloco Q_IMU_CENA no
     topo), nunca mais em pivot.rotation — Euler foi a origem do problema.
     A ordem fica em 'ZYX' só porque é nela que a leitura de Euler mostrada
     no painel bate com a convenção de roll/pitch/yaw do feed.py. */
  pivot.rotation.order = 'ZYX';
  var orient = new THREE.Group(); pivot.add(orient);
  var d2r = Math.PI / 180;
  orient.rotation.set(ORIENT.rx * d2r, ORIENT.ry * d2r, ORIENT.rz * d2r);

  var modelo = null, mixer = null, acoes = [], esqueleto = null, maxDim = 1;
  // de onde saiu o pivo da rotacao (osso do pulso ou fallback), e o marcador
  // que deixa isso VISIVEL — ver o botao "esqueleto"
  var nomePivo = '—', marcaPivo = null;

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

  /* osso raiz do esqueleto: o unico cujo pai nao e osso. No braco real e o
     ponto mais proximo da pulseira, entao e o pivo natural da rotacao que
     vem da IMU — a mao balança a partir do pulso, nao do meio da palma. */
  function acharOssoRaiz(raizObj) {
    var achado = null;
    raizObj.traverse(function (n) {
      if (achado || !n.isBone) return;
      if (!n.parent || !n.parent.isBone) achado = n;
    });
    return achado;
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
    var original = pivot.quaternion.clone();
    pivot.quaternion.identity();
    cena.updateMatrixWorld(true);
    var m = malha.matrixWorld.clone();
    pivot.quaternion.copy(original);
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

  /* Mira o corpo da mao ONDE ELE ESTA AGORA, nao onde estava ao carregar:
     a mao gira em torno do pulso (que fica na origem do pivot), entao com
     uma rotacao grande o corpo dela sai do quadro e o botao "camera" nao
     resolvia — mirava um ponto congelado do load. Recalcular a caixa em
     coordenadas de mundo faz o botao sempre reachar a mao, qualquer que
     seja a pose/rotacao (foi assim que a mao "desapareceu" ao testar uma
     pose inicial nova). */
  function enquadrar() {
    if (!modelo) return;
    cena.updateMatrixWorld(true);
    var centroMundo = new THREE.Box3().setFromObject(modelo).getCenter(new THREE.Vector3());
    // um pouco acima do centro, pra sobrar menos vazio embaixo no quadro
    centroMundo.y += maxDim * 0.07;

    var dist = maxDim / (2 * Math.tan(cam.fov * Math.PI / 360)) * CAM.zoom;
    var a = CAM.az * d2r, e = CAM.el * d2r;
    cam.position.set(centroMundo.x + dist * Math.cos(e) * Math.sin(a),
                     centroMundo.y + dist * Math.sin(e),
                     centroMundo.z + dist * Math.cos(e) * Math.cos(a));
    orbita.target.copy(centroMundo);
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

  /* espaco: a orientacao ATUAL do braço passa a valer como pose de repouso
     (Q_REF) — a mao pula pra ela na hora e, dali em diante, os movimentos
     do braço continuam sendo lidos normalmente, relativos a este novo
     zero. Existe porque o Myo nao tem bussola: o heading dele é arbitrário
     a cada conexão, então sem isto toda sessão começaria numa pose
     estranha diferente. A 1a leitura de cada conexão já faz isto sozinha
     (ver calibrouAuto); a tecla serve pra refazer quando quiser. */
  /* orientação do antebraço nos eixos da CENA (só a constante medida) */
  function orientacaoDaCena(qImu) {
    return new THREE.Quaternion().copy(Q_IMU_CENA).multiply(qImu);
  }

  /* espaço (e a auto-calibração da 1a leitura): a orientação ATUAL do braço
     passa a valer como pose de repouso. Resolve
        Q_MONTAGEM = (A * q_imu)^-1 * Q_REF
     que é exatamente o fator da direita — entra depois da rotação da IMU,
     no referencial do corpo, então muda a pose de partida sem mexer em
     como o movimento é mapeado. */
  function recalibrar() {
    if (!st.temQuat) {
      st.qAlvo.copy(Q_REF);
      st.qAtual.copy(Q_REF);
      return;
    }
    st.qMontagem.copy(orientacaoDaCena(st.qImu)).invert().multiply(Q_REF);
    st.qAlvo.copy(Q_REF);
    st.qAtual.copy(Q_REF);
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
    maxDim = Math.max(tam.x, tam.y, tam.z) || 1;

    /* ---------- PIVO DA ROTACAO ----------
       Onde estava o bug: o codigo media a caixa com Box3.setFromObject(),
       que devolve MUNDO (ja com o ORIENT aplicado), e subtraia isso de
       obj.position, que vive no espaco do ORIENT. O proprio comentario
       antigo dizia "+Z local" — a intencao era local, a implementacao usou
       mundo. Com ORIENT = (-90, 0, 90) os eixos estao permutados, entao
       "caixa.min.z" nao era o lado do pulso: a mao girava em torno de um
       ponto qualquer e saia voando pela tela.

       Por que 'modelo' e o padrao: no Unity, que funcionava, a rotacao ia em
       transform.rotation do proprio objeto — ou seja, o pivo era a ORIGEM DO
       FBX, sem deslocamento nenhum (ver
       3DORientaion_test/Assets/myListener.cs). O deslocamento foi invencao
       desta port. Voltar ao comportamento do Unity e mais fiel do que
       inventar um pivo melhor.

       'pulso' fica disponivel caso a origem do FBX NAO esteja no pulso: usa
       o osso raiz do esqueleto (o mais proximo da pulseira no braco real) e
       converte pro espaco do ORIENT com worldToLocal, que e o espaco de
       obj.position. Trocar uma palavra aqui e ver: o botao "esqueleto"
       desenha eixos na origem do pivot. */
    /* Queremos a BORDA DO PULSO, nao o centro da mao (a origem do FBX cai
       perto do centro, e girar ali faz a mao "girar em si mesma" em vez de
       balançar no pulso como o braço de verdade).

       O eixo da mao NAO e chutado: sai do proprio esqueleto — do osso raiz
       (a junta do pulso) para a media das PONTAS dos dedos. Com esse eixo em
       maos, a borda do pulso e o ponto da malha mais "atras" ao longo dele.
       Tudo e calculado em MUNDO e convertido uma unica vez pro espaco do
       ORIENT com worldToLocal — que e o espaco de obj.position. Era
       exatamente essa conversao que faltava antes (media em mundo, aplicada
       em local, com os eixos permutados pelo ORIENT). */
    obj.updateWorldMatrix(true, true);
    var pivoW = null;
    var osso = acharOssoRaiz(obj);
    var malhaPivo = acharMalha();
    if (osso && malhaPivo) {
      var pulsoW = new THREE.Vector3().setFromMatrixPosition(osso.matrixWorld);
      var ossosPorNome = {};
      malhaPivo.skeleton.bones.forEach(function (b) { ossosPorNome[b.name] = b; });
      var pontas = new THREE.Vector3();
      var nPontas = 0;
      Object.keys(DEDOS).forEach(function (dedo) {
        var cadeia = DEDOS[dedo];
        var b = ossosPorNome[cadeia[cadeia.length - 1]];
        if (!b) return;
        pontas.add(new THREE.Vector3().setFromMatrixPosition(b.matrixWorld));
        nPontas++;
      });
      if (nPontas) {
        pontas.divideScalar(nPontas);
        var eixoMao = pontas.clone().sub(pulsoW);
        if (eixoMao.length() > 1e-6) {
          eixoMao.normalize();
          /* Projeta os VERTICES no eixo da mao (nao os cantos da caixa: a
             caixa e alinhada aos eixos do mundo e, com o eixo da mao na
             diagonal, o canto passaria longe da borda real). Roda uma vez
             no carregamento, entao o custo nao importa. */
          var pos = malhaPivo.geometry.attributes.position;
          var v = new THREE.Vector3();
          var menor = Infinity;
          for (var iv = 0; iv < pos.count; iv++) {
            v.fromBufferAttribute(pos, iv).applyMatrix4(malhaPivo.matrixWorld);
            var t = v.sub(pulsoW).dot(eixoMao);
            if (t < menor) menor = t;
          }
          pivoW = pulsoW.clone().addScaledVector(eixoMao, menor);
          nomePivo = 'borda do pulso (eixo do esqueleto: "' + osso.name +
                     '" -> ' + nPontas + ' pontas de dedo)';
        }
      }
      if (!pivoW) {
        pivoW = pulsoW;                     // pelo menos a junta do pulso
        nomePivo = 'junta do pulso, osso "' + osso.name + '"';
      }
    }
    if (pivoW) {
      orient.worldToLocal(pivoW);
      obj.position.sub(pivoW);
    } else {
      nomePivo = 'origem do FBX (nao achei esqueleto pra medir o pulso)';
    }
    console.log('[hand3d] pivo da rotacao:', nomePivo);
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
    ws.onclose = function () {
      // o "zero" do Myo é arbitrário a cada conexão: ao reconectar, deixa a
      // 1a leitura nova recalibrar sozinha (ver recalibrar()).
      st.calibrouAuto = false;
      setTimeout(conectar, 4000);
    };
    ws.onmessage = function (ev) {
      var d;
      try { d = JSON.parse(ev.data); } catch (e) { return; }
      st.ultimoDado = performance.now();
      // a ponte pode estar no ar SEM bracelete: nesse caso src vem "—".
      // Tratar tudo que nao e 'sim' como 'myo' faria a tela mentir.
      st.fonte = d.src === 'myo' ? 'myo' : d.src === 'sim' ? 'sim' : 'ponte';
      if (d.fs) st.fs = +d.fs;
      if (d.gesture !== undefined) st.classe = +d.gesture;
      // quat CRU da IMU: (w,x,y,z) no pacote, (x,y,z,w) no three.js
      if (d.quat && d.quat.length === 4) {
        st.qImu.set(+d.quat[1], +d.quat[2], +d.quat[3], +d.quat[0]);
        st.temQuat = true;
        // 1a leitura da sessão já zera: o "zero" do Myo é arbitrário a cada
        // conexão, então sem isto a mão começaria numa pose qualquer
        if (!st.calibrouAuto) {
          st.calibrouAuto = true;
          recalibrar();
        }
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
    if (esqueleto) {
      cena.remove(esqueleto); esqueleto = null;
      if (marcaPivo) { pivot.remove(marcaPivo); marcaPivo = null; }
      this.classList.remove('on');
      return;
    }
    esqueleto = new THREE.SkeletonHelper(modelo);
    cena.add(esqueleto);
    /* eixos no PIVO da rotacao: e o jeito de VER se o centro de giro esta no
       pulso ou num ponto qualquer (foi exatamente esse o bug do offset
       medido em mundo e aplicado em local). Fica preso ao pivot, entao gira
       junto com a mao. */
    marcaPivo = new THREE.AxesHelper(maxDim * 0.35);
    pivot.add(marcaPivo);
    console.log('[hand3d] eixos na origem do pivot (X vermelho, Y verde, ' +
                'Z azul). Pivo:', nomePivo);
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
    if (segue && st.temQuat) {
      st.qAlvo.copy(orientacaoDaCena(st.qImu)).multiply(st.qMontagem);
    } else {
      st.qAlvo.copy(Q_REF);
    }
    // slerp: caminho mais curto de graça. O truque de "±180" que existia
    // aqui era remendo do Euler; quaternion não embrulha.
    st.qAtual.slerp(st.qAlvo, ke);
    pivot.quaternion.copy(st.qAtual);
    if (st.girar) {
      // gira em torno da VERTICAL da cena; premultiply = rotação de mundo
      st.giroExtra += dt * 0.55;
      pivot.quaternion.premultiply(
        new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), st.giroExtra));
    }
    // leitura do painel: Euler do que foi realmente aplicado
    var eLido = new THREE.Euler().setFromQuaternion(st.qAtual, 'ZYX');
    st.euler = [eLido.x / d2r, eLido.y / d2r, eLido.z / d2r];

    if (agora - ultimoPainel > 120) { ultimoPainel = agora; pintarPainel(); }

    orbita.update();
    ren.render(cena, cam);
  }

  redimensionar();
  conectar();
  quadro();
})();
