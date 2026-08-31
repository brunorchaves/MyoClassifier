# hand3d — a mão 3D no navegador, sem Unity

O mesmo `handAnimations.fbx` que o projeto Unity usava, renderizado com
[three.js](https://threejs.org) e animado pela classificação do Myo. Roda numa
aba do navegador: sem abrir editor, sem dar Play, sem deixar janela em foco.

```bash
cd hand3d
python run.py
```

Um comando sobe a ponte, lê o bracelete e abre a página. `Ctrl+C` derruba tudo.

---

## O que substituiu o quê

| Unity (`3DORientaion_test/`) | agora |
|---|---|
| `myListener.cs` — servidor TCP 25001 → `transform.rotation` | `bridge.py` → WebSocket → `pivot.rotation` no `web/hand.js` |
| `handController.cs` — `Animator` com blend tree Grip × Trigger | `AnimationMixer` com um peso por clipe |
| gesto via **tecla emulada** (`Input.GetKeyDown`, lib `keyboard`) | campo `gesture` no JSON |
| editor + build + janela em foco | uma aba do navegador |

O detalhe que fez isso ser simples: o FBX traz as quatro poses (`Relaxed`,
`fist`, `spock`, `Pointing`) como **clipes de um quadro**. Clipe de um quadro é
exatamente o que um blend tree mistura — então a transição suave entre gestos
sai de interpolar peso, não de animação gravada.

O `keyboard.press()` era o ponto mais frágil da montagem antiga: só funcionava
com a janela do Unity em foco, e em alguns sistemas pede privilégio de
administrador. Agora o gesto viaja no mesmo pacote da orientação.

---

## As três peças

```
hand3d/
  run.py        sobe tudo com um comando, com nova tentativa se o Myo dorme
  bridge.py     WebSocket :8765 + TCP :25001 — o lugar do Unity
  feed.py       lê o Myo (pyomyo), classifica (1-NN) e alimenta a ponte
  serve.py      servidor estático da página
  web/
    index.html  a interface
    hand.js     three.js: carrega o FBX, mistura as poses, ouve a ponte
    model/hand.fbx     cópia de Assets/handAnimations.fbx
    vendor/            three.js r147 + FBXLoader (local, funciona offline)
```

### `bridge.py` — fala o protocolo do Unity

Escuta na **porta 25001 com o mesmo protocolo do `myListener.cs`**, inclusive o
eco que o `sock.recv()` do `src/myoControlsHand.py` espera. Ou seja: **o seu
script antigo roda sem alteração** e a mão já gira.

Para o **gesto** também chegar, é uma linha nele:

```python
data_str = f"{roll},{-yaw},{pitch}"                 # antes
data_str = f"{roll},{-yaw},{pitch},{int(pose)}"     # depois
```

O WebSocket escuta na 8765 e é ali que a página se conecta.

### `feed.py` — o caminho recomendado

Substitui o `src/myoControlsHand.py` para este uso, e resolve dois problemas
dele:

- **Dependências.** O script antigo importa `pygame`, `keyboard`, `joblib` e
  `scikit-learn`. O `feed.py` usa só o `pyomyo` (que já está em `src/`) e
  `numpy`.
- **O scan que trava.** O `connect()` do `pyomyo` espera o evento de conexão
  **sem timeout**, e o Myo dorme quando fica parado — o processo prende para
  sempre. O `feed.py` conecta **pelo MAC**, que ele descobre uma vez e guarda em
  `myo_mac.txt`, e tem um cão de guarda que desiste em 10 s com mensagem clara.
  O `run.py` então tenta de novo: quando você pega o bracelete e mexe, a
  tentativa seguinte pega.

A classificação é fiel à sua: o mesmo **1-NN** sobre os mesmos
`src/data/vals*.dat`, e a mesma suavização por voto de maioria numa janela de
25 amostras.

---

## Modos

```bash
python run.py                 # ponte + Myo + página
python run.py --sim           # sem bracelete: gesto e orientação inventados
python run.py --sem-myo       # ponte + página, e você alimenta como quiser
python run.py --porta 9000
```

Ou as peças na mão, em três terminais:

```bash
python bridge.py
python feed.py
python serve.py
```

Sem ponte nenhuma, a página ainda abre e os **botões da lateral** (ou as teclas
`1`–`4`) comandam a pose. Útil para conferir o modelo, ou para mostrar as poses
quando o bracelete não colabora.

---

## A página

- **arraste** para orbitar, **scroll** para zoom, **`1`–`4`** trocam a pose
- **seguir a IMU** — desliga a rotação vinda do bracelete, se você quiser só as
  poses
- **esqueleto** e **wireframe** — para conferir o rig
- **EMG por canal** — as 8 barras, alimentadas pela ponte
- o rótulo no alto esquerdo diz de onde vem o dado: `Myo ao vivo`,
  `ponte em simulação`, `ponte no ar, sem dado do bracelete` ou `sem ponte`.
  Ele **não** diz "ao vivo" quando não está — isso importa numa demonstração.

---

## ⚠️ Falta treino para um dos gestos

Os `src/data/vals*.dat` têm 976 amostras, mas só nas classes 1, 2 e 3:

| classe | amostras | clipe |
|---|---|---|
| 1 | 259 | `Relaxed` — mão aberta |
| 2 | 315 | `fist` — punho |
| 3 | 402 | `spock` — dedos separados |
| 4 | **0** | `Pointing` — **nunca vai sair do classificador** |

O botão "apontando" funciona (é manual), mas o Myo nunca vai classificar essa
pose. Grave a classe 4 com o `src/emgGestureTrainer.py`, ou ajuste o mapa
`GESTOS` no `bridge.py` (e `POSES` no `web/hand.js`) para trabalhar com três
gestos. O `feed.py` avisa isso ao subir.

---

## Dá pra apagar o projeto Unity?

Do ponto de vista desta pasta, sim: o FBX está copiado em `web/model/hand.fbx`
e nada mais aqui olha para `3DORientaion_test/`. Aquela pasta nunca foi
commitada no repositório (o `.gitignore` agora a ignora explicitamente), e a
`Library/` do Unity ocupa bastante espaço.

Antes de apagar, guarde o `.blend` original em algum lugar — é dele que sai o
rig, e o FBX é só a exportação.

---

## Notas técnicas

- **three.js r147** é a última versão com os scripts clássicos
  (`examples/js/`), que carregam com `<script src>` sem módulos ES. Isso mantém
  a página funcionando também ao ser aberta de um servidor simples, sem
  bundler. Está vendorizado em `web/vendor/` para funcionar offline.
- O FBX é **7.4 binário**, que o `FBXLoader` abre direto: 29 ossos, 1 malha
  skinada, 105.696 vértices, 4 clipes. Carrega em ~0,8 s.
- O modelo vem deitado (Blender é Z-up), então há uma correção de eixo fixa
  (`rx: -90, rz: 90`) num grupo entre o pivot e a malha. Se você trocar o
  modelo e ele aparecer torto, é esse `ORIENT` no `hand.js` que se ajusta.
- O handshake do WebSocket no `bridge.py` é escrito na mão, sem dependências.
  Se você mexer nele, confira contra o vetor de teste do RFC 6455: a chave
  `dGhlIHNhbXBsZSBub25jZQ==` tem de dar
  `s3pPLMBiTxaQ9kYGzzhZRbK+xOo=`. Uma letra fora do lugar no GUID mágico faz o
  navegador recusar com `code 1006` e nenhuma mensagem útil.
