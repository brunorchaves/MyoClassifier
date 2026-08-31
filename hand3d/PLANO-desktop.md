# Plano: mão 3D em janela nativa, sem navegador

> **Status: planejado, ainda NÃO implementado.**
> Nenhum dos arquivos descritos aqui (`desktop.py`, `modelo.py`, `gestos.json`)
> existe no repositório. O que funciona hoje é a versão web em `hand3d/web/`,
> descrita no [`README.md`](README.md).
>
> Este documento existe para registrar o levantamento e as medições que já foram
> feitas, para que a implementação não recomece do zero. Ao implementar, atualize
> ou remova este arquivo.

## Contexto

A mão riggada roda hoje em `hand3d/web/` — three.js numa página, servida por
`serve.py`, alimentada pela ponte WebSocket. Funciona, e já aposentou o Unity. Mas
para ver a mão é preciso subir servidor, ponte e alimentador, e abrir o navegador.

A ideia é uma versão **desktop**: janela nativa, um comando, sem navegador. Duas
decisões tomadas:

- **Dependências mínimas**: `moderngl` + `moderngl-window` + `pyglet` (~1,45 MB),
  com o skinning na GPU.
- **Lê o Myo direto**, no mesmo processo — sem ponte, sem servidor, sem porta.

A versão web **continua existindo**: o deck de apresentação
([bci_hci_presentation](https://github.com/brunorchaves/bci_hci_presentation))
depende dela. O modo desktop é adição, não substituição.

### O que o terreno impõe

Levantamento feito na máquina de desenvolvimento (Windows, Python 3.12.10):

| fato levantado | consequência |
|---|---|
| Zero biblioteca 3D instalada (só numpy, scipy, matplotlib, pillow) | qualquer caminho exige `pip install` |
| **Nenhum conversor de FBX** — sem Blender, sem fbx2gltf, sem assimp CLI, sem Node | ler o FBX em Python é o problema central, não o render |
| PyPI aberto, wheels cp312/win_amd64 prontas, nenhuma compila | instalar é barato |
| NVIDIA RTX 3050 + Intel UHD (híbrido) | janela OpenGL nativa é viável |
| O FBX usa `PreRotation` e `GeometricTranslation` | parser FBX próprio sairia com o rig torto — **descartado** |

Duas medições feitas na mesma máquina, que definem a arquitetura:

- Skinning completo (105.696 vértices, 29 ossos, 4 influências) em NumPy na CPU:
  **32 ms/quadro → ~31 fps**. Inviável.
- Interpolar poses pré-calculadas: **2 ms/quadro → ~500 fps**.

Logo: o skinning vai para a GPU, e a CPU só compõe 29 matrizes de osso por quadro
(custo desprezível). `mat4[29]` são 116 componentes vec4 — cabe folgado no limite
de qualquer GPU, sem precisar de textura de ossos. Os buffers dão ~7 MB na GPU.

---

## Passo 0 — Spike: como ler o FBX (decide todo o resto)

É a única incógnita real. `pip install assimp-py` (1,80 MB, wheel cp312 pronta) e
~20 linhas que carregam `web/model/hand.fbx` e imprimem o que ele expõe: malhas,
ossos, pesos e — principalmente — **animações**.

Critério de decisão:

| o que o assimp-py expõe | rota |
|---|---|
| ossos + pesos + animações | **A** — tudo em Python, roda de um clone limpo |
| ossos + pesos, sem animações | **A'** — malha e esqueleto do assimp; as 4 poses (TRS local por osso, ~7 KB de JSON) saem de um export único da página web, que já funciona |
| nem ossos | **B** — export completo pelo navegador (posições já skinadas das 4 poses, ~4 MB) e o desktop só interpola vértices; perde a mistura em espaço de osso, ganha simplicidade |

Aposta: **A'** é o resultado mais provável. Em qualquer rota o runtime do app é o
mesmo; muda só de onde vêm os dados.

## Passo 1 — `hand3d/modelo.py`: os dados do modelo

Um módulo que devolve uma estrutura única, independente da rota escolhida:

```
posicoes(N,3) normais(N,3) indices(M,3)
juntas(N,4) pesos(N,4)
hierarquia: pai[29]   bind_inversa[29]
poses: {"Relaxed": TRS_local[29], "fist": ..., "spock": ..., "Pointing": ...}
```

- cacheia em `web/model/hand.cache.npz` (ignorado pelo git) para o arranque ser
  instantâneo
- **valida na carga**: soma dos pesos ≈ 1, índices de junta dentro do intervalo, as
  4 poses presentes. Falhar cedo com mensagem clara é melhor que rig torto.

## Passo 2 — `hand3d/desktop.py`: a janela

`moderngl_window.WindowConfig` (gl_version 3.3) dá janela, laço, eventos de mouse e
teclado sem código de plumbing.

- VBO/IBO montados uma vez; uniform `mat4 ossos[29]` atualizado por quadro
- **vertex shader**: linear blend skinning com 4 influências; normal pela mesma
  combinação
- **fragment shader**: Lambert + luz de borda + fresnel suave, na paleta da versão
  web (fundo escuro, mão clara)
- controles: mouse orbita, scroll dá zoom, `1`–`4` poses, `g` gira, `w` wireframe,
  `i` alterna seguir a IMU, `f` reenquadra
- **HUD sem dependência nova**: texto renderizado com PIL (já instalado) numa
  textura, desenhado num quad — pose, fonte do dado, Hz, euler, fps
- imprimir `ctx.info['GL_RENDERER']` no arranque: numa máquina híbrida é assim que
  se descobre que caiu na Intel em vez da NVIDIA

## Passo 3 — Fidelidade ao three.js

Os números têm de ser os mesmos de `web/hand.js`, senão a mão "sente" diferente:

- pesos das poses: `k = 1 - exp(-dt/0.10)`
- por osso: **nlerp dos quaternions** das 4 poses ponderado pelos pesos + lerp de
  translação/escala, depois compõe a hierarquia e multiplica pela bind inversa.
  Misturar em espaço de osso (e não de matriz nem de vértice) é o que reproduz o
  `AnimationMixer`.
- orientação: euler suavizado com `ke = 1 - exp(-dt/0.12)` e aplicado na **ordem
  XYZ do three.js** — `R = Rx(roll)·Ry(yaw)·Rz(pitch)` — com a correção fixa de
  eixo `ORIENT = Rx(-90)·Rz(90)` num nível acima
- giro automático: `+0.55 rad/s` no eixo Y
- 2,5 s sem dado volta para o modo manual

## Passo 4 — O Myo no mesmo processo

Reaproveitar de `feed.py` **sem reescrever**: `Classificador`,
`euler_de_quaternion`, `descobrir_mac`, `ler_mac_salvo`, `salvar_mac`, e a
suavização por voto de maioria (janela 25).

- thread do Myo rodando `m.run()`, escrevendo num dict protegido por lock; o laço
  de render só lê
- o mesmo cão de guarda do `feed.py` (o `connect()` do pyomyo espera sem timeout e
  prende para sempre se o bracelete dormir) — mas aqui o `desktop.py` deve **tentar
  reconectar sozinho** em vez de morrer, porque a janela já está aberta
- `--sim` inventa gesto e orientação, para ensaiar sem bracelete
- `--foto saida.png` renderiza um quadro e sai — permite conferir o resultado sem
  abrir janela

## Passo 5 — Mapa de gestos num lugar só

Hoje o mapa classe→pose vive em `bridge.py` (`GESTOS`) e em `web/hand.js`
(`POSES`). Com o desktop seriam **três cópias** — e o aviso da classe 4 sem treino
já mostra que esse mapa muda. Criar `hand3d/gestos.json` como fonte única:
`bridge.py` e `desktop.py` leem, `hand.js` busca por fetch.

## Passo 6 — Documentação

Seção "modo desktop" no [`README.md`](README.md) (quando usar cada modo, controles,
o que instalar) e `hand3d/requirements-desktop.txt`.

---

## Arquivos

| arquivo | o que faz |
|---|---|
| `hand3d/desktop.py` | **novo** — janela, shaders, laço, Myo em thread |
| `hand3d/modelo.py` | **novo** — extrai/carrega malha, esqueleto e poses; cache npz |
| `hand3d/gestos.json` | **novo** — mapa classe→clipe, fonte única |
| `hand3d/requirements-desktop.txt` | **novo** — moderngl, moderngl-window, pyglet (+assimp-py nas rotas A/A') |
| `hand3d/README.md` | seção do modo desktop |
| `hand3d/bridge.py` | passa a ler `gestos.json` em vez do dict embutido |
| `hand3d/web/hand.js` | lê `gestos.json`; e, nas rotas A'/B, ganha o botão de exportar |
| `.gitignore` (raiz) | ignorar `hand3d/web/model/hand.cache.npz` |

## Verificação

1. **Spike**: o script imprime malhas, ossos, pesos e animações que o assimp expõe
   → define a rota antes de escrever o resto.
2. **Fidelidade por imagem**: `python desktop.py --sim --foto p1.png` para cada uma
   das 4 poses, e comparação lado a lado com a página web no mesmo ângulo. É o
   teste que pega rig torto, mão espelhada e normal invertida.
3. **Janela**: `python desktop.py --sim` — órbita, zoom, teclas `1`–`4`, `g`, `w`,
   `i`, `f`; fps no HUD (esperado 60, travado pelo vsync).
4. **Com o bracelete**: `python desktop.py` — gesto trocando e a mão girando com o
   braço; conferir os ~48–50 Hz já medidos na cadeia web.
5. **Regressão da web**: `python run.py --sim` e `python run.py` continuam
   funcionando, inclusive depois da mudança do `gestos.json`.

## Riscos

| risco | como se manifesta | diagnóstico |
|---|---|---|
| assimp-py sem animações | spike não lista clipes | cai para a rota A' (export do navegador) |
| ordem de rotação trocada | mão espelhada, ou girando no eixo errado | comparar foto com a web na mesma pose |
| `PreRotation` mal composta | dedos torcidos **já na pose de bind** | renderizar sem animação: se a bind sai torta, é a composição |
| normais invertidas | mão escura, iluminação ao contrário | desligar `cull_face` e ver se melhora |
| GPU híbrida cai na Intel | fps baixo sem motivo | o `GL_RENDERER` impresso no arranque diz |
| FBO offscreen sem sessão gráfica | `--foto` falha no shell | usar a janela real e capturar um quadro |

## Fora de escopo

Não mexer no deck do outro repositório, não remover a versão web, não tocar em
`src/myoControlsHand.py`. O modo desktop é adição.
