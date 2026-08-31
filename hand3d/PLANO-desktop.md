# Plan: 3D hand in a native window, no browser

> **Status: implemented (route B).** `desktop.py`, `modelo.py` and
> `gestos.json` exist and work — see the "Desktop mode" section in the
> [`README.md`](README.md) for day-to-day use. The web version in
> `hand3d/web/` keeps working with no change in behavior.
>
> This document remains as a record of the research, the measurements, and
> the decisions (mainly route B, decided in Step 0) — useful if the FBX ever
> gets swapped out or the cache needs to be regenerated.
>
> **Update after this was written**: the app grew past the FBX's original 4
> poses — `gestos.json`/`desktop.py` now handle any number of poses, and 3
> more (`ThumbsUp`, `Peace`, `RockOn`) were added by posing the skeleton
> directly in `web/hand.js`, no 3D authoring tool needed. See the README's
> ["Creating new poses"](README.md#creating-new-poses) section.

## Context

The rigged hand runs today in `hand3d/web/` — three.js on a page, served by
`serve.py`, fed by the WebSocket bridge. It works, and it already retired
Unity. But to see the hand you need to bring up a server, a bridge, and a
feeder, and open a browser.

The idea is a **desktop** version: a native window, one command, no browser.
Two decisions already made:

- **Minimal dependencies**: `moderngl` + `moderngl-window` + `pyglet`
  (~1.45 MB), with skinning on the GPU.
- **Reads the Myo directly**, in the same process — no bridge, no server, no
  port.

The web version **keeps existing**: the presentation deck
([bci_hci_presentation](https://github.com/brunorchaves/bci_hci_presentation))
depends on it. Desktop mode is an addition, not a replacement.

### What the terrain imposes

Survey done on the development machine (Windows, Python 3.12.10):

| fact found | consequence |
|---|---|
| Zero 3D libraries installed (only numpy, scipy, matplotlib, pillow) | any path requires a `pip install` |
| **No FBX converter at all** — no Blender, no fbx2gltf, no assimp CLI, no Node | reading the FBX in Python is the central problem, not the rendering |
| PyPI open, cp312/win_amd64 wheels ready, none need to compile | installing is cheap |
| NVIDIA RTX 3050 + Intel UHD (hybrid) | a native OpenGL window is viable |
| The FBX uses `PreRotation` and `GeometricTranslation` | a homegrown FBX parser would come out with a crooked rig — **ruled out** |

Two measurements taken on the same machine, which shape the architecture:

- Full skinning (105,696 vertices, 29 bones, 4 influences) in NumPy on the
  CPU: **32 ms/frame → ~31 fps**. Not viable.
- Interpolating precomputed poses: **2 ms/frame → ~500 fps**.

So: skinning goes to the GPU, and the CPU only composes 29 bone matrices per
frame (negligible cost). `mat4[29]` is 116 vec4 components — comfortably fits
within the limits of any GPU, with no need for a bone texture. The buffers
come out to ~7 MB on the GPU.

---

## Step 0 — Spike: how to read the FBX (decides everything else) — ✅ done, route B

`pip install assimp-py` (cp312 wheel ready, fine) and a ~40-line script
loading `web/model/hand.fbx` with `Process_Triangulate`. The actual result,
different from the bet:

```
Scene:  materials, meshes, num_materials, num_meshes, root_node   (NO animations)
Mesh:   bitangents, colors, indices, material_index, name, normals,
        num_faces, num_indices, num_uv_components, num_vertices,
        tangents, texcoords, vertices                              (NO bones)
```

The Python binding of `assimp-py` 1.1.0 is a minimal wrapper: it only exposes
static mesh data (position/normal/UV/index). **No bones, no weights, no
animations** — even though `root_node` shows the `Armature/Bone.*` hierarchy
just fine (the skeleton exists in the file; the binding simply doesn't map it
into Python). There's also no viable `pyassimp` without compiling (it needs
the native libassimp library separately, out of scope for "just pip
install").

Decision: **route B**. Practical consequence, straight from the table above:

- **No bone-space blending.** What's left for desktop mode isn't
  bone+weight+hierarchy — it's position and normal **already skinned**, per
  vertex, once per pose. Steps 1–3 below were rewritten around that: no
  `mat4[29]`, no per-bone quaternion nlerp; the shader blends 4
  position/normal buffers by weight (it's literally morph-target blending).
- **Data source = three.js, not assimp.** `assimp-py` is not in
  `requirements-desktop.txt` — it isn't used at runtime at all.
- **Export implemented**: `web/hand.js` gained `exportarPoses()` (the
  "export poses" button on the page), which replicates the exact formula from
  three.js's `skinning_vertex.glsl.js`/`skinnormal_vertex.glsl.js` —
  `skinned = bindMatrixInverse · Σ weight_i · boneMatrix_i · bindMatrix · vertex`
  — for each of the 4 poses, and returns position+normal as base64.
- **Cache generated**: `web/model/hand.cache.npz` (105,696 vertices, no
  index — the FBXLoader mesh already comes out as loose triangles, 3
  vertices per triangle; `nrm_*` normalized). Generated once with
  `exportar_poses_headless.py` (playwright + headless Chromium, a
  **dev-only** dependency, not the app's) — the same computation a manual
  click on the button would do.

## Step 1 — `hand3d/modelo.py`: the model data — ✅ implemented

Route B: no joints/weights/hierarchy in Python at all — that was already
consumed on the three.js side at export time. What's left:

```
n_vertices: int                          # 105,696, a multiple of 3 (loose triangles)
posicoes: {"Relaxed": (N,3) f32, "fist": ..., "spock": ..., "Pointing": ...}
normais:  {"Relaxed": (N,3) f32, ...}
ordem: ["Relaxed", "fist", "spock", "Pointing"]   # same order as gestos.json
```

- reads `web/model/hand.cache.npz` (git-ignored); if it doesn't exist, a
  clear error telling you to click "export poses" on the page or run
  `exportar_poses_headless.py`
- **validates on load**: all 4 poses present, all with the same `(N,3)`
  shape, N a multiple of 3, no NaN. Failing fast with a clear message beats a
  crooked rig.

## Step 2 — `hand3d/desktop.py`: the window — ✅ implemented

`moderngl_window.WindowConfig` (gl_version 3.3, pyglet backend) gives you a
window, the loop, and mouse/keyboard events with no plumbing code.

- 4 position VBOs + 4 normal VBOs (one pair per pose), no index — draws
  `TRIANGLES` directly (the FBXLoader mesh already comes out as loose
  triangles)
- **vertex shader**: `pos = Σ weight_i · pos_i`, `normal = Σ weight_i ·
  normal_i` (a blend of 4 positions per vertex — literally a morph target,
  not bone skinning; see Step 0/route B)
- **fragment shader**: Lambert + rim light + soft fresnel, in the web
  version's palette (`HemisphereLight` 0xbcd4ff/0x141a26, light 0xffffff,
  rim 0x22d3ee, fill 0xf472b6) — a visual approximation, not pixel-perfect
  PBR
- `CULL_FACE` **deliberately off**: no guarantee about winding direction
  after the skinning exported from three.js (see Risks)
- controls: mouse orbits, scroll zooms, `1`–`4` for poses, `g` spins, `w`
  wireframe, `i` toggles following the IMU, `f` re-frames the camera
- **HUD with no new dependency**: text rendered with PIL onto an RGBA
  texture, drawn on a quad in NDC — pose, data source, euler, Hz, fps
- `ctx.info['GL_RENDERER']` printed on startup — on this machine (hybrid
  Intel UHD + NVIDIA RTX 3050) the window opens on the **Intel** GPU by
  default; force the NVIDIA one in Windows' graphics settings if you want
  (see README)

## Step 3 — Fidelity to three.js — ✅ implemented, with one simplification

Route B changes this step: there's no more "bone space" at all on the
desktop side — the `AnimationMixer`'s bone-space blend already happened (or
didn't; see below) on the three.js side, at export time.

- pose weights: `k = 1 - exp(-dt/0.10)` — same as the web version, now
  weighting vertex positions instead of `AnimationAction` weights
- orientation: euler smoothed with `ke = 1 - exp(-dt/0.12)`, applied as
  `R = Rx(roll)·Ry(yaw)·Rz(pitch)` (three.js's XYZ order)
- **ORIENT and the centering (`obj.position.sub(caixa.getCenter())`) are NOT
  recomputed in Python.** `exportarPoses()` zeroes out `pivot` before
  exporting and applies `malha.matrixWorld` (which already includes the
  centering and `ORIENT`) to every vertex — so the exported data already
  comes in the "world with pivot at zero" frame. `desktop.py` only
  multiplies by `R` (the rotation coming from the IMU) on top of that. One
  fewer matrix reimplementation outside of three.js, one fewer chance to flip
  a sign.
- automatic spin: `+0.55 rad/s`, accumulated in its own counter (`spin`) that
  adds onto the yaw component — same effect as hand.js's
  `pivot.rotation.y += dt*0.55`
- 2.5 s with no data falls back to manual mode
- **Accepted loss from route B**: the transition between poses blends
  *already-skinned vertex positions*, not bones — for a hand opening/closing
  that's visually identical; for very abrupt transitions between very
  different poses it could "shrink" a little partway through (not observed
  by eye across the 4 real poses).

## Step 4 — The Myo in the same process — ✅ implemented

Reuses from `feed.py` **with no rewrite**: `Classificador`,
`euler_de_quaternion`, `descobrir_mac`, `ler_mac_salvo`, `salvar_mac`, and the
majority-vote smoothing (window of 25, `feed.HIST`).

- a Myo thread running `m.run()`, writing into `COMPARTILHADO` (a dict) under
  `LOCK`; the render loop only reads
- **no watchdog that kills the process.** pyomyo's `connect()` waits with no
  timeout and hangs if the armband is asleep — but since the window is
  already open, that's acceptable: the Myo thread just sits there blocked
  (manual poses keep working via the keyboard), and as soon as the armband
  wakes up `connect()` returns. On a disconnect WHILE RUNNING (not on the
  initial connect), the thread catches the exception, calls `m.disconnect()`
  and retries after 2 s — hence "tries to reconnect on its own" instead of
  the `os._exit()` that `feed.py` uses (which makes sense there, because
  `run.py` restarts the process; here there's no process to restart without
  closing the window)
- `--sim` makes up a gesture and orientation (same logic as `bridge.py`'s
  `simulador()`), to rehearse without the armband
- `--foto saida.png` renders a few frames (the window opens for real — an
  offscreen FBO with no graphics session isn't reliable, see Risks) and saves
  with PIL

## Step 5 — One single place for the gesture map — ✅ implemented

`hand3d/gestos.json` is the single source of truth: `bridge.py` and
`desktop.py` both read the file (each with its own ~8-line
`carregar_gestos()` — small enough to duplicate rather than build a shared
module for), `web/hand.js` fetches it with `fetch('gestos.json')` before
building the buttons and loading the model. `serve.py` gained a special route
to serve that file (it lives in `hand3d/`, outside the static root
`hand3d/web/`).

## Step 6 — Documentation — ✅ implemented

"Desktop mode" section in the [`README.md`](README.md) and
`hand3d/requirements-desktop.txt`.

---

## Files

| file | what it does |
|---|---|
| `hand3d/desktop.py` | window, shaders, loop, Myo on a thread |
| `hand3d/modelo.py` | reads/validates `hand.cache.npz`; `--importar` converts the JSON from the page's button |
| `hand3d/gestos.json` | class→clip map, single source of truth |
| `hand3d/exportar_poses_headless.py` | builds the cache without opening a browser (playwright, dev-only) |
| `hand3d/requirements-desktop.txt` | moderngl, moderngl-window, pyglet, numpy, Pillow (no assimp-py — route B doesn't read the FBX in Python) |
| `hand3d/README.md` | "Desktop mode" section |
| `hand3d/bridge.py` | reads `gestos.json` (`carregar_gestos()`) instead of the embedded dict |
| `hand3d/serve.py` | extra route to serve `hand3d/gestos.json` (outside the static root `web/`) |
| `hand3d/web/hand.js` | fetches `gestos.json`; gained `exportarPoses()` + an "export poses" button |
| `.gitignore` (root) | ignores `hand3d/web/model/hand.cache.npz` |

## Verification

1. **Spike** — ✅ done: assimp-py doesn't expose bones/animations for this
   FBX → route B.
2. **Fidelity by image** — ✅ done informally: `desktop.py --sim` renders
   all 4 poses correctly (open hand, fist, spread fingers, pointing — each
   with the right silhouette, eyeballed against the web page). No automated
   pixel-by-pixel side-by-side comparison was done.
3. **Window** — ✅ `python desktop.py --sim` opens, renders the 4 poses, the
   HUD updates (pose/source/euler/Hz/fps). FPS observed **well above 60**
   (300–500+) on this machine: the `vsync=True` requested from pyglet isn't
   being honored by the Intel driver in this setup — not a logic bug (real
   `dt` is used in the smoothing, not fps), but worth noting: on a machine
   where vsync actually works, it should cap around the monitor's refresh
   rate.
4. **With the armband** — ✅ **confirmed working** with a real Myo connected:
   live gesture classification and arm-orientation tracking both drove the
   hand correctly in the native window.
5. **Web regression** — ✅ done: `bridge.py --sim` + `serve.py` opened in an
   automated Chromium, all 4 pose buttons present, `fonte` showing "ponte em
   simulação" (bridge in simulation), zero console errors — after migrating
   `POSES` to `gestos.json` via fetch.

## Risks

| risk | how it shows up | diagnosis / outcome |
|---|---|---|
| assimp-py with no animations | spike lists no clips | **happened, and worse**: also no bones/weights → route B (not A') |
| swapped rotation order | mirrored hand, or spinning on the wrong axis | not observed; ORIENT+centering come pre-baked from three.js (Step 3) |
| `PreRotation` composed wrong | fingers twisted already in the bind pose | avoided by construction: the hierarchy is never recomposed in Python |
| inverted normals | dark hand, inverted-looking lighting | not observed across the 4 poses; `CULL_FACE` left off as a precaution |
| hybrid GPU falls back to Intel | low fps for no obvious reason | happened (opened on Intel), but fps came out **high**, not low — vsync that doesn't cap (see Verification 3) |
| offscreen FBO with no graphics session | `--foto` fails in the shell | avoided: `--foto` uses the real (visible) window, not an offscreen FBO |

## Out of scope

Don't touch the other repository's deck, don't remove the web version, don't
touch `src/myoControlsHand.py`. Desktop mode is an addition.
