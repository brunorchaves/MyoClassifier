# hand3d — the 3D hand in the browser, no Unity

The same `handAnimations.fbx` the Unity project used, rendered with
[three.js](https://threejs.org) and animated by the Myo's classification. Runs
in a browser tab: no editor to open, no Play to hit, no window that needs
focus.

```bash
cd hand3d
python run.py
```

One command brings up the bridge, reads the armband, and opens the page.
`Ctrl+C` tears everything down.

---

## What replaced what

| Unity (`3DORientaion_test/`) | now |
|---|---|
| `myListener.cs` — TCP server on 25001 → `transform.rotation` | `bridge.py` → WebSocket → `pivot.rotation` in `web/hand.js` |
| `handController.cs` — `Animator` with a Grip × Trigger blend tree | `AnimationMixer` with one weight per clip |
| gesture via **emulated keypress** (`Input.GetKeyDown`, the `keyboard` lib) | `gesture` field in the JSON |
| editor + build + a window that needs focus | one browser tab |

The detail that made this simple: the FBX carries the four poses (`Relaxed`,
`fist`, `spock`, `Pointing`) as **single-frame clips**. A single-frame clip is
exactly what a blend tree mixes — so the smooth transition between gestures
comes from interpolating a weight, not from a recorded animation.

`keyboard.press()` was the most fragile part of the old setup: it only worked
with the Unity window focused, and on some systems it asks for admin
privileges. Now the gesture travels in the same packet as the orientation.

---

## The pieces

```
hand3d/
  run.py        brings everything up with one command, retries if the Myo is asleep
  bridge.py     WebSocket :8765 + TCP :25001 — Unity's old spot
  feed.py       reads the Myo (pyomyo), classifies (1-NN) and feeds the bridge
  serve.py      static server for the page
  gestos.json   class -> pose, single source of truth (bridge.py, desktop.py, hand.js)
  desktop.py    desktop mode: native window, no browser (see below)
  modelo.py     reads/validates the pose cache for desktop mode
  exportar_poses_headless.py   builds the cache without opening a browser (dev-only)
  web/
    index.html  the UI
    hand.js     three.js: loads the FBX, blends the poses, listens to the bridge
    model/hand.fbx     copy of Assets/handAnimations.fbx
    model/hand.cache.npz   desktop-mode cache (generated, not in git)
    vendor/            three.js r147 + FBXLoader (vendored, works offline)
```

### `bridge.py` — speaks Unity's protocol

Listens on **port 25001 with the same protocol as `myListener.cs`**, including
the echo that `sock.recv()` in `src/myoControlsHand.py` expects. In other
words: **your old script runs unchanged** and the hand already turns.

To get the **gesture** through too, it's one line in it:

```python
data_str = f"{roll},{-yaw},{pitch}"                 # before
data_str = f"{roll},{-yaw},{pitch},{int(pose)}"     # after
```

The WebSocket listens on 8765, and that's where the page connects.

### `feed.py` — the recommended path

Replaces `src/myoControlsHand.py` for this use case, and fixes two problems
with it:

- **Dependencies.** The old script imports `pygame`, `keyboard`, `joblib` and
  `scikit-learn`. `feed.py` only uses `pyomyo` (already in `src/`) and
  `numpy`.
- **The scan that hangs.** `pyomyo`'s `connect()` waits for the connection
  event **with no timeout**, and the Myo goes to sleep when idle — the
  process hangs forever. `feed.py` connects **by MAC**, which it discovers
  once and stores in `myo_mac.txt`, and has a watchdog that gives up after
  10 s with a clear message. `run.py` then retries: once you pick up the
  armband and move it, the next attempt gets through.

Classification stays true to yours: the same **1-NN** over the same
`src/data/vals*.dat`, and the same majority-vote smoothing over a 25-sample
window.

---

## Modes

```bash
python run.py                 # bridge + Myo + page
python run.py --sim           # no armband: gesture and orientation made up
python run.py --sem-myo       # bridge + page, and you feed it however you like
python run.py --porta 9000
```

Or the pieces by hand, in three terminals:

```bash
python bridge.py
python feed.py
python serve.py
```

With no bridge at all, the page still opens and the **side buttons** (or the
`1`–`4` keys) drive the pose. Useful for checking the model, or for showing
the poses when the armband isn't cooperating.

---

## The page

- **drag** to orbit, **scroll** to zoom, **`1`–`4`** switch the pose
- **follow the IMU** — turns off the rotation coming from the armband, if you
  just want the poses
- **skeleton** and **wireframe** — to check the rig
- **EMG per channel** — the 8 bars, fed by the bridge
- the label in the top-left says where the data is coming from: `Myo ao vivo`
  (live), `ponte em simulação` (bridge in simulation), `ponte no ar, sem dado
  do bracelete` (bridge up, no data from the armband), or `sem ponte` (no
  bridge). It **doesn't** say "live" when it isn't — that matters in a demo.

---

## ⚠️ One gesture is missing training data

`src/data/vals*.dat` has 976 samples, but only for classes 1, 2 and 3:

| class | samples | clip |
|---|---|---|
| 1 | 259 | `Relaxed` — open hand |
| 2 | 315 | `fist` — closed fist |
| 3 | 402 | `spock` — spread fingers |
| 4 | **0** | `Pointing` — **will never come out of the classifier** |

The "pointing" button works (it's manual), but the Myo will never classify
that pose. Record class 4 with `src/emgGestureTrainer.py`, or adjust the
`GESTOS` map in `bridge.py` (and `POSES` in `web/hand.js`) to work with three
gestures. `feed.py` warns about this on startup.

---

## Desktop mode — native window, no browser

```bash
cd hand3d
pip install -r requirements-desktop.txt
python desktop.py                    # reads the Myo directly, in the same process
python desktop.py --sim              # no armband: gesture/orientation made up
python desktop.py --foto quadro.png  # renders one still frame and exits
```

No bridge, no server, no port: `desktop.py` reads the Myo in the same process
(reuses `feed.py`) and draws into a native OpenGL window (moderngl +
moderngl-window + pyglet). Controls: **drag** to orbit, **scroll** to zoom,
**1**–**4** switch the pose, **g** spins, **w** wireframe, **i** toggles
following the IMU, **f** re-frames the camera.

The FBX can't be read in Python with bones/weights/animations available (see
[`PLANO-desktop.md`](PLANO-desktop.md), Step 0) — `assimp-py`, the only
package installable with just `pip` on this machine, doesn't expose that for
this file. The data instead comes pre-baked from `web/model/hand.cache.npz`:
the web page itself (which already knows how to do skinning) computes the
already-skinned position and normal of each of the 4 poses and exports that
once — via the "export poses" button on the page, or without opening a
browser with `python exportar_poses_headless.py` (dev-only,
`pip install playwright && playwright install chromium`). `desktop.py` just
reads that cache; the GPU blends 4 positions per vertex, weighted by pose —
simpler than bone skinning, at the cost of losing the bone-space blend during
the transition between gestures (imperceptible in practice).

The class → pose map lives in [`gestos.json`](gestos.json), the single source
of truth read by `bridge.py`, `desktop.py`, and by `web/hand.js` via `fetch`.

On a machine with hybrid GPUs (Intel + NVIDIA), the window may open on the
Intel one — the terminal prints `GL_RENDERER` on startup to confirm. To force
the NVIDIA one, add `python.exe` under Settings → System → Display → Graphics
settings, and mark it "high performance".

---

## Can the Unity project be deleted?

From this folder's point of view, yes: the FBX is copied into
`web/model/hand.fbx` and nothing else here looks at `3DORientaion_test/`. That
folder was never committed to the repository (`.gitignore` now explicitly
ignores it), and Unity's `Library/` takes up a fair amount of space.

Before deleting it, keep the original `.blend` file somewhere — that's where
the rig comes from, and the FBX is just the export.

---

## Technical notes

- **three.js r147** is the last version with the classic scripts
  (`examples/js/`), which load via `<script src>` with no ES modules. That
  keeps the page working when opened from a plain server too, with no
  bundler. It's vendored under `web/vendor/` to work offline.
- The FBX is **binary 7.4**, which `FBXLoader` opens directly: 29 bones, 1
  skinned mesh, 105,696 vertices, 4 clips. Loads in ~0.8 s.
- The model comes in lying down (Blender is Z-up), so there's a fixed axis
  correction (`rx: -90, rz: 90`) in a group between the pivot and the mesh. If
  you swap the model and it shows up crooked, that `ORIENT` in `hand.js` is
  what needs adjusting.
- The WebSocket handshake in `bridge.py` is hand-written, no dependencies. If
  you touch it, check it against RFC 6455's test vector: the key
  `dGhlIHNhbXBsZSBub25jZQ==` has to produce
  `s3pPLMBiTxaQ9kYGzzhZRbK+xOo=`. One letter out of place in the magic GUID
  makes the browser refuse the connection with `code 1006` and no useful
  message.
