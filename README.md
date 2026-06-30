# Metric-agnostic geodesic ray tracer (WebGPU)

A general-relativistic ray tracer that bends light by integrating **null geodesics**
through an arbitrary spacetime metric, entirely on the GPU via **WebGPU / WGSL**.
"Metric-agnostic" means the renderer has *no spacetime-specific code* except a single
function that returns the metric `gᵃᵇ(x)`. Everything else — the geodesic equations
and their integration — is derived automatically.

This is a browser port of the autodiff geodesic engine in
`../raylib_raytrace_blackhole_old` (C), restructured to run as a fragment shader.

![spacetimes: Schwarzschild + Minkowski] <!-- open index.html to view -->

---

## Run it

It's a single self-contained file — **open `index.html` in Chrome/Edge (v113+)**.
WebGPU is required. If your browser blocks WebGPU on `file://`, serve it:

```bash
npm run serve          # python -m http.server 8000  → http://localhost:8000
```

The panel only shows controls relevant to the selected spacetime (spin appears for Kerr,
throat/lensing + an "other-side image" appear for the wormhole, disk controls for the black
holes, etc.).

Controls: **drag** to orbit, **scroll** to zoom. The panel switches spacetime
(Schwarzschild / **Kerr** / **wormhole** / flat), mass, **spin a/M** (Kerr), accretion disk, background
(grid celestial sphere / starfield / **image** / none), and quality (steps/ray, step scale,
render resolution, exposure). Crank the spin and watch the shadow go asymmetric (a D-shape)
and the disk lensing skew — that's frame dragging.

**Accretion disk:** *Disk inner/outer (M)* set its extent (in units of M, so it scales with
the hole), *Disk thickness (M)* its vertical extent, *Disk brightness* its intensity. The
disk is **volumetric** — a Gaussian slab integrated as emission + absorption along each ray
(so a thick disk self-occults softly), not an infinitely-thin plane. Two relativistic effects
shape its colour and brightness, combined into one redshift factor `g`:
- **Doppler beaming** (toggle, on by default): material rotating toward you blazes (∝ δ³) and
  turns bluer; the receding side dims and reddens. The famous Interstellar / EHT asymmetry —
  and it flips with Kerr spin direction.
- **Gravitational redshift**: light climbing out of the well loses energy, so the inner disk
  is dimmed and reddened by the local lapse `1/√(−g⁰⁰)`. Always on (it's emission physics).

**Bloom:** the scene is rendered to an HDR buffer and the bright parts (beamed disk, stars)
are bright-passed, blurred, and added back — *Bloom* / *Bloom amount* control it. This is why
the beamed side of the disk glows.

**Wormhole (Thorne / Dneg):** the ultrastatic Ellis–Dneg wormhole from
[arXiv:1502.03809](https://arxiv.org/abs/1502.03809) — *no horizon*. Rays either thread the
throat into the **other universe** or are lensed back into ours, set by whether the impact
parameter is below the throat radius `ρ`. It's a separate trace path (the two-sheet topology
can't live in one Cartesian chart), and exploits spherical symmetry to integrate a pole-free
2D problem per ray. Knobs: throat radius `ρ`, throat length `2a`, lensing width `W`. Load an
**other-side image** to see a second sky through the throat (the iconic Interstellar shot) —
otherwise the far universe shows a warm-tinted grid so the two sides read as distinct.

**Custom sky:** pick a local image with *Sky image* to wrap it on the celestial sphere
(equirectangular / 2:1 panoramas look best — e.g. a Milky Way pano or an HDRI). It's mapped
by ray direction (longitude `atan2`, latitude `acos`), so you watch it lens and smear around
the shadow. Loading a file auto-switches Background to *Image*. Everything stays local —
the image never leaves your machine.

**It renders on demand** — the GPU only draws when you change something, then idles
(the FPS readout shows `idle`). The disk swirl is off by default; tick **Animate disk**
for continuous motion (capped at 30 fps). See [Performance](#performance).

---

## How it works

The whole thing is the same idea used to avoid hand-deriving Christoffel symbols:
**use the Hamiltonian form of the geodesic equation, and get the metric derivatives
by automatic differentiation.**

```
                                                  ┌─────────────────────────────┐
  per pixel: shoot a backward null ray            │  metricInverse(x) → gᵃᵇ      │ ← ONLY
        │                                         │  (returned as dual numbers)  │   physics-
        ▼                                         └──────────────┬──────────────┘   specific
  makePhoton(): null initial momentum (H = 0)                    │                  code
        │                                          autodiff gives both gᵃᵇ and ∂gᵃᵇ
        ▼                                                         │
  RK4 integrate Hamilton's equations  ◄───────────  rhs():  dqⁱ/dλ =  gⁱʲ pⱼ
        │                                                    dpᵢ/dλ = -½ (∂ᵢ gᵃᵇ) pₐ p_b
        ▼
  terminate: horizon → black · disk plane → emission · escaped → lensed starfield
```

### 1. Dual numbers (forward-mode autodiff)
`struct Dual { v: f32, d: vec4<f32> }` carries a value and its gradient
`(∂/∂t, ∂/∂x, ∂/∂y, ∂/∂z)`. Seeding each coordinate's gradient as a basis vector and
evaluating the metric once yields **every partial derivative** the equations of motion
need. (The original C did one evaluation per partial; widening the dual to a 4-gradient
collapses that to a single evaluation — the right shape for the GPU.)

### 2. The metric is the only spacetime-specific code
`metricInverse(q, metric, M, a)` returns the contravariant metric `gᵃᵇ` as a 4×4 of duals.
That is the *entire* definition of "which universe we're in." Three are built in:
`FLAT` (Minkowski — straight lines, a sanity check), `SCHWARZSCHILD` (isotropic Cartesian
coordinates), and `KERR` (rotating, in **Kerr–Schild** Cartesian coordinates — `gᵘᵛ =
ηᵘᵛ − f lᵘlᵛ`). Both black-hole metrics are horizon-penetrating, so there's no coordinate
singularity at the horizon — important for f32. Kerr's metric is **non-diagonal** (`g⁰ⁱ ≠ 0`),
which is frame dragging; the geodesic engine handles it without any special-casing, and
`makePhoton` solves the full quadratic null condition rather than assuming a diagonal metric.

### 3. Hamiltonian geodesics, integrated with RK4
`rhs()` implements `dqⁱ/dλ = gⁱʲ pⱼ` and `dpᵢ/dλ = -½ (∂ᵢ gᵃᵇ) pₐ p_b` using only the
duals returned by the metric — it never knows which metric it's integrating. `rk4()`
advances the 8-vector state `(qᵃ, pₐ)`. Step size shrinks near the hole
(`h = clamp(stepScale·ρ, hmin, hmax)`).

### 4. Shading
A ray ends in one of three ways: **captured** (black), it crosses the equatorial disk
plane within `[rIn, rOut]` (temperature-ramped accretion-disk emission), or it **escapes**
past `escapeR` — in which case the **bent** outgoing direction samples the background, so
you see the lensing.

Capture is each spacetime's **event horizon** — the metric-derived null surface (`Δ = 0`),
following standard GRRT practice (RAPTOR, ipole, Blacklight, etc.). This is *not* an arbitrary
`2GM/c²`: it's the actual horizon, and it reduces to `2M` for Schwarzschild.

- **Kerr:** the **Kerr radial coordinate** `r < r₊ = M + √(M²−a²)`. Crucially this uses the
  oblate Kerr radius, exact at every latitude — using the Euclidean radius (or a local
  null-direction scalar) is only right on the equator and lets near-extremal rays leak
  through the shadow off-equator. (This was a real bug; the fix was switching to `r`.)
- **Static slicings** (isotropic Schwarzschild): the lapse → 0 at the horizon, so the
  redshift `−g⁰⁰ = 1/lapse²` → ∞; a threshold catches it (and doubles as a deep-field safety).
- **Step-budget exhaustion** → black: a photon that neither escapes nor turns back is trapped.

A subtlety worth knowing: a backward-traced ray reaches the horizon only at *infinite* affine
parameter (it asymptotes), so you terminate on a thin shell just outside `r₊` rather than
waiting to "hit" it. Near-extremal (`a ≳ 0.95`) still leaves faint photon-ring fuzz at the
shadow *edge* (the capture/escape boundary is nearly fractal there) — the interior is solid;
push *Steps / ray* up and *Step scale* down to sharpen the edge.

---

## Add a new spacetime

This is the payoff. Kerr (added as `metric == 2`) is the worked example: it's just one new
branch in `metricInverse` — nothing in `rhs`, `rk4`, `makePhoton`, or the renderer changed
(only `makePhoton` was generalized once to the quadratic null solve, which all metrics use).
To add, say, a wormhole or charged (Reissner–Nordström) metric, add `metric == 3`:

```wgsl
if (metric == 3) {
  // build dual coordinates (seed gradients as basis vectors), then fill m.c[a*4+b]
  // with gᵃᵇ as duals using dAdd/dMul/dDiv/dSqrt/... Off-diagonal terms are fine —
  // just set both m.c[i*4+j] and m.c[j*4+i] (see the Kerr branch for the pattern).
  return m;
}
```

Mirror the same function in `test/engine.mjs` and the validation suite covers it too.

> Note on the test interpreter: `wgsl_reflect`'s CPU interpreter has a bug evaluating
> nested dual-number calls and variable-argument metric calls. The Kerr WGSL is therefore
> written in flat SSA form (one op per `let`), and `wgsl_exec.mjs` checks the Kerr metric
> via a literal-argument call. Both are interpreter work-arounds; real GPUs (Tint/Dawn)
> have well-defined WGSL semantics and are unaffected.

---

## Validation

Because WebGPU can't run headless here, correctness is established **without a GPU**, in
three layers (`npm test`):

| Check | File | Result |
|---|---|---|
| Autodiff vs. central differences | `test/test.mjs` | matches to ~1e-10 |
| Flat space ⇒ exactly straight rays | `test/test.mjs` | `1−dot ≈ 2e-16` |
| Light deflection vs. `α = 4M/b + (15π/4)(M/b)²` | `test/test.mjs` | <0.3% incl. 2nd-order GR term |
| Photon inside shadow falls to horizon | `test/test.mjs` | reaches ρ→M/2 |
| Kerr autodiff vs. central differences (incl. off-diagonal) | `test/test.mjs` | matches to ~1e-10 |
| Kerr with `a=0` reduces to Schwarzschild deflection | `test/test.mjs` | <0.01 |
| Null condition `H≈0` created and **conserved** along the geodesic | `test/test.mjs` | `|H|/|p|²` < 1e-8 (Schwarzschild & Kerr) |
| Frame dragging: prograde < Schwarzschild < retrograde deflection | `test/test.mjs` | monotone in spin |
| Capture surfaces: Kerr `r → r₊` at all latitudes (→2M at a=0); Schwarzschild redshift | `test/test.mjs` | r₊ exact eq & off-eq; `2.0` at a=0 |
| Wormhole thread-vs-reflect dichotomy (b<ρ → other universe, b>ρ → back) | `test/test.mjs` | ℓ→−35 vs +35 |
| WGSL parses | `test/wgsl_check.mjs` | OK |
| **The actual shader code, CPU-executed, vs. the validated JS engine** | `test/wgsl_exec.mjs` | Flat/Schwarzschild full pipeline + Kerr metric all match to ~1e-7 |

The Hamiltonian-conservation test is the linchpin for Kerr: `H = ½ gᵃᵇ pₐ p_b` stays 0 to
~1e-9 along the integration only if the off-diagonal metric *and* its autodiff derivatives
are mutually consistent — a wrong `∂g` would make it drift.

The last row is the key one: `test/wgsl_exec.mjs` extracts the engine WGSL straight out
of `index.html` and runs it on the CPU (via `wgsl_reflect`'s interpreter), so the shader
that ships is the shader that's tested. The JS twin in `test/engine.mjs` is kept
line-for-line parallel to the WGSL.

```bash
npm install   # restores wgsl_reflect (dev-only)
npm test
```

**Not auto-verified:** real-GPU shader compilation (the WGSL parses and its semantics are
confirmed on CPU, but driver-specific compilation isn't) and the visual presentation
(camera, disk look, tonemap). Open `index.html` to confirm those. Compilation errors, if
any, are surfaced in an on-screen overlay.

---

## Performance

Per pixel this integrates RK4 (4 metric evaluations/step) for up to a few hundred steps,
so it's a genuinely heavy fragment shader — left running unthrottled it will peg a GPU.
Two things keep it in check:

- **On-demand rendering.** The frame loop only draws when the camera or a control changes
  (`dirty` flag), then idles. A static view costs ~nothing. The disk animation is opt-in
  (**Animate disk**) and capped at 30 fps.
- **Quality knobs.** *Render resolution* (internal scale), *Steps / ray*, and *Step scale*
  all trade fidelity for speed. Drop resolution or steps first if a frame feels slow.

If your GPU still runs hot, lower *Render resolution* to ~0.6 and *Steps / ray* to ~250 —
the lensing stays correct, only the photon-ring sharpness softens.

**Step scale and the shadow.** The integrator step is `h ≈ stepScale · ρ`, refined smaller
in the strong field. A *larger* step scale leaps over the tight windings near the photon
orbits, so rays leak through the shadow (it looks see-through); a *smaller* one resolves
them and the shadow goes solid — at the cost of more steps (raise *Steps / ray* to match).
This matters most for **near-extremal Kerr** (`a ≳ 0.95`): the photon-orbit structure there
is nearly fractal, so its shadow edge stays a little noisy unless you push step scale down
(~0.04) and steps up (~800+). Through `a ≈ 0.9` the defaults give a clean, solid shadow.

## The f32 caveat

WebGPU shaders are **f32-only** (no f64). The physics tests above run in double precision;
the shader runs in single. This matters most in the strong-field region near the horizon
and photon sphere, where geodesics are stiff. Mitigations already in place: isotropic
coordinates (no coordinate singularity at the horizon), step-size shrinking near the hole,
and terminating rays just outside the horizon. If you push `M` high or zoom into the photon
ring, expect some f32 shimmer — that's the precision ceiling, not a logic bug.

---

## Files

```
index.html          self-contained app: WGSL engine + WebGPU host + UI
test/engine.mjs     double-precision JS twin of the WGSL engine
test/test.mjs       physics validation against analytic GR
test/wgsl_check.mjs WGSL parse smoke-test
test/wgsl_exec.mjs  runs the real shader on CPU, compares to engine.mjs
package.json        `npm test`, `npm run serve`
```

## Relation to Galacto

[galacto.org](https://galacto.org) is a GPU N-body simulation (brute-force gravity,
symplectic leapfrog) in Rust→wasm→`wgpu`. This project shares the "all the physics lives
in a WGSL kernel" approach but solves a different problem: instead of evolving N
interacting bodies, each GPU thread integrates one independent light ray through curved
spacetime. The Rust/`wasm-pack` wrapper Galacto uses is an optional packaging layer — the
WGSL engine here is identical with or without it — so this port stays dependency-free.
```
