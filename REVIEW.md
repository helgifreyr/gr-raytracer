> **Status (2026-07-01): Batches A + B + C + D applied.**
> Fixed — MAJ-1 (disk-noise early-out gate), MAJ-3 (`pushErrorScope` around pipeline + target
> creation → on-page panel), MAJ-4 (shaders fetched inside the IIFE with `r.ok` checks → `file://`
> failures now reach the error panel), MIN-1 (phase wrapped at 3600 s), MIN-2 (relaxed the hard
> density gate to 4H), MIN-3 (cached color-attachment views + resolution-slider-only retargeting),
> MIN-4 (image-load failures surface in `#hint`), MIN-5 (docs overlay: tab roles/`tabindex`, Enter/
> Space, `role="dialog"`/`aria-modal`, focus in on open + restore on close), MIN-6 (`pointercancel`
> ends the drag + releases capture), MIN-7 (reduced-motion keeps disk animation off), PHY-1/PHY-2
> (physics.html now notes the Newtonian-β approximation and that the lapse identity is exact only for
> a static slicing). **Partially done:** MAJ-2 — device-loss now branches on `reason` and shows a
> clear reload message; full auto re-acquire/rebuild was deferred (can't be verified without a real
> GPU, and a mis-wired rebuild would break the normal path silently). Nitpicks left by choice.

# Code Review — Metric-Agnostic GR Ray Tracer (`H:/study/raytracer`)

**Review date:** 2026-07-01 · **Method:** four specialist agents (GR physicist, WGSL specialist,
WebGPU specialist, JavaScript engineer), reusable definitions in `.claude/agents/`.
Prior review archived in `REVIEW-archive-2026-06.md`.

## Executive summary

Overall health is **very good**. `npm test` is green (21 physics + parse checks + 17 CPU-executed
WGSL-vs-engine parity, agreement ~1e-7). **No Critical issues, and no Major *correctness* bugs in the
shipped physics, shaders, or render graph.** All four reviewers independently confirmed the core is
sound: the Hamiltonian geodesic integrator, both black-hole metrics (incl. Kerr–Schild `l^μ`/`f`/Kerr
radius and frame dragging), the future-directed photon root, both horizon-capture criteria, the
wormhole path, the uniform-buffer layout, every bind-group index, and the 8-pass bloom render graph.

The actionable findings cluster into four themes: **(1) one real GPU perf hotspot** (the disk noise
loop), **(2) robustness/error-surfacing gaps** (device loss, uncaptured errors, the `file://` blank
screen), **(3) per-frame/per-event resource churn**, and **(4) accessibility of the docs overlay**.
Plus a small **documentation-accuracy** fix (the lapse formula is exact only for the static slicing).

---

## Verified correct (what NOT to worry about)

- **Physics:** Hamiltonian RHS + full (a,b) double sum; Schwarzschild isotropic `g^{tt}=-(B/A)²`,
  `g^{ii}=B^{-4}` (2nd-order deflection pinned to <0.3%); Kerr–Schild `g^{μν}=η−f l^μ l^ν` with all
  signs correct and `a→0`→Schwarzschild; future-directed photon root (E>0, infalling branch);
  Kerr capture `r<r₊` exact off-equator; wormhole `r(ℓ)` + smooth radial ODE + thread/reflect
  dichotomy. All equations in `physics.html` correct as written.
- **Shaders:** dual-number rules, RK4 stages + `m0` reuse, row-major indexing, `select()` arg order,
  `textureSampleLevel` (LOD 0) everywhere (correct under non-uniform flow).
- **Host/WebGPU:** UBO sizing exact (40 floats = 10×vec4, no padding traps); all scene/blur/composite
  binding indices match the WGSL (incl. Comp uniform @5); format matching; sampler address modes;
  resize destroys old targets; `deviceLost` guard stops submission; render-on-demand + clock
  re-anchor.
- **JS:** `u`-array packing model, `data-m` metric-visibility system, `createImageBitmap` guards +
  `bmp.close()`, KaTeX lazy-load with offline fallback, content-fetch `.catch()`.

---

## Findings by severity

### Critical
None.

### Major

**MAJ-1 — Disk noise loop is the GPU hotspot and runs full cost on empty/grazing samples** · perf ·
`scene.wgsl` `diskDensity` (~line 436) + sub-sample loop.
Each disk sub-sample issues ~17 `vnoise` (= ~136 `hash31`): 2×`fbm` domain-warp + `fbm` envelope +
5-octave `fbmRidged`. ×8 samples ≈ **~1088 `hash31` per geodesic step inside the disk bbox**. The
warp/FBM run *before* any density gate, so grazing rays that contribute ~0 still pay full price.
**Fix (cheapest first):** (1) compute the cheap analytic `radial*vfall` first and `continue` when
below ε *before* the warp/FBM — recovers most wasted work; (2) drop `fbmRidged` 5→4 octaves (5th is
sub-pixel); (3) fewer sub-samples (4–6) when the disk is thin. *(WGSL reviewer)*

**MAJ-2 — No recovery from a real device loss** · robustness · `main.js:37`.
`device.lost.then()` only shows the fail overlay. A transient TDR/driver-reset/backgrounding loss
(common on laptops) then bricks the page until manual reload. **Fix:** branch on
`info.reason === "destroyed"` (intentional) vs. real loss; for real loss, re-acquire adapter/device
and rebuild pipelines/buffers/textures/bind groups. *(WebGPU reviewer)*

**MAJ-3 — Validation/uncaptured errors never reach the UI; no `pushErrorScope`** · robustness ·
`main.js:38`. `uncapturederror` is only `console.error`'d; a future binding/pipeline mismatch would
silently yield a black canvas. **Fix:** wrap pipeline + first bind-group creation in
`device.pushErrorScope("validation")`/`popErrorScope()` → route to `fail()`. Cheap self-diagnosing
insurance. *(WebGPU reviewer)*

**MAJ-4 — `file://` / fetch failure = blank screen that bypasses the `#err` fallback** · robustness ·
`main.js:10,16`. The shader fetches are module-top-level `await`s that run *before* the IIFE's
`fail()` helper, and don't check `r.ok`. The single most likely first-run failure (double-clicking
`index.html`) shows a black page with no guidance — defeating the very "serve over http" message the
HTML was written to show. **Fix:** move the fetches inside the IIFE after the `navigator.gpu` check,
add `r.ok` checks, and route failure to `fail()`. *(JS reviewer)*

### Minor

**MIN-1 — Animated noise phase decays in f32 over long runtimes** · `scene.wgsl:333`.
`camFwd.w = time` grows unbounded; once `time·swirl·noiseScale` ≳ 2^18–2^20 the disk noise quantizes
and eventually freezes into blocky cells. **Fix:** fold the phase in JS before upload, e.g.
`time = ((now - t0)/1000) % 3600`. *(WGSL reviewer)*

**MIN-2 — Hard density gate causes photon-ring shimmer** · `scene.wgsl:435`. The `abs(pp.z)<3H` + `rc`
bounds are hard cutoffs; `diskDensity` already has smooth radial/vertical windows, so the hard gate is
redundant *and* clips the smooth tails, producing edge shimmer that bloom amplifies. **Fix:** relax
the hard gate to a loose work-skipping bbox and let the smooth density antialias. *(WGSL reviewer)*

**MIN-3 — Per-frame `createView()` churn + resize/slider retargeting** · `main.js:372-374, 196, 307`.
8 `createView()` calls/frame (only the swapchain view must be per-frame); every UI `input` calls
`resize()` (only the `res` slider needs it); rapid resize rebuilds 7 textures + 7 bind groups per
event. **Fix:** cache offscreen views in `rebuildTargets()`; move `resize()` to the `res` slider only;
optionally rAF-debounce the resize handler. *(WebGPU + JS reviewers — same theme)*

**MIN-4 — Image-load failures are silent to the user** · `main.js:213-215, 231-233`. Both catch
blocks only `console.error`. **Fix:** surface a brief message (reusing `#hint` is proportionate).
*(JS reviewer)*

**MIN-5 — Docs overlay is keyboard-inaccessible** · `index.html:60-63`, `main.js:262-277`. Nav tabs
are `<span>` (not focusable, no Enter/Space); overlay never receives focus, no `role="dialog"`/
`aria-modal`, active tab not marked. The whole Physics/Engineering docs system can't be reached by
keyboard/screen-reader. **Fix:** make tabs `<button>`s, add `role="dialog" aria-modal="true"`, focus
the close button on open, restore focus on close. *(JS reviewer)*

**MIN-6 — `pointercancel` not handled → stuck drag** · `main.js:281-282`. `setPointerCapture` is never
released and there's no `pointercancel`; an interrupted touch gesture can leave the orbit stuck.
**Fix:** add a `pointercancel` listener clearing `dragging`. *(JS reviewer)*

**MIN-7 — No `prefers-reduced-motion` handling** · `main.js:322`. Low impact (animation is off by
default). **Fix (optional):** gate any future auto-animate behind the media query. *(JS reviewer)*

### Physics modelling notes (acceptable — disk is declared artistic; one doc fix)

**PHY-1 — Disk orbital speed `β=√(M/rc)` is Newtonian and in coordinate (not areal) radius** ·
`scene.wgsl:438`. Beaming direction is right; the δ³ *magnitude* is off by an O(1) factor in the
strong field, and Kerr orbital speed ignores spin (only rotation sign is used). Acceptable given the
disk is explicitly artistic; for fidelity use `β=√(M/rc)/√(1−2M/rc)` in areal radius. *(GR reviewer)*

**PHY-2 (doc fix worth doing) — `α_lapse = 1/√(−g^{00})` is exact only for the static slicing** ·
`physics.html:87` presents it as exact, but in Kerr–Schild `−g^{00}=1+f ≠ 1/α²` (nonzero shift). The
disk redshift applied to Kerr is the KS-time redshift, not the true static-observer lapse. **Fix:**
note in `physics.html` that the lapse identity holds for the static (Schwarzschild) slicing shown;
label the Kerr disk redshift accordingly. *(GR reviewer)*

**PHY-3 — Redshift-threshold capture (`−g^{00}>50`) is slicing-dependent** · `scene.wgsl:407`. No live
bug (Schwarzschild isotropic leaks negligibly at r≈2.02M; Kerr has its own `r<r₊`), but a future
horizon-penetrating static-looking metric added under `metric!=2` would leak. Flag for future metrics.
*(GR reviewer)*

### Nitpicks (no action needed)
- `postSampler` comment says "clamp" but relies on the default (correct, implicit). *(WebGPU)*
- `syncLabels()` is 16 near-identical lines; could be table-driven if it grows. *(JS)*
- Magic numbers (drag 0.006, zoom 0.0009, pitch ±1.48, dist [6,120], escapeR) → optional named consts. *(JS)*
- `image2Loaded` uses 0/1 while siblings use `.checked`; cosmetic consistency. *(JS)*
- `otherBackground` channel-swizzle can push >1 into HDR (harmless, tiny extra bloom). *(WGSL)*
- `rhsM` defensive `var` copies may be unneeded on current Tint (leave if targeting old Dawn). *(WGSL)*

---

## Recommended fix batches

- **Batch A — Robustness (high value, low risk):** MAJ-4 (`file://` + `r.ok`), MAJ-3 (pushErrorScope +
  surface uncaptured errors), MAJ-2 (device-loss reason branch / recovery), MIN-4 (image-load
  feedback), MIN-6 (pointercancel).
- **Batch B — Performance:** MAJ-1 (gate disk noise behind cheap density), MIN-1 (phase wrap),
  MIN-2 (relax hard gate), MIN-3 (cached views + res-only resize + debounce).
- **Batch C — Accessibility:** MIN-5 (overlay keyboard/aria), MIN-7 (reduced-motion).
- **Batch D — Docs accuracy:** PHY-2 (lapse-slicing note), PHY-3 (comment/guard note), PHY-1 (optional
  fidelity note or fix).
