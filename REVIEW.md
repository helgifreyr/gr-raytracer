> **Status (fixes applied):** Batches A + B + C from this review have been applied.
> Addressed: H1 (dedup metric eval), L1 (bloom isotropy), L2 (exposure on all backgrounds),
> L4 (animate clock re-anchor), L6 (disk spin sign), M6 + L3 + L8 (device-loss handling,
> NaN floor, g00 guard + documented precondition), and the test-integrity items — POST_WGSL
> now parse-checked, the Kerr full pipeline (rhsM/makePhotonM) is now CPU-executed, the
> wormhole is single-sourced in engine.mjs, and the README "what's tested" claim is corrected.
> Remaining (low/nit, by choice): host-JS uniform packing has no automated test (needs a real
> GPU; layout is documented + review-verified); N1/N2/N3 cosmetic cleanups.

# Code Review Report — Metric-Agnostic GR Ray Tracer (`H:/study/raytracer`)

## 1. Executive Summary

Overall code health is **good**. The GR physics core (metric autodiff, Hamiltonian geodesic `rhs`/`rk4`, `makePhoton` null-root, horizon capture) is correct and genuinely validated against analytic GR and a double-precision JS twin. No correctness bugs were found in the shipped ray-tracing math. The substantive findings cluster in two areas: **test-coverage integrity** (the test suite executes only the geodesic engine slice of the shader, while `trace()`/`disk()`/`traceWormhole()`/backgrounds/post-processing and the host uniform packing are never CPU-executed, and the README overclaims "the shader that ships is the shader that's tested"), and **one performance hotspot** (a duplicate `metricInverse` evaluation every integration step). Everything else is low-severity polish — cosmetic shader/UX issues, magic numbers, and defensive-hardening gaps that are latent rather than firing. This is a healthy result for a project of this complexity.

## 2. Findings by Severity

### Critical
None.

### High

**H1 — Redundant `metricInverse` evaluation every geodesic step (performance)**
`index.html:400` and `rhs`/`rk4` at `index.html:220,242`.
The trace loop calls `metricInverse(s.q, ...)` at line 400 (for `redshift`/`radial`), then `rk4(s, ...)` at line 417 whose first stage `k1 = rhs(s,...)` (line 242 → 220) re-evaluates `metricInverse(s.q,...)` at the identical, unmutated state. `metricInverse` is the dominant inner-loop cost (full dual-number Kerr/Schwarzschild autodiff). This is 5 evals/step where 4 suffice — ~20% of the hottest op, across up to 1200 steps/pixel/frame.
**Fix:** compute `m = metricInverse(s.q,...)` once at loop top; derive `redshift`/`radial` *and* `k1` from it via an `rhsFromMetric(m, p)` helper, and pass `k1` into an `rk4` variant that skips recomputing the first-stage metric. Result-preserving.

### Medium

**M1 — Test suite never CPU-executes `trace()`, `disk()`, `traceWormhole()`, capture logic, or backgrounds**
`test/wgsl_exec.mjs:11-13`; shader from `index.html:273` (`// ---- shading`) onward.
`wgsl_exec.mjs` slices the WGSL from `struct Dual` to `// ---- shading` and only wraps `metricInverse`/`rhs`/`rk4`/`makePhoton`. The capture thresholds (`redshift>50`, `rPlus*1.01`), the volumetric disk transfer/beaming, the wormhole integrator, and all backgrounds are validated only by hand-reimplemented JS twins, not the shipped shader. A bug in `trace()` or `disk()` (wrong capture threshold, flipped Doppler sign) passes `npm test`. The file's header claim that it executes "the ACTUAL shader engine" overstates coverage.
**Fix:** add `WgslExec` kernels that call `trace()`/`disk()`/`traceWormhole()` (feeding `u` via a storage buffer) and compare to the JS twins — or at minimum document loudly that these paths are unexecuted by any test.

**M2 — Kerr full pipeline (`rhs`/`rk4`/`makePhoton`) is never CPU-executed; only the bare metric tensor is, via a literal arg**
`test/wgsl_exec.mjs:86-116`.
Section B verifies Kerr by calling `metricInverse(..., 2, 1.0, ...)` with a hard-coded literal `metric=2` (because `WgslExec` mis-binds Kerr when `metric`/`M` arrive as variables through `rhs`'s parameter forwarding). But the shipped shader always reaches Kerr via `rhs`/`rk4`/`makePhoton` with these as runtime variables (`index.html:386,400,412,417`). So the off-diagonal `g^{0i}` derivative sum in `rhs` and the non-diagonal `makePhoton` root are never executed for Kerr. Comparing `engine.mjs` to itself proves nothing about the shipped WGSL.
**Fix:** add a Kerr full-pipeline kernel that hard-codes `metric=2` as a literal *inside* the kernel but still routes through `rhs`/`rk4`/`makePhoton`, comparing to `engine.mjs`.

**M3 — `makePhoton` null-root is asserted only via JS twin, not the shipped WGSL** *(closely related to M2)*
`test/test.mjs:233-277`; `test/wgsl_exec.mjs:80`.
Test [10] ("catches a wrong `p_t` root") runs entirely on `engine.mjs` + its own JS `captured()` helper. `wgsl_exec` compares `makePhoton` momentum (line 80) only for Schwarzschild/Flat — where the root choice is explicitly harmless (B=0). The one case where the root sign matters (Kerr, off-diagonal B≠0) is exactly what `wgsl_exec` cannot run (see M2). The two `makePhoton` copies agree today, so this is a drift risk, not a live bug.
**Fix:** the same literal-`metric=2` Kerr kernel from M2 should compare `p0` to `engine.mjs`, closing the root-sign loop in the actual shader.

**M4 — Wormhole math is triple-duplicated by hand, absent from `engine.mjs`, and untested by `wgsl_exec`; README overclaims**
`index.html:455-505` (`wh_r`/`wh_drdl`/`wh_rhs`/`traceWormhole`); `test/test.mjs:280-305`; `test/engine.mjs` (absent); `test/wgsl_exec.mjs:11-13`.
The Ellis/Dneg integrator exists as three hand-synced copies: the shipped WGSL, and a from-scratch JS reimplementation inlined in `test.mjs[11]` (lines 286-289). It is **not** in `engine.mjs` at all, and it lives *after* the `// ---- shading` slice marker so `wgsl_exec` never touches it. Test [11] also diverges from the shader: it treats `b` as a free input with `vr = -sqrt(1 - b²/r0²)` (line 292) whereas the shader derives `b = rcam·tmag`, `vr = dir·rhat` (`index.html:477-484`); it uses fixed `h=0.05` (line 295) vs the shader's adaptive step (`index.html:491`); and it asserts only `sign(finalL)` — a binary thread/reflect outcome a coarse wrong-but-monotone integrator would still pass. The README claims of `engine.mjs` as the single twin and "the shader that ships is the shader that's tested" do not hold for the wormhole.
**Fix:** move the wormhole math into `engine.mjs` as the single source, have `test.mjs` import it, and extend `wgsl_exec` to slice & execute the wormhole WGSL using the shader's own `b`/`vr` construction, asserting a quantitative bend angle — or document the wormhole's exemption from the testing guarantee in the README.

**M5 — No test exercises `POST_WGSL` or any host JS (uniform packing, bind groups, resize, UI)**
`package.json:8`; `index.html` `POST_WGSL` string and host `<script>`.
`wgsl_check.mjs:7` only matches `const WGSL = ...`, so a syntax error in `POST_WGSL` (bright-pass / Gaussian blur / Reinhard composite) is caught by no test. The host JS packing the 9-vec4 uniform — the exact layout `trace()` reads back via `u.p1.z`, `u.camUp.w`, `u.p0.w`, etc. (`index.html:376-385`, written at `836-845`) — is untested, so a packing/offset mismatch (a classic high-likelihood bug) ships undetected.
**Fix:** extend `wgsl_check.mjs` to also parse `POST_WGSL`; add a host-side test (or documented checklist) asserting the `u.set(...)` offsets match the `u.p0..p4`/`.w` reads in `trace()`.

**M6 — `requestDevice()` has no error handling and no `device.lost`/`uncapturederror` handler**
`index.html:595`.
`requestAdapter()` is carefully try/caught with a null-check (lines 592-594), but `const device = await adapter.requestDevice();` is awaited bare. A rejection (device limits, OOM, lost adapter) escapes the top-level async IIFE as an unhandled rejection — the user sees a blank canvas with no message, despite the friendly `fail()`/`#err` overlay existing for every other path. There is no `device.lost` handler and no `uncapturederror` listener, so mid-session device loss leaves the frame loop calling `writeBuffer`/`submit` on a dead device forever (`index.html:846-859`).
**Fix:** wrap `requestDevice` in try/catch → `fail(String(e))`; add `device.lost.then(info => fail('GPU device lost: ' + info.message))` and a `device.addEventListener('uncapturederror', …)`; guard the frame loop to stop submitting after loss.

### Low

**L1 — Bloom blur is anisotropic (~2× wider vertically than horizontally)** *(two findings from different dimensions merged)*
`index.html:665,680-681` (rebuildTargets); `POST_WGSL` fsBlur at `index.html:553-563`.
The separable blur composes over half-res buffers. The horizontal pass samples full-res `texHDR` with offset `spread/w`, the vertical pass samples half-res `texBlurA` with offset `spread/hh` (`hh = h>>1`). In common UV/screen units the vertical kernel reaches ~2× the horizontal extent — a visible directional smear on bright features. The two passes are meant to compose into one isotropic Gaussian (shared `spread`).
**Fix:** make both passes step in half-res texel units: `new Float32Array([spread / hw, 0, 0.7, 0])` for `blurHBuf` (compute `hw` alongside `hh`), leaving `blurVBuf` at `spread / hh`.

**L2 — Exposure slider does nothing on the default grid-sphere / starfield backgrounds**
`index.html:290-295` (starfield), `305-318` (celestial), vs `322-327`/`335-340`/`348-365`.
`skyImage`/`skyImage2`/`disk` multiply output by `u.p2.y` (exposure), but `celestial()` and `starfield()` do not. Moving the Exposure slider visibly changes the disk and image skies but leaves the *default* backgrounds unchanged — a user-observable inconsistency. (Minor: default exposure is 1.2, so image/disk paths bake in a 1.2× scale even at slider default.)
**Fix:** apply exposure uniformly — multiply `celestial()`/`starfield()` output by `u.p2.y`, or remove the per-path `* u.p2.y` everywhere and apply exposure once in the composite shader before Reinhard tonemap.

**L3 — `dSqrt` and dependent divisions can produce NaN at the exact origin / spin axis**
`index.html:116` (`dSqrt`), `165` (`lz = dDiv(kz, r)`), `199-200` (Schwarzschild `rho`/`half`).
`dSqrt(a)` returns gradient `a.d/(2*sqrt(a.v))`, which is Inf/NaN when `a.v == 0`. Reachable only when the radicand rounds to *exactly* 0 in f32 (the exact origin/axis-origin) — essentially measure-zero, and capture thresholds defend it for normal use. RK4 intermediate sub-samples are not loop-top-screened, so it remains a latent injection point.
**Fix:** floor the radius in the gradient, e.g. `let rs = max(r, 1e-8); return Dual(r, a.d/(2.0*rs));`, and guard `rho`/`r` with `max(.v, 1e-6)` before forming `half`/`lz`. Capture stays the primary defense; the metric simply shouldn't emit NaN.

**L4 — Enabling Animate jumps disk phase by (now − pageload); `t0` is never re-anchored**
`index.html:801,803-815`.
`t0` is captured once at script start and never reset. When Animate is toggled on, `time = (now - t0)/1000` is seconds since page load, not since enable, so the disk phase snaps to an arbitrary large value (and jumps again on each re-enable). `frozenTime` is only ever written *from* `time`, never used to re-anchor. Drives the disk swirl at `index.html:361`.
**Fix:** re-anchor on enable, e.g. `ui.anim.addEventListener('change', () => { t0 = performance.now() - frozenTime*1000; });`.

**L5 — Disk Keplerian speed uses isotropic radius as if it were the Schwarzschild areal radius**
`index.html:435` (caller), `rc` at `index.html:431`.
The engine runs in isotropic coordinates, so `rc = sqrt(x²+y²)` is the isotropic cylindrical radius, but `beta = sqrt(M/rc)` is the orbital coordinate speed expressed in the *areal* radius. Near the hole the two differ substantially, so the Doppler/beaming factor is evaluated at the wrong speed and hits the `0.95` clamp earlier than physical. A visualization-fidelity bias, not a stability bug.
**Fix:** convert before computing speed: `let rA = rc*pow(1.0 + M/(2.0*rc), 2.0); let beta = clamp(sqrt(M/rA), 0.0, 0.95);` — or document the coordinate-radius approximation as intentional.

**L6 — Disk rotation is hard-coded prograde, ignoring Kerr spin sign**
`index.html:434` (within `trace`).
`tang = (-y, x, 0)/rc` is always +φ regardless of spin `a` (`u.p0.w`). For Kerr the inner disk co-rotates with the hole, so `a → −a` should flip the blue/red Doppler side — but the disk beaming is unchanged, contradicting the lensing handedness the geodesic integrator correctly reproduces (`test.mjs[8]`). Manifests only for `metric==2 && a<0 && disk on && Doppler on`.
**Fix:** tie rotation sense to spin sign, e.g. `let spinSign = select(1.0, sign(a), metric == 2 && a != 0.0); let tang = spinSign * vec3(-pp.y, pp.x, 0.0)/rc;` — or document the always-prograde modeling choice.

**L7 — Gravitational lapse for disk redshift is sampled once at segment start, reused for all 8 sub-samples**
`index.html:428` and the k-loop at `429-441`.
`lapse = 1/sqrt(redshift)` is computed once from the metric at the step start `s.q` and applied to every disk sub-sample across the whole RK4 segment. Since the lapse feeds the temperature ramp and cubic beaming (`pow(dop,3.0)`), the position-dependent bias is amplified near the hole where the gradient is steepest. Accuracy bias in a visualizer, not a crash.
**Fix:** recompute the lapse per sub-sample from the local metric at `pp`, e.g. `let lapseK = 1.0/sqrt(max(-metricInverse(vec4(0.0, pp, ...)).c[0].v, 1e-3));` inside the k-loop (8 extra evals only inside the disk slab) — or document the single-sample approximation.

**L8 — `makePhoton` divides by `g^{00}` (A) with no guard**
`index.html:269`; `test/engine.mjs:210`.
`p0 = (-B + sqrt(disc)) / A` with no check that `A = g^{00} ≠ 0`. Safe only because every built-in metric keeps A strictly negative (Minkowski −1, isotropic Schwarzschild −(B/A)², Kerr-Schild −1−f). A future contributor following the README's "Add a new spacetime" recipe gets no warning that their `g^{00}` must be non-vanishing at the camera, or `makePhoton` silently yields Inf/NaN and the frame goes black. Latent, metric-agnostic-integrity hazard.
**Fix:** document the `g^{00} ≠ 0` precondition in the `makePhoton` comment and the README recipe; optionally fall back to the linear root when `abs(A) < eps` so a new metric degrades visibly rather than producing NaN.

### Nits

**N1 — `image2Loaded` is monotonic ("has ever been loaded"), never reset**
`index.html:646,760,845`; shader read at `index.html:344`.
Set to 1 on second-image load (line 760), never reset; `skyTexture2` is only re-pointed by the `imgfile2` handler, so once a far-side image is loaded there's no way back to the default warm grid without reloading the page. Latent trap for anyone adding a "clear image" control.
**Fix:** rename to `image2Present` for clarity; add a clear affordance that recreates the 1×1 default texture and resets the flag — or document the one-way-per-session behavior.

**N2 — Capture/disk thresholds are magic numbers duplicated across files**
`index.html:405,408,425,427,438,443` (`redshift>50.0`, `rPlus*1.01`, `3.0*H`, `/8.0`, `dens*4.0`, `trans<0.02`).
The `50.0` redshift threshold and `rPlus*1.01` are re-typed in `test.mjs:228,250`, `shadow_probe.mjs:27`, and the README — four places that must change in sync or the JS validation stops mirroring the shader. The subsample count `8` appears three times (loop bound, divisor in `ds`, weight divisor): edit one, miss another, and the emission integral silently mis-scales. (The constants *are* explained in adjacent comments, so "unexplained" overstates it; this is purely DRY/single-source.)
**Fix:** hoist into named consts at the top of `trace()` (`const DISK_SUBSAMPLES = 8u;` used for both bound and divisor); export the shared redshift threshold as a single constant the tests import.

**N3 — `shadow_probe.mjs` has zero assertions, is not in `npm test`, and its "Mirrors trace() exactly" claim is unverified**
`test/shadow_probe.mjs:1-74`; `package.json:8`.
It reimplements `trace()` and prints an ASCII map but has no `check()`/`assert`/`process.exit`, and isn't in the test script. It has already drifted: it derives `escapeR`/`hmax` locally (lines 5-6) whereas the shader reads them from `u` (`index.html:381-382`). No bug can pass through CI because of this (it's a diagnostic, not a test) — the issue is the misleading "Mirrors … exactly" wording.
**Fix:** either promote it to a real asserted test added to `npm test`, or relabel it explicitly as a non-test diagnostic.

## 3. Cross-Cutting Themes

The dominant theme is a **test-coverage/documentation-honesty gap**: the suite rigorously validates the geodesic *engine* (metric, `rhs`, `rk4`, `makePhoton` for diagonal metrics) but never CPU-executes the shipped `trace()`, `disk()`, `traceWormhole()`, backgrounds, post-processing, or host uniform packing — and the README's "the shader that ships is the shader that's tested" reads more broadly than the slice actually covers (M1, M2, M3, M4, M5, N3). A second recurring pattern is **hand-synced duplication** of constants and whole subsystems across the WGSL, `engine.mjs`, and `test.mjs` (the wormhole math has three independent copies; the `50.0`/`rPlus*1.01` thresholds live in four places), each a silent-divergence risk (M4, N2). A third, milder theme is **defensive-hardening / metric-agnostic-integrity gaps** that are latent today but undocumented for future contributors (L3, L8, M6). The shader's **post-processing and UX layer** carries the bulk of the cosmetic issues (L1, L2, L4, N1), all isolated from the physics core.

## 4. What's Solid

- **The GR geodesic core is correct.** No bug was found in `metricInverse`, `rhs`, `rk4`, or the `makePhoton` future-directed null root. These are genuinely CPU-executed against the shipped WGSL (for Flat/Schwarzschild) and validated against analytic GR.
- **Spin sign is correctly handled in the integrator** — `test.mjs[8]` confirms prograde vs retrograde equatorial deflection differ with `a=0` between them. (Only the *disk visual* ignores it — L6.)
- **The Kerr metric tensor itself is verified** (`wgsl_exec` section B), and the three independent wormhole copies currently agree — the M-series findings are about *test reachability and drift risk*, not present numerical error.
- **Existing error-handling and capture machinery is sound** where present: the `requestAdapter` path, the `#err`/`fail()` overlay, and the redshift/Kerr-radius capture thresholds all work correctly; the gaps (M6, L3) are about extending the same patterns to uncovered paths.
- The disk temperature ramp is internally consistent (isotropic radius used consistently for bounds; only `beta` mixes conventions — L5).

**Bottom line:** this is a short, high-quality findings list for a project of this complexity — **zero critical, one high (a clean perf win), six medium (all test-integrity/robustness, no live correctness bugs), and the rest cosmetic or latent.** The physics that the renderer actually computes is correct; the real work to do is closing the gap between what the tests *claim* to cover and what they *execute*.
