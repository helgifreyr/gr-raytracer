// engine.mjs — reference implementation of the metric-agnostic geodesic engine.
//
// This is the double-precision JS twin of the WGSL engine in index.html. The two
// are kept structurally identical so that the Node validation suite (which can run
// here but not in a headless browser) transitively validates the shader math.
//
// Design (ported from H:\study\raylib_raytrace_blackhole_old):
//   1. Forward-mode autodiff via dual numbers — but widened to carry the full
//      4-gradient (d/dt, d/dx, d/dy, d/dz) so ONE metric evaluation yields every
//      partial derivative the Hamiltonian needs. (The C did one eval per partial.)
//   2. The spacetime is the ONLY physics-specific code: `metricInverse(q)` returns
//      the contravariant metric g^{ab}(q) as duals. Nothing downstream knows which
//      metric it is — that is what "metric-agnostic" means.
//   3. Hamiltonian geodesics: H = 1/2 g^{ab}(q) p_a p_b, integrated with RK4.
//
// Units: G = c = 1, mass M in geometric units. Coordinates q = [t, x, y, z].

// ----------------------------------------------------------------------------
// Dual numbers: value `v` plus gradient `d` = [∂/∂t, ∂/∂x, ∂/∂y, ∂/∂z].
// ----------------------------------------------------------------------------

export const D = {
  konst: (v) => ({ v, d: [0, 0, 0, 0] }),                 // constant
  varAt: (v, i) => { const d = [0, 0, 0, 0]; d[i] = 1; return { v, d }; }, // seed coord i
  add: (a, b) => ({ v: a.v + b.v, d: a.d.map((x, i) => x + b.d[i]) }),
  sub: (a, b) => ({ v: a.v - b.v, d: a.d.map((x, i) => x - b.d[i]) }),
  mul: (a, b) => ({ v: a.v * b.v, d: a.d.map((x, i) => x * b.v + a.v * b.d[i]) }),
  div: (a, b) => ({
    v: a.v / b.v,
    d: a.d.map((x, i) => (x * b.v - a.v * b.d[i]) / (b.v * b.v)),
  }),
  addScalar: (a, s) => ({ v: a.v + s, d: a.d.slice() }),
  mulScalar: (a, s) => ({ v: a.v * s, d: a.d.map((x) => x * s) }),
  scalarDiv: (s, b) => ({                                  // s / b
    v: s / b.v,
    d: b.d.map((x) => (-s * x) / (b.v * b.v)),
  }),
  sqrt: (a) => {
    const r = Math.sqrt(a.v);
    return { v: r, d: a.d.map((x) => x / (2 * r)) };
  },
};

// ----------------------------------------------------------------------------
// Metrics. Each returns the symmetric contravariant metric g^{ab} as a 4x4 of
// duals. To add a spacetime, write one of these — and nothing else changes.
// ----------------------------------------------------------------------------

export const METRIC = { FLAT: 0, SCHWARZSCHILD: 1, KERR: 2 };

function zeros4x4() {
  return [0, 1, 2, 3].map(() => [0, 1, 2, 3].map(() => D.konst(0)));
}

// Minkowski: signature (-,+,+,+). Constant ⇒ zero gradients ⇒ straight lines.
function metricFlat(_q) {
  const g = zeros4x4();
  g[0][0] = D.konst(-1);
  g[1][1] = D.konst(1);
  g[2][2] = D.konst(1);
  g[3][3] = D.konst(1);
  return g;
}

// Schwarzschild in isotropic Cartesian coordinates (single hole at origin).
//   A = 1 - M/2ρ,  B = 1 + M/2ρ,  ρ = sqrt(x²+y²+z²)
//   g_tt = -(A/B)²,  g_ii = B⁴   (covariant)
//   g^tt = -(B/A)²,  g^ii = B^-4 (contravariant — what we return)
// Horizon sits at ρ = M/2; the spatial part stays finite there (good for f32),
// only g^tt diverges as the lapse → 0, and we terminate rays before reaching it.
function metricSchwarzschild(q, M) {
  const x = D.varAt(q[1], 1);
  const y = D.varAt(q[2], 2);
  const z = D.varAt(q[3], 3);
  const rho = D.sqrt(D.add(D.add(D.mul(x, x), D.mul(y, y)), D.mul(z, z)));
  const half = D.scalarDiv(M / 2, rho);        // M/(2ρ)
  const A = D.sub(D.konst(1), half);           // 1 - M/2ρ
  const B = D.add(D.konst(1), half);           // 1 + M/2ρ
  const B2 = D.mul(B, B);
  const B4 = D.mul(B2, B2);
  const ratio = D.div(B, A);                   // B/A
  const gtt = D.mulScalar(D.mul(ratio, ratio), -1); // -(B/A)²
  const gii = D.scalarDiv(1, B4);              // B^-4

  const g = zeros4x4();
  g[0][0] = gtt;
  g[1][1] = gii;
  g[2][2] = gii;
  g[3][3] = gii;
  return g;
}

// Kerr in Kerr-Schild Cartesian coordinates (spin a about the z-axis).
//   contravariant: g^{μν} = η^{μν} - f lᵘlᵛ  (with l η-null; index 0 of l raised → -1)
//   f = 2 M r³ / (r⁴ + a² z²),  l = (1, (rx+ay)/(r²+a²), (ry-ax)/(r²+a²), z/r)
//   r = Kerr radial coord: r² = ((ρ²-a²) + sqrt((ρ²-a²)² + 4a²z²)) / 2,  ρ²=x²+y²+z²
// Horizon-penetrating (no coordinate singularity at r₊), so f32-friendly. a=0 → Schwarzschild.
function metricKerr(q, M, a) {
  const x = D.varAt(q[1], 1), y = D.varAt(q[2], 2), z = D.varAt(q[3], 3);
  const a2 = a * a;
  const x2 = D.mul(x, x), y2 = D.mul(y, y), z2 = D.mul(z, z);
  const rho2 = D.add(D.add(x2, y2), z2);
  const w = D.sub(rho2, D.konst(a2));
  const Dd = D.sqrt(D.add(D.mul(w, w), D.mulScalar(z2, 4 * a2)));   // sqrt(w² + 4a²z²)
  const r2 = D.mulScalar(D.add(w, Dd), 0.5);
  const r = D.sqrt(r2);
  const r2a = D.add(r2, D.konst(a2));                              // r² + a²
  const lx = D.div(D.add(D.mul(r, x), D.mulScalar(y, a)), r2a);
  const ly = D.div(D.sub(D.mul(r, y), D.mulScalar(x, a)), r2a);
  const lz = D.div(z, r);
  const r3 = D.mul(r2, r), r4 = D.mul(r2, r2);
  const f = D.div(D.mulScalar(r3, 2 * M), D.add(r4, D.mulScalar(z2, a2))); // 2Mr³/(r⁴+a²z²)

  const g = zeros4x4();
  g[0][0] = D.sub(D.konst(-1), f);                                 // -1 - f
  const g0x = D.mul(f, lx), g0y = D.mul(f, ly), g0z = D.mul(f, lz);
  g[0][1] = g0x; g[1][0] = g0x;
  g[0][2] = g0y; g[2][0] = g0y;
  g[0][3] = g0z; g[3][0] = g0z;
  g[1][1] = D.sub(D.konst(1), D.mul(f, D.mul(lx, lx)));
  g[2][2] = D.sub(D.konst(1), D.mul(f, D.mul(ly, ly)));
  g[3][3] = D.sub(D.konst(1), D.mul(f, D.mul(lz, lz)));
  const gxy = D.mulScalar(D.mul(f, D.mul(lx, ly)), -1);
  const gxz = D.mulScalar(D.mul(f, D.mul(lx, lz)), -1);
  const gyz = D.mulScalar(D.mul(f, D.mul(ly, lz)), -1);
  g[1][2] = gxy; g[2][1] = gxy;
  g[1][3] = gxz; g[3][1] = gxz;
  g[2][3] = gyz; g[3][2] = gyz;
  return g;
}

export function metricInverse(q, metric, M, a = 0) {
  if (metric === METRIC.FLAT) return metricFlat(q);
  if (metric === METRIC.KERR) return metricKerr(q, M, a);
  return metricSchwarzschild(q, M);
}

// ----------------------------------------------------------------------------
// Hamiltonian geodesic equations (metric-agnostic).
//   dq^i/dλ = g^{ij} p_j
//   dp_i/dλ = -1/2 (∂_i g^{ab}) p_a p_b
// w = [q0..q3, p0..p3]; returns dw/dλ (length 8).
// ----------------------------------------------------------------------------

export function rhs(w, metric, M, a = 0) {
  const q = w.slice(0, 4);
  const p = w.slice(4, 8);
  const g = metricInverse(q, metric, M, a); // 4x4 of duals: .v = g^{ab}, .d = ∂g^{ab}

  const dq = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    let s = 0;
    for (let j = 0; j < 4; j++) s += g[i][j].v * p[j];
    dq[i] = s;
  }

  const dp = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    let s = 0;
    // Full double sum over (a,b): symmetric off-diagonals contribute twice,
    // which is exactly the 2*p_a*p_b convention from the C reference.
    for (let a = 0; a < 4; a++)
      for (let b = 0; b < 4; b++) s += g[a][b].d[i] * p[a] * p[b];
    dp[i] = -0.5 * s;
  }

  return [...dq, ...dp];
}

export function rk4Step(w, h, metric, M, a = 0) {
  const addv = (u, b, s) => u.map((x, i) => x + s * b[i]);
  const k1 = rhs(w, metric, M, a);
  const k2 = rhs(addv(w, k1, h / 2), metric, M, a);
  const k3 = rhs(addv(w, k2, h / 2), metric, M, a);
  const k4 = rhs(addv(w, k3, h), metric, M, a);
  return w.map((x, i) => x + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// Hamiltonian H = ½ g^{ab} p_a p_b. Should be ≈0 for a photon and conserved along λ.
export function hamiltonian(w, metric, M, a = 0) {
  const q = w.slice(0, 4), p = w.slice(4, 8);
  const g = metricInverse(q, metric, M, a);
  let H = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) H += g[i][j].v * p[i] * p[j];
  return 0.5 * H;
}

// ----------------------------------------------------------------------------
// Initial conditions for a photon. Given a start position and a spatial direction
// `dir`, build a null momentum (H = 0) — general form valid for non-diagonal metrics
// (Kerr). We set the covariant spatial momentum p_i = dir_i; at the (far, weak-field)
// camera this gives coordinate velocity ≈ dir. Then solve the null condition
//   A p_t² + 2 B p_t + C = 0,  A=g^{tt}, B=g^{ti}p_i, C=g^{ij}p_i p_j
// taking the "+" (future-directed) root. In ingoing Kerr–Schild coordinates — adapted to
// the FUTURE horizon — this is the branch whose geodesic actually falls inward; the other
// root sends would-be-captured rays back out (an empty shadow). For diagonal metrics B=0
// and the spatial geodesic is identical for either branch, so this is harmless there.
// ----------------------------------------------------------------------------

export function makePhoton(pos, dir, metric, M, a = 0) {
  const q = [0, pos[0], pos[1], pos[2]];
  const g = metricInverse(q, metric, M, a);
  const G = (i, j) => g[i][j].v;
  const p = [0, dir[0], dir[1], dir[2]];
  const A = G(0, 0);
  const B = G(0, 1) * p[1] + G(0, 2) * p[2] + G(0, 3) * p[3];
  let C = 0;
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) C += G(i, j) * p[i] * p[j];
  const disc = Math.max(B * B - A * C, 0);
  p[0] = (-B + Math.sqrt(disc)) / A; // future-directed root (falls inward in Kerr-Schild)
  return [...q, ...p];
}

// Coordinate-space velocity dx^i/dλ (handy for measuring asymptotic directions).
export function coordVelocity(w, metric, M, a = 0) {
  const d = rhs(w, metric, M, a);
  return [d[1], d[2], d[3]];
}

// ----------------------------------------------------------------------------
// Ellis/Dneg wormhole (ultrastatic; no horizon). Single source of truth for the
// wormhole, mirroring traceWormhole() in index.html: r(ℓ) shape, the smooth 2nd-order
// radial equation ℓ̈ = (b²/r³)(dr/dℓ), and the camera b/vr construction. Spherically
// symmetric ⇒ each ray is planar, integrated as state (ℓ, ℓ̇, φ). Params p = {rho, a, Mw}.
// ----------------------------------------------------------------------------
export const Wormhole = {
  r(l, p) {
    const al = Math.abs(l);
    if (al <= p.a) return p.rho;
    const x = 2 * (al - p.a) / (Math.PI * p.Mw);
    return p.rho + p.Mw * (x * Math.atan(x) - 0.5 * Math.log(1 + x * x));
  },
  drdl(l, p) {
    const al = Math.abs(l);
    if (al <= p.a) return 0;
    return Math.sign(l) * (2 / Math.PI) * Math.atan(2 * (al - p.a) / (Math.PI * p.Mw));
  },
  rhs(st, b, p) { const r = this.r(st[0], p); return [st[1], (b * b) / (r * r * r) * this.drdl(st[0], p), b / (r * r)]; },
  // Trace a camera ray exactly as the shader does (b = rcam·tmag, ℓ̇₀ = dir·rhat).
  // Returns the final ℓ — sign tells which universe the ray exits (ℓ<0 = the other side).
  traceFinalL(camPos, dir, p, opt = {}) {
    const { maxSteps = 40000, stepScale = 0.08, hmin = 0.01, hmax = 2.0 } = opt;
    const lcam = Math.hypot(...camPos);
    const rhat = camPos.map((c) => c / lcam);
    const vr = dir[0] * rhat[0] + dir[1] * rhat[1] + dir[2] * rhat[2];
    const tmag = Math.hypot(...dir.map((c, i) => c - vr * rhat[i]));
    const b = this.r(lcam, p) * tmag;
    let st = [lcam, vr, 0];
    const lfar = Math.max(lcam * 1.4, 40);
    for (let i = 0; i < maxSteps; i++) {
      if (Math.abs(st[0]) > lfar) break;
      const h = Math.min(Math.max(stepScale * (Math.abs(st[0]) + this.r(st[0], p)), hmin), hmax);
      const k1 = this.rhs(st, b, p), k2 = this.rhs(st.map((x, j) => x + 0.5 * h * k1[j]), b, p),
        k3 = this.rhs(st.map((x, j) => x + 0.5 * h * k2[j]), b, p), k4 = this.rhs(st.map((x, j) => x + h * k3[j]), b, p);
      st = st.map((x, j) => x + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    }
    return st[0];
  },
};
