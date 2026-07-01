struct U {
  camPos   : vec4<f32>,   // xyz position, w = tan(fov/2)
  camRight : vec4<f32>,   // xyz basis,    w = aspect
  camUp    : vec4<f32>,   // xyz basis,    w = mass M
  camFwd   : vec4<f32>,   // xyz basis,    w = time
  p0       : vec4<f32>,   // rIn, rOut, escapeR, spin
  p1       : vec4<f32>,   // maxSteps, stepScale, metric, diskOn
  p2       : vec4<f32>,   // bgMode, exposure, hmin, hmax
  p3       : vec4<f32>,   // dopplerOn, diskBrightness, diskHalfThickness, _
  p4       : vec4<f32>,   // wormhole: throatRadius ρ, throatHalfLen a, lensing M, image2Loaded
  p5       : vec4<f32>,   // disk: noiseScale, swirlSpeed, _, _
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var skyTex : texture_2d<f32>;
@group(0) @binding(2) var skySamp : sampler;
@group(0) @binding(3) var skyTex2 : texture_2d<f32>;   // wormhole: "other universe" sky

// ---- dual numbers: value v, gradient d = (d/dt, d/dx, d/dy, d/dz) -----------
struct Dual { v : f32, d : vec4<f32> };
fn dC(v : f32) -> Dual { return Dual(v, vec4<f32>(0.0)); }
fn dAdd(a : Dual, b : Dual) -> Dual { return Dual(a.v + b.v, a.d + b.d); }
fn dSub(a : Dual, b : Dual) -> Dual { return Dual(a.v - b.v, a.d - b.d); }
fn dMul(a : Dual, b : Dual) -> Dual { return Dual(a.v * b.v, a.d * b.v + a.v * b.d); }
fn dDiv(a : Dual, b : Dual) -> Dual { return Dual(a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)); }
fn dMulS(d : Dual, s : f32) -> Dual { return Dual(d.v * s, d.d * s); }
fn dSDiv(s : f32, b : Dual) -> Dual { return Dual(s / b.v, (-s * b.d) / (b.v * b.v)); }
fn dSqrt(a : Dual) -> Dual { let r = sqrt(max(a.v, 1e-30)); return Dual(r, a.d / (2.0 * r)); }  // floor avoids NaN/Inf at the exact origin

// ---- the spacetime: contravariant metric g^{ab} as a 4x4 (row-major) of duals
struct Metric { c : array<Dual, 16> };

fn metricInverse(q : vec4<f32>, metric : i32, M : f32, a : f32) -> Metric {
  var m : Metric;
  for (var i = 0; i < 16; i = i + 1) { m.c[i] = dC(0.0); }

  if (metric == 0) {                       // Minkowski (-,+,+,+)
    m.c[0]  = dC(-1.0);
    m.c[5]  = dC(1.0);
    m.c[10] = dC(1.0);
    m.c[15] = dC(1.0);
    return m;
  }

  if (metric == 2) {
    // Kerr in Kerr-Schild Cartesian coords (spin a about z). g^{μν} = η^{μν} - f lᵘlᵛ.
    // Horizon-penetrating (no coord-singularity at r₊) → f32-friendly. a=0 → Schwarzschild.
    // Written in flat SSA form (one binary op per let): equivalent on the GPU, and
    // avoids a nested-call evaluation bug in the CPU interpreter used for testing.
    let kx = Dual(q.y, vec4<f32>(0.0, 1.0, 0.0, 0.0));
    let ky = Dual(q.z, vec4<f32>(0.0, 0.0, 1.0, 0.0));
    let kz = Dual(q.w, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let a2 = a * a;
    let a2d = dC(a2);
    let kx2 = dMul(kx, kx);
    let ky2 = dMul(ky, ky);
    let kz2 = dMul(kz, kz);
    let sxy = dAdd(kx2, ky2);
    let rho2 = dAdd(sxy, kz2);
    let w = dSub(rho2, a2d);
    let ww = dMul(w, w);
    let z2t = dMulS(kz2, 4.0 * a2);
    let disc = dAdd(ww, z2t);
    let root = dSqrt(disc);                                   // sqrt(w² + 4a²z²)
    let wD = dAdd(w, root);
    let r2 = dMulS(wD, 0.5);
    let r = dSqrt(r2);
    let r2a = dAdd(r2, a2d);                                  // r² + a²
    let rkx = dMul(r, kx);
    let aky = dMulS(ky, a);
    let lxn = dAdd(rkx, aky);
    let lx = dDiv(lxn, r2a);                                  // (rx + ay)/(r²+a²)
    let rky = dMul(r, ky);
    let akx = dMulS(kx, a);
    let lyn = dSub(rky, akx);
    let ly = dDiv(lyn, r2a);                                  // (ry - ax)/(r²+a²)
    let lz = dDiv(kz, r);                                     // z/r
    let r3 = dMul(r2, r);
    let r4 = dMul(r2, r2);
    let azz = dMulS(kz2, a2);
    let denom = dAdd(r4, azz);
    let r3M = dMulS(r3, 2.0 * M);
    let f = dDiv(r3M, denom);                                 // 2Mr³/(r⁴+a²z²)
    let negone = dC(-1.0);
    let one = dC(1.0);
    m.c[0] = dSub(negone, f);                                 // g^00 = -1 - f
    let g0x = dMul(f, lx); let g0y = dMul(f, ly); let g0z = dMul(f, lz);
    m.c[1] = g0x; m.c[4] = g0x;
    m.c[2] = g0y; m.c[8] = g0y;
    m.c[3] = g0z; m.c[12] = g0z;
    let lxlx = dMul(lx, lx); let flxlx = dMul(f, lxlx);
    let lyly = dMul(ly, ly); let flyly = dMul(f, lyly);
    let lzlz = dMul(lz, lz); let flzlz = dMul(f, lzlz);
    m.c[5]  = dSub(one, flxlx);
    m.c[10] = dSub(one, flyly);
    m.c[15] = dSub(one, flzlz);
    let lxly = dMul(lx, ly); let flxly = dMul(f, lxly); let gxy = dMulS(flxly, -1.0);
    let lxlz = dMul(lx, lz); let flxlz = dMul(f, lxlz); let gxz = dMulS(flxlz, -1.0);
    let lylz = dMul(ly, lz); let flylz = dMul(f, lylz); let gyz = dMulS(flylz, -1.0);
    m.c[6] = gxy; m.c[9] = gxy;
    m.c[7] = gxz; m.c[13] = gxz;
    m.c[11] = gyz; m.c[14] = gyz;
    return m;
  }

  // Schwarzschild, isotropic Cartesian coordinates (single hole at origin).
  //   A = 1 - M/2ρ,  B = 1 + M/2ρ;  g^tt = -(B/A)²,  g^ii = B^-4
  let x = Dual(q.y, vec4<f32>(0.0, 1.0, 0.0, 0.0));
  let y = Dual(q.z, vec4<f32>(0.0, 0.0, 1.0, 0.0));
  let z = Dual(q.w, vec4<f32>(0.0, 0.0, 0.0, 1.0));
  let rho  = dSqrt(dAdd(dAdd(dMul(x, x), dMul(y, y)), dMul(z, z)));
  let half = dSDiv(M * 0.5, rho);
  let A = dSub(dC(1.0), half);
  let B = dAdd(dC(1.0), half);
  let B2 = dMul(B, B);
  let B4 = dMul(B2, B2);
  let ratio = dDiv(B, A);
  let gtt = dMulS(dMul(ratio, ratio), -1.0);
  let gii = dSDiv(1.0, B4);
  m.c[0]  = gtt;
  m.c[5]  = gii;
  m.c[10] = gii;
  m.c[15] = gii;
  return m;
}

// ---- Hamiltonian geodesic RHS (metric-agnostic) -----------------------------
//   dq^i/dλ = g^{ij} p_j ;  dp_i/dλ = -1/2 (∂_i g^{ab}) p_a p_b
struct State { q : vec4<f32>, p : vec4<f32> };

// RHS from a PRE-COMPUTED metric m at s.q. Splitting this out lets the trace loop reuse
// the metric it already evaluated for the capture test (one eval/step instead of two), and
// lets the test suite exercise this exact code for Kerr without the CPU interpreter's
// variable-metric dispatch bug (it feeds m from a literal-metric metricInverse call).
fn rhsM(s : State, m : Metric) -> State {
  var mm = m;
  var pv = s.p;
  var dq = vec4<f32>(0.0);
  for (var i = 0; i < 4; i = i + 1) {
    var acc = 0.0;
    for (var j = 0; j < 4; j = j + 1) { acc = acc + mm.c[i * 4 + j].v * pv[j]; }
    dq[i] = acc;
  }
  var dp = vec4<f32>(0.0);
  for (var i = 0; i < 4; i = i + 1) {
    var acc = 0.0;
    for (var a = 0; a < 4; a = a + 1) {
      for (var b = 0; b < 4; b = b + 1) {
        acc = acc + mm.c[a * 4 + b].d[i] * pv[a] * pv[b];
      }
    }
    dp[i] = -0.5 * acc;
  }
  return State(dq, dp);
}
fn rhs(s : State, metric : i32, M : f32, a : f32) -> State {
  return rhsM(s, metricInverse(s.q, metric, M, a));
}

// rk4 with the first-stage metric (m0 = metricInverse at s.q) passed in, so the caller that
// already has it doesn't pay for a second evaluation.
fn rk4(s : State, h : f32, metric : i32, M : f32, a : f32, m0 : Metric) -> State {
  let k1 = rhsM(s, m0);
  let s2 = State(s.q + k1.q * (h * 0.5), s.p + k1.p * (h * 0.5));
  let k2 = rhs(s2, metric, M, a);
  let s3 = State(s.q + k2.q * (h * 0.5), s.p + k2.p * (h * 0.5));
  let k3 = rhs(s3, metric, M, a);
  let s4 = State(s.q + k3.q * h, s.p + k3.p * h);
  let k4 = rhs(s4, metric, M, a);
  return State(
    s.q + (h / 6.0) * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q),
    s.p + (h / 6.0) * (k1.p + 2.0 * k2.p + 2.0 * k3.p + k4.p),
  );
}

// Null initial condition from a PRE-COMPUTED metric m at the camera. Set covariant spatial
// momentum p_i = dir_i (≈ coordinate direction in the far, weak field), then solve
// A p_t² + 2B p_t + C = 0 and take the "+" (future-directed) root — in ingoing Kerr-Schild
// it is the branch that falls inward (the other gives an empty shadow). Harmless for diagonal
// metrics. PRECONDITION: A = g^{00} ≠ 0 at the camera (true for every built-in metric); a new
// metric whose g^{00} vanishes there must guard this. The select() degrades to a finite value
// instead of NaN if that ever happens.
fn makePhotonM(pos : vec3<f32>, dir : vec3<f32>, m : Metric) -> State {
  let q = vec4<f32>(0.0, pos.x, pos.y, pos.z);
  let px = dir.x;  let py = dir.y;  let pz = dir.z;
  let A = m.c[0].v;
  let B = m.c[1].v * px + m.c[2].v * py + m.c[3].v * pz;             // g^{0i} p_i
  let C = m.c[5].v * px * px + m.c[10].v * py * py + m.c[15].v * pz * pz
        + 2.0 * (m.c[6].v * px * py + m.c[7].v * px * pz + m.c[11].v * py * pz);
  let disc = max(B * B - A * C, 0.0);
  let Aq = select(A, -1e-6, abs(A) < 1e-9);          // guard g^{00}=0 (see precondition)
  let p0 = (-B + sqrt(disc)) / Aq;                   // future-directed root (falls inward)
  return State(q, vec4<f32>(p0, px, py, pz));
}
fn makePhoton(pos : vec3<f32>, dir : vec3<f32>, metric : i32, M : f32, a : f32) -> State {
  return makePhotonM(pos, dir, metricInverse(vec4<f32>(0.0, pos.x, pos.y, pos.z), metric, M, a));
}

// ---- shading ---------------------------------------------------------------
fn hash31(p3 : vec3<f32>) -> f32 {
  var p = fract(p3 * 0.1031);
  p = p + dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
fn starLayer(dir : vec3<f32>, scale : f32, thresh : f32) -> f32 {
  let p = dir * scale;
  let i = floor(p);
  let f = p - i;
  let r = hash31(i);
  if (r < thresh) { return 0.0; }
  let c = vec3<f32>(hash31(i + 1.3), hash31(i + 2.7), hash31(i + 4.1));
  let d = length(f - c);
  let bright = (r - thresh) / (1.0 - thresh);
  return bright * smoothstep(0.09, 0.0, d);
}
fn starfield(dir : vec3<f32>) -> vec3<f32> {
  var col = mix(vec3<f32>(0.015, 0.02, 0.05), vec3<f32>(0.02, 0.01, 0.03), dir.y * 0.5 + 0.5);
  let s = starLayer(dir, 180.0, 0.975) + starLayer(dir, 320.0, 0.985) * 0.8
        + starLayer(dir, 90.0, 0.99) * 1.3;
  return (col + vec3<f32>(0.95, 0.97, 1.0) * s) * u.p2.y;   // honour exposure, like disk/image
}

const PI : f32 = 3.14159265359;
fn gridLine(c : f32, spacing : f32, width : f32) -> f32 {
  let d = abs(fract(c / spacing + 0.5) - 0.5) * spacing;   // angular distance to nearest line
  return smoothstep(width, width * 0.4, d);
}
// Celestial-sphere test pattern (cf. Cunha, Herdeiro, Radu & Runarsson 2015, arXiv:1509.00021):
// four coloured azimuth quadrants, dimmer southern hemisphere, lat/long grid.
// Lensing is read directly off how the quadrant edges and grid lines warp.
fn celestial(dir : vec3<f32>) -> vec3<f32> {
  let theta = acos(clamp(dir.z, -1.0, 1.0));   // polar angle from +z (disk normal)
  let phi = atan2(dir.y, dir.x);               // azimuth -pi..pi
  let quad = i32(floor((phi + PI) / (PI * 0.5))) % 4;
  var base = vec3<f32>(0.18, 0.42, 0.85);                  // quadrant 0: blue
  if (quad == 1) { base = vec3<f32>(0.92, 0.62, 0.16); }   // amber
  else if (quad == 2) { base = vec3<f32>(0.22, 0.72, 0.42); } // green
  else if (quad == 3) { base = vec3<f32>(0.85, 0.28, 0.55); } // magenta
  if (dir.z < 0.0) { base = base * 0.4; }       // southern hemisphere dimmer
  let latDeg = theta * 180.0 / PI;
  let lonDeg = (phi + PI) * 180.0 / PI;
  let g = max(gridLine(latDeg, 15.0, 0.6), gridLine(lonDeg, 15.0, 0.6));
  return mix(base, vec3<f32>(1.0), g * 0.85) * u.p2.y;   // honour exposure, like disk/image
}
// User-loaded sky image, equirectangular (lat/long) mapping. textureSampleLevel (not
// textureSample) because background() is called from non-uniform control flow, and the
// per-pixel uv has wild derivatives under lensing — explicit LOD 0 avoids artifacts.
fn skyImage(dir : vec3<f32>) -> vec3<f32> {
  let phi = atan2(dir.y, dir.x);                 // longitude -pi..pi
  let theta = acos(clamp(dir.z, -1.0, 1.0));     // colatitude 0..pi (pole = spin/disk axis)
  let uv = vec2<f32>(phi / (2.0 * PI) + 0.5, theta / PI);
  return textureSampleLevel(skyTex, skySamp, uv, 0.0).rgb * u.p2.y;  // * exposure
}
fn background(dir : vec3<f32>) -> vec3<f32> {
  let mode = i32(u.p2.x + 0.5);                 // 0 none, 1 starfield, 2 grid, 3 image
  if (mode == 0) { return vec3<f32>(0.0); }
  if (mode == 2) { return celestial(dir); }
  if (mode == 3) { return skyImage(dir); }
  return starfield(dir);
}
fn skyImage2(dir : vec3<f32>) -> vec3<f32> {
  let phi = atan2(dir.y, dir.x);
  let theta = acos(clamp(dir.z, -1.0, 1.0));
  let uv = vec2<f32>(phi / (2.0 * PI) + 0.5, theta / PI);
  return textureSampleLevel(skyTex2, skySamp, uv, 0.0).rgb * u.p2.y;
}
// The wormhole's far side. If a second image is loaded, use it; otherwise a warm-tinted
// grid sphere so the two universes are visually distinct out of the box.
fn otherBackground(dir : vec3<f32>) -> vec3<f32> {
  if (u.p4.w > 0.5) { return skyImage2(dir); }
  let c = celestial(dir);
  return c.zyx * vec3<f32>(1.15, 0.95, 0.7);
}
// ---- value noise + FBM, for turbulent accretion gas ------------------------
// Technique adapted from the cuneus black-hole shader (Enes Altun, MIT-licensed,
// github.com/altunenes/cuneus): multifractal cloud noise sampled in log-spiral
// coordinates gives the disk filamentary, swirling structure instead of a smooth
// ramp. This is shading only — the geodesics/metrics/capture are untouched.
fn vnoise(p : vec3<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let w = f * f * (3.0 - 2.0 * f);                    // smoothstep weights (C¹ continuity)
  let c000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
  let c100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let c010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let c110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let c001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let c101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let c011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let c111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
  let x00 = mix(c000, c100, w.x);
  let x10 = mix(c010, c110, w.x);
  let x01 = mix(c001, c101, w.x);
  let x11 = mix(c011, c111, w.x);
  let y0 = mix(x00, x10, w.y);
  let y1 = mix(x01, x11, w.y);
  return mix(y0, y1, w.z);
}
fn fbm(p : vec3<f32>) -> f32 {
  var v = 0.0;  var amp = 0.5;  var pp = p;
  for (var i = 0; i < 4; i = i + 1) {
    v = v + amp * vnoise(pp);
    pp = pp * 2.02;
    amp = amp * 0.5;
  }
  return v;
}
// Ridged noise: sharp crests where the value noise crosses ½ (squared to thin them).
// Summed over octaves this builds the fine, filamentary wisps of turbulent gas — round
// FBM blobs give smoke, ridges give threads.
fn ridge(p : vec3<f32>) -> f32 {
  let n = 1.0 - abs(2.0 * vnoise(p) - 1.0);
  return n * n;
}
fn fbmRidged(p : vec3<f32>) -> f32 {
  var v = 0.0;  var amp = 0.5;  var pp = p;
  for (var i = 0; i < 5; i = i + 1) {
    v = v + amp * ridge(pp);
    pp = pp * 2.13;
    amp = amp * 0.5;
  }
  return v;                                            // ~0 .. 0.97
}
// Turbulent gas density at a point in the disk plane. Combines a vertical Gaussian
// slab, a radial window, and multifractal cloud noise twisted into log-spiral arms.
fn diskDensity(pos : vec3<f32>, rc : f32, H : f32) -> f32 {
  let tnorm = clamp((rc - u.p0.x) / (u.p0.y - u.p0.x), 0.0, 1.0);
  let radial = smoothstep(0.0, 0.1, tnorm) * smoothstep(1.0, 0.72, tnorm);
  let vfall = exp(-(pos.z * pos.z) / (H * H));
  let base = radial * vfall;
  if (base < 0.003) { return 0.0; }   // cheap analytic gate: skip the ~17 noise evals on empty/grazing samples
  let ang = atan2(pos.y, pos.x);
  // log-spiral coordinate: a fixed point in (ρ, ang) traces a spiral arm as ρ grows.
  // Make the spiral angle the DOMINANT axis so features stretch ALONG the arms (filaments),
  // not across them (concentric rings); the radial axis only modulates them weakly.
  let spiral = ang + u.p5.y * u.camFwd.w - log(max(rc, 1e-3)) * 3.2;
  let q = vec3<f32>(spiral * 0.5, rc * 0.32, pos.z * 1.3) * u.p5.x;
  // Domain warp: displace the sample point by a low-freq noise field so the streaks swirl
  // and tangle organically instead of running as clean parallel arcs.
  let warp = vec2<f32>(fbm(q * 0.6), fbm(q * 0.6 + vec3<f32>(5.2, 1.3, 8.9))) - vec2<f32>(0.5);
  let qw = q + vec3<f32>(warp.x, warp.y, 0.0) * 1.3;
  let envelope = smoothstep(0.25, 0.72, fbm(qw));     // broad gas distribution (carves dark gaps)
  let wisps = fbmRidged(qw * 3.2);                     // sharp thin filaments inside the gas
  let turb = envelope * (0.10 + 1.05 * pow(wisps, 2.3));
  return base * clamp(turb, 0.0, 1.3);
}
fn disk(r : f32, phi : f32, dop : f32) -> vec3<f32> {
  let rIn = u.p0.x;  let rOut = u.p0.y;
  let t = clamp((r - rIn) / (rOut - rIn), 0.0, 1.0);   // 0 inner .. 1 outer
  // Observed temperature: a Shakura–Sunyaev profile T ∝ r^(−3/4) shifted by the relativistic
  // factor g = dop (Doppler × gravitational redshift). It *indexes* a stylized hot→cool palette
  // (white-hot core → orange → cool steel-blue gas) rather than a literal blackbody map —
  // a real disk's temperature range is almost all pale white, which renders as a flat cream
  // wash. Doppler still drives it physically: approaching side whitens, receding side reddens.
  let Tobs = 11000.0 * pow(rIn / max(r, rIn), 0.75) * dop;
  let h = clamp((Tobs - 2500.0) / 9000.0, 0.0, 1.0);   // 0 cool .. 1 hot
  let cCool = vec3<f32>(0.32, 0.46, 0.72);             // cool outer gas: steel blue-grey
  let cWarm = vec3<f32>(1.0, 0.55, 0.20);              // orange
  let cHot  = vec3<f32>(1.0, 0.93, 0.82);             // white-hot core
  var col = mix(cCool, cWarm, smoothstep(0.05, 0.55, h));
  col = mix(col, cHot, smoothstep(0.62, 1.0, h));
  // Inner-edge incandescence: a tight white-hot ring at the ISCO, additive.
  let ring = exp(-pow(t * 9.0, 2.0));
  col = col + vec3<f32>(1.0, 0.8, 0.55) * ring * 1.3;
  // Concentrate luminance strongly toward the inner edge, so the disk has the
  // dim-outer / blazing-inner dynamic range instead of a uniform mid-grey wash.
  let bright = mix(0.35, 3.2, pow(1.0 - t, 2.4));
  let beam = pow(dop, 3.0);                            // relativistic beaming (δ³)
  return col * bright * beam * u.p3.y * u.p2.y;        // * diskBrightness * exposure
}

// Kerr radial coordinate r(x,y,z): the natural (oblate) radius whose level sets are the
// Kerr horizons. r² = ½((ρ²-a²) + sqrt((ρ²-a²)²+4a²z²)),  ρ²=x²+y²+z².
fn kerrRadius(pos : vec3<f32>, a2 : f32) -> f32 {
  let w = dot(pos, pos) - a2;
  return sqrt(max(0.5 * (w + sqrt(w * w + 4.0 * a2 * pos.z * pos.z)), 0.0));
}

// ---- the ray tracer --------------------------------------------------------
fn trace(camPos : vec3<f32>, dir : vec3<f32>) -> vec3<f32> {
  let metric = i32(u.p1.z + 0.5);
  let M = u.camUp.w;
  let a = u.p0.w;                                   // spin (0 unless Kerr)
  let maxSteps = i32(u.p1.x + 0.5);
  let stepScale = u.p1.y;
  let escapeR = u.p0.z;
  let hmin = u.p2.z;  let hmax = u.p2.w;
  let diskOn = u.p1.w;

  let H = max(u.p3.z, 1e-3);                        // disk half-thickness (vertical scale)
  var s = makePhoton(camPos, dir, metric, M, a);
  var accum = vec3<f32>(0.0);                        // emission gathered along the ray
  var trans = 1.0;                                   // remaining transmittance (1 → clear)
  for (var i = 0; i < maxSteps; i = i + 1) {
    let pos = s.q.yzw;
    let rho = length(pos);
    // Capture at each spacetime's event horizon — the metric-derived null surface (Δ=0),
    // following standard GRRT practice (e.g. RAPTOR/ipole/Blacklight). NOT an arbitrary
    // "2GM": it is the actual horizon, and reduces to 2M for Schwarzschild.
    //   • Kerr:  the Kerr radial coordinate r < r₊ = M + √(M²−a²) — exact at every
    //            latitude (the Euclidean radius is only right on the equator).
    //   • static slicings (isotropic Schwarzschild): the lapse → 0, so the redshift
    //            -g^{00}=1/lapse² → ∞; a threshold catches it. (Also a deep-field safety.)
    // FLAT: redshift=1, no horizon → never captured. radial is kept for step refinement.
    let gm = metricInverse(s.q, metric, M, a);
    let redshift = -gm.c[0].v;
    let radial = (gm.c[5].v * pos.x * pos.x + gm.c[10].v * pos.y * pos.y + gm.c[15].v * pos.z * pos.z
      + 2.0 * (gm.c[6].v * pos.x * pos.y + gm.c[7].v * pos.x * pos.z + gm.c[11].v * pos.y * pos.z))
      / max(dot(pos, pos), 1e-6);
    var captured = redshift > 50.0;
    if (metric == 2) {
      let rPlus = M + sqrt(max(M * M - a * a, 0.0));
      if (kerrRadius(pos, a * a) < rPlus * 1.01) { captured = true; }
    }
    if (captured) { return accum; }                  // disk emission in front of a black hole
    if (rho > escapeR) {
      let d = rhs(s, metric, M, a);                  // bent coordinate velocity
      return accum + trans * background(normalize(d.q.yzw));
    }
    let refine = 0.25 + 0.75 * clamp(radial, 0.0, 1.0);
    let h = clamp(stepScale * rho * refine, hmin, hmax);
    let sN = rk4(s, h, metric, M, a, gm);              // reuse the metric already evaluated above
    // Volumetric accretion disk: a Gaussian slab about z=0. The disk integral is sub-sampled
    // ALONG the segment (independent of the geodesic step), so a thick / edge-on disk stays
    // smooth without forcing tiny geodesic steps. Emission–absorption transfer dI=(S−I)κ ds,
    // with emission written S·κ so the opaque disk emerges at its source colour S.
    let p0 = s.q.yzw;  let p1 = sN.q.yzw;
    let zlo = min(p0.z, p1.z);  let zhi = max(p0.z, p1.z);
    let midr = length(0.5 * (p0.xy + p1.xy));
    if (diskOn > 0.5 && zlo < 3.0 * H && zhi > -3.0 * H && midr > u.p0.x - h && midr < u.p0.y + h) {
      let vray = normalize(p1 - p0);
      let ds = length(p1 - p0) / 8.0;
      let lapse = 1.0 / sqrt(max(redshift, 1e-3));   // gravitational redshift (slowly varying)
      let spinSign = select(1.0, sign(a), metric == 2 && a != 0.0); // disk co-rotates with Kerr spin
      for (var k = 0; k < 8; k = k + 1) {
        let pp = mix(p0, p1, (f32(k) + 0.5) / 8.0);
        let rc = sqrt(pp.x * pp.x + pp.y * pp.y);
        if (rc >= u.p0.x && rc <= u.p0.y && abs(pp.z) < 4.0 * H) {   // loose bbox; diskDensity's smooth windows antialias the edges
          let dens = diskDensity(pp, rc, H);
          let tang = spinSign * vec3<f32>(-pp.y, pp.x, 0.0) / rc;
          let beta = clamp(sqrt(M / rc), 0.0, 0.95);
          let delta = sqrt(1.0 - beta * beta) / (1.0 - beta * dot(tang, -vray));
          let g = select(1.0, delta, u.p3.x > 0.5) * lapse;
          let kappa = dens * 4.0;
          accum = accum + trans * disk(rc, atan2(pp.y, pp.x), g) * kappa * ds;
          trans = trans * exp(-kappa * ds);
        }
      }
      if (trans < 0.02) { return accum; }            // fully opaque
    }
    s = sN;
  }
  return accum;   // step budget exhausted near the hole → trapped → black (disk over it)
}

// ---- Ellis/Dneg wormhole (Thorne et al. 2015, arXiv:1502.03809) -------------
// Ultrastatic: ds² = -dt² + dℓ² + r(ℓ)²(dθ² + sin²θ dφ²). No horizon — rays either thread
// the throat to the OTHER universe (ℓ<0) or are lensed back to ours (ℓ>0). Spherical
// symmetry ⇒ every ray is planar, so we integrate a pole-free 2D problem per ray.
//   r(ℓ) = ρ for |ℓ|≤a;  else ρ + M[x·atan x − ½ln(1+x²)], x = 2(|ℓ|−a)/(πM).
fn wh_r(l : f32) -> f32 {
  let rho = u.p4.x;  let a = u.p4.y;  let Mw = max(u.p4.z, 1e-4);
  let al = abs(l);
  if (al <= a) { return rho; }
  let x = 2.0 * (al - a) / (PI * Mw);
  return rho + Mw * (x * atan(x) - 0.5 * log(1.0 + x * x));
}
fn wh_drdl(l : f32) -> f32 {
  let a = u.p4.y;  let Mw = max(u.p4.z, 1e-4);
  let al = abs(l);
  if (al <= a) { return 0.0; }
  return sign(l) * (2.0 / PI) * atan(2.0 * (al - a) / (PI * Mw));
}
// state = (ℓ, ℓ̇, φ). Second-order radial eqn ℓ̈ = (b²/r³)(dr/dℓ) is smooth through the
// turning point (no sqrt), so RK4 handles reflect-vs-thread without special cases.
fn wh_rhs(st : vec3<f32>, b : f32) -> vec3<f32> {
  let r = wh_r(st.x);
  return vec3<f32>(st.y, (b * b) / (r * r * r) * wh_drdl(st.x), b / (r * r));
}
fn traceWormhole(camPos : vec3<f32>, dir : vec3<f32>) -> vec3<f32> {
  let lcam = length(camPos);
  let rhat = camPos / lcam;
  let vr = dot(dir, rhat);                          // radial component of the view ray
  let tvec = dir - vr * rhat;
  let tmag = length(tvec);
  var that = rhat;
  if (tmag > 1e-5) { that = tvec / tmag; }
  let rcam = wh_r(lcam);
  let b = rcam * tmag;                              // impact parameter (b < ρ ⇒ threads throat)
  var st = vec3<f32>(lcam, vr, 0.0);
  let maxSteps = i32(u.p1.x + 0.5);
  let stepScale = u.p1.y;
  let hmin = u.p2.z;  let hmax = u.p2.w;
  let lfar = max(lcam * 1.4, 40.0);
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (abs(st.x) > lfar) { break; }
    let h = clamp(stepScale * (abs(st.x) + wh_r(st.x)), hmin, hmax);
    let k1 = wh_rhs(st, b);
    let k2 = wh_rhs(st + 0.5 * h * k1, b);
    let k3 = wh_rhs(st + 0.5 * h * k2, b);
    let k4 = wh_rhs(st + h * k3, b);
    st = st + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  }
  let phi = st.z;
  let radialDir = cos(phi) * rhat + sin(phi) * that;
  let tangDir = -sin(phi) * rhat + cos(phi) * that;
  let vel = st.y * radialDir + (b / max(wh_r(st.x), 1e-3)) * tangDir;
  let outDir = normalize(vel);
  if (st.x < 0.0) { return otherBackground(outDir); } // emerged in the other universe
  return background(outDir);                           // lensed back into ours
}

// ---- fullscreen triangle + entry points ------------------------------------
struct VSOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VSOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv = p[vi];
  return o;
}
@fragment
fn fs(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let dir = normalize(u.camFwd.xyz
      + uv.x * u.camRight.w * u.camPos.w * u.camRight.xyz   // aspect * tanHalfFov
      + uv.y * u.camPos.w * u.camUp.xyz);
  if (i32(u.p1.z + 0.5) == 3) {                             // wormhole has its own trace path
    return vec4<f32>(traceWormhole(u.camPos.xyz, dir), 1.0);
  }
  return vec4<f32>(trace(u.camPos.xyz, dir), 1.0);          // linear HDR; tonemap in composite
}
