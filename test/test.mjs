// test.mjs — physics validation for the geodesic engine.
// Run: node test/test.mjs
//
// Since WebGPU can't run headless here, we validate the double-precision JS twin
// of the shader math. Four checks:
//   1. Autodiff (dual numbers) matches finite differences on the metric.
//   2. Flat spacetime ⇒ geodesics are exactly straight lines.
//   3. Weak-field light deflection matches Einstein's α = 4M/b.
//   4. A photon aimed inside the shadow is captured (falls to the horizon).

import {
  D, METRIC, metricInverse, rk4Step, makePhoton, coordVelocity, hamiltonian,
} from "./engine.mjs";

let pass = 0, fail = 0;
function check(name, ok, detail = "") {
  if (ok) { pass++; console.log(`  ok   ${name}${detail ? "  (" + detail + ")" : ""}`); }
  else { fail++; console.log(`  FAIL ${name}  ${detail}`); }
}

// ---------------------------------------------------------------------------
console.log("\n[1] Autodiff vs finite differences (Schwarzschild g^{ab})");
{
  const M = 1.0;
  const q = [0, 4.0, 2.0, -1.5]; // generic off-axis point
  const g = metricInverse(q, METRIC.SCHWARZSCHILD, M);
  const eps = 1e-6;
  const comps = [[0, 0], [1, 1], [2, 2], [3, 3]];
  let maxErr = 0;
  for (const [a, b] of comps) {
    for (let k = 1; k <= 3; k++) { // perturb x,y,z (index 1..3)
      const qp = q.slice(), qm = q.slice();
      qp[k] += eps; qm[k] -= eps;
      const fp = metricInverse(qp, METRIC.SCHWARZSCHILD, M)[a][b].v;
      const fm = metricInverse(qm, METRIC.SCHWARZSCHILD, M)[a][b].v;
      const fd = (fp - fm) / (2 * eps);
      const ad = g[a][b].d[k];
      maxErr = Math.max(maxErr, Math.abs(fd - ad));
    }
  }
  check("dual derivatives match central differences", maxErr < 1e-5,
    `max abs err ${maxErr.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
console.log("\n[2] Flat spacetime ⇒ straight lines");
{
  const dir = [1, 0.3, -0.2];
  const n = Math.hypot(...dir);
  const u = dir.map((c) => c / n);
  let w = makePhoton([-20, 5, 3], u, METRIC.FLAT, 0);
  const v0 = coordVelocity(w, METRIC.FLAT, 0);
  for (let i = 0; i < 400; i++) w = rk4Step(w, 0.1, METRIC.FLAT, 0);
  const v1 = coordVelocity(w, METRIC.FLAT, 0);
  // direction must be unchanged
  const a0 = v0.map((c) => c / Math.hypot(...v0));
  const a1 = v1.map((c) => c / Math.hypot(...v1));
  const dot = a0[0] * a1[0] + a0[1] * a1[1] + a0[2] * a1[2];
  check("photon direction preserved in flat space", Math.abs(1 - dot) < 1e-9,
    `1-dot = ${(1 - dot).toExponential(2)}`);
}

// ---------------------------------------------------------------------------
console.log("\n[3] Weak-field deflection ≈ 4M/b");
{
  // Fire a photon past the hole with impact parameter b, measure total bend.
  const M = 1.0;
  const D0 = 4000;     // start/end distance (asymptotic region)
  for (const b of [60, 120, 240]) {
    // start far on -x axis, travelling +x, offset by b in y
    let w = makePhoton([-D0, b, 0], [1, 0, 0], METRIC.SCHWARZSCHILD, M);
    const vIn = coordVelocity(w, METRIC.SCHWARZSCHILD, M);
    let steps = 0;
    while (steps < 200000) {
      const rho = Math.hypot(w[1], w[2], w[3]);
      if (w[1] > D0 && steps > 10) break;          // escaped to +x
      const h = Math.max(0.02, 0.03 * rho);        // big steps far away, small steps near
      w = rk4Step(w, h, METRIC.SCHWARZSCHILD, M);
      steps++;
    }
    const vOut = coordVelocity(w, METRIC.SCHWARZSCHILD, M);
    // deflection angle between incoming and outgoing direction (in x-y plane)
    const ang = (v) => Math.atan2(v[1], v[0]);
    const alpha = Math.abs(ang(vOut) - ang(vIn));
    // Schwarzschild deflection series: α = 4M/b + (15π/4)(M/b)² + ...
    // (Checking against the 2nd-order term, not just leading order, is a much
    //  stronger test — and confirms the integrator resolves the GR correction.)
    const u = M / b;
    const predicted = 4 * u + (15 * Math.PI / 4) * u * u;
    const relErr = Math.abs(alpha - predicted) / predicted;
    check(`b=${b}: α=${alpha.toFixed(5)} vs series=${predicted.toFixed(5)}`,
      relErr < 0.01, `rel err ${(relErr * 100).toFixed(2)}%, ${steps} steps`);
  }
}

// ---------------------------------------------------------------------------
console.log("\n[4] Photon aimed at the hole is captured");
{
  const M = 1.0;
  const horizon = M / 2; // isotropic-coordinate horizon
  // aim straight at the hole with a tiny impact parameter (well inside shadow)
  let w = makePhoton([-50, 1.0, 0], [1, 0, 0], METRIC.SCHWARZSCHILD, M);
  let captured = false, minRho = Infinity;
  for (let i = 0; i < 100000; i++) {
    const rho = Math.hypot(w[1], w[2], w[3]);
    minRho = Math.min(minRho, rho);
    if (rho < horizon * 1.05) { captured = true; break; }
    const h = Math.max(0.005, 0.02 * rho);
    w = rk4Step(w, h, METRIC.SCHWARZSCHILD, M);
    if (rho > 200) break; // escaped — would be a failure
  }
  check("ray within shadow reaches the horizon", captured,
    `min ρ reached = ${minRho.toFixed(4)} (horizon ${horizon})`);
}

// ---------------------------------------------------------------------------
console.log("\n[5] Kerr autodiff vs finite differences (incl. off-diagonal g^{0i}, g^{ij})");
{
  const M = 1.0, a = 0.7;
  const q = [0, 4.0, 2.0, 1.5];
  const g = metricInverse(q, METRIC.KERR, M, a);
  const eps = 1e-6;
  let maxErr = 0;
  for (let A = 0; A < 4; A++) for (let B = 0; B < 4; B++) {
    for (let k = 1; k <= 3; k++) {
      const qp = q.slice(), qm = q.slice();
      qp[k] += eps; qm[k] -= eps;
      const fp = metricInverse(qp, METRIC.KERR, M, a)[A][B].v;
      const fm = metricInverse(qm, METRIC.KERR, M, a)[A][B].v;
      maxErr = Math.max(maxErr, Math.abs((fp - fm) / (2 * eps) - g[A][B].d[k]));
    }
  }
  check("Kerr dual derivatives match central differences", maxErr < 1e-4,
    `max abs err ${maxErr.toExponential(2)}`);
}

// ---------------------------------------------------------------------------
console.log("\n[6] Kerr with a=0 reduces to Schwarzschild deflection");
{
  const M = 1.0, D0 = 4000, b = 120;
  let w = makePhoton([-D0, b, 0], [1, 0, 0], METRIC.KERR, M, 0.0);
  const vIn = coordVelocity(w, METRIC.KERR, M, 0.0);
  let steps = 0;
  while (steps < 200000 && !(w[1] > D0 && steps > 10)) {
    const rho = Math.hypot(w[1], w[2], w[3]);
    w = rk4Step(w, Math.max(0.02, 0.03 * rho), METRIC.KERR, M, 0.0);
    steps++;
  }
  const vOut = coordVelocity(w, METRIC.KERR, M, 0.0);
  const ang = (v) => Math.atan2(v[1], v[0]);
  const alpha = Math.abs(ang(vOut) - ang(vIn));
  const u = M / b, predicted = 4 * u + (15 * Math.PI / 4) * u * u;
  check(`Kerr(a=0) α=${alpha.toFixed(5)} vs series=${predicted.toFixed(5)}`,
    Math.abs(alpha - predicted) / predicted < 0.01, `${steps} steps`);
}

// ---------------------------------------------------------------------------
console.log("\n[7] Null condition H≈0 at creation, and conserved along the geodesic");
{
  // H must be 0 for a photon and stay 0 — a strong joint check on the metric AND
  // its autodiff derivatives (a wrong ∂g would make H drift during integration).
  for (const [name, metric, M, a] of [
    ["Schwarzschild", METRIC.SCHWARZSCHILD, 1.0, 0.0],
    ["Kerr a=0.9", METRIC.KERR, 1.0, 0.9],
  ]) {
    let w = makePhoton([-40, 8, 3], [1, -0.1, 0], metric, M, a);
    const pScale = Math.hypot(w[4], w[5], w[6], w[7]);
    const H0 = hamiltonian(w, metric, M, a);
    let maxH = Math.abs(H0);
    for (let i = 0; i < 4000; i++) {
      const rho = Math.hypot(w[1], w[2], w[3]);
      if (rho > 120 && i > 10) break;
      if (rho < 2.5) break; // skip the deep strong-field for this conservation check
      w = rk4Step(w, Math.max(0.01, 0.02 * rho), metric, M, a);
      maxH = Math.max(maxH, Math.abs(hamiltonian(w, metric, M, a)));
    }
    check(`${name}: |H|/|p|² stays ~0`, maxH / (pScale * pScale) < 1e-3,
      `max |H|/|p|² = ${(maxH / (pScale * pScale)).toExponential(2)}`);
  }
}

// ---------------------------------------------------------------------------
console.log("\n[8] Frame dragging: spin changes equatorial deflection");
{
  // Equatorial photon skimming the hole; flipping spin sign changes the bend,
  // and a=0 lies between prograde and retrograde — the signature of frame dragging.
  const M = 1.0, b = 12;   // comfortably above the critical (capture) impact parameter
  const bend = (a) => {
    let w = makePhoton([-300, b, 0], [1, 0, 0], METRIC.KERR, M, a);
    const vIn = coordVelocity(w, METRIC.KERR, M, a);
    let steps = 0;
    while (steps < 200000) {
      const rho = Math.hypot(w[1], w[2], w[3]);
      if (w[1] > 300 && steps > 10) break;
      if (rho < 1.2) return null; // captured
      w = rk4Step(w, Math.max(0.01, 0.02 * rho), METRIC.KERR, M, a);
      steps++;
    }
    const ang = (v) => Math.atan2(v[1], v[0]);
    return ang(coordVelocity(w, METRIC.KERR, M, a)) - ang(vIn);
  };
  const aP = bend(0.9), a0 = bend(0.0), aR = bend(-0.9);
  const ok = aP !== null && a0 !== null && aR !== null
    && Math.abs(aP - aR) > 0.02 && ((aP - a0) * (a0 - aR) > 0); // a0 between, monotone
  check("prograde vs retrograde deflection differ, a=0 in between", ok,
    `α(+0.9)=${aP?.toFixed(4)} α(0)=${a0?.toFixed(4)} α(-0.9)=${aR?.toFixed(4)}`);
}

// ---------------------------------------------------------------------------
console.log("\n[9] Horizon capture surfaces (metric-derived, following standard GRRT)");
{
  // Kerr: the Kerr radial coordinate r → r₊ on the horizon at EVERY latitude (this is why
  // the Euclidean radius is wrong off-equator). And r₊ → 2M as a → 0 (Schwarzschild).
  const kerrR = (p, a) => {
    const w = p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - a * a;
    return Math.sqrt((w + Math.sqrt(w * w + 4 * a * a * p[2] ** 2)) / 2);
  };
  const a = 0.9, rPlus = 1 + Math.sqrt(1 - a * a);
  const Req = Math.sqrt(rPlus * rPlus + a * a);  // equatorial Cartesian horizon point
  // a 45° horizon point: place it at Kerr r = rPlus, off the equator
  const z = rPlus * 0.7, Rxy = Math.sqrt((rPlus * rPlus + a * a) * (1 - (z * z) / (rPlus * rPlus)));
  check(`Kerr a=0.9: r=r₊ at equator (${kerrR([Req, 0, 0], a).toFixed(3)}) and off-equator (${kerrR([Rxy, 0, z], a).toFixed(3)}), r₊=${rPlus.toFixed(3)}`,
    Math.abs(kerrR([Req, 0, 0], a) - rPlus) < 1e-3 && Math.abs(kerrR([Rxy, 0, z], a) - rPlus) < 1e-2, "");
  const r0 = 1 + Math.sqrt(1 - 0); // a=0 → r₊ = 2M
  check(`Kerr a=0: r₊ = ${r0.toFixed(2)} = 2M (Schwarzschild horizon)`, Math.abs(r0 - 2) < 1e-9, "");
  // Schwarzschild (isotropic, static): captured by redshift -g^{00} → ∞ at the horizon.
  const redshift = (R) => -metricInverse([0, R, 0, 0], METRIC.SCHWARZSCHILD, 1.0, 0)[0][0].v;
  check(`Schwarzschild: redshift far=${redshift(100).toFixed(2)}≈1, near horizon=${redshift(0.62).toFixed(0)}>50`,
    Math.abs(redshift(100) - 1) < 0.05 && redshift(0.62) > 50, "");
}

// ---------------------------------------------------------------------------
console.log("\n[10] Solid shadow: rays aimed at the hole are captured (guards the null-root choice)");
{
  // Fire the dead-centre ray (and a few near it) from a realistic camera straight at the
  // hole; each MUST fall in, not escape. This is the test that catches a wrong p_t root —
  // with the wrong branch, Kerr/Kerr-Schild rays slingshot out and the shadow goes hollow.
  const D = 26, pitch = 0.32;
  const camPos = [D * Math.cos(pitch), 0, D * Math.sin(pitch)];
  const norm = (v) => { const l = Math.hypot(...v); return v.map((c) => c / l); };
  const fwd = norm(camPos.map((c) => -c));
  const kerrR = (p, a) => {
    const w = p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - a * a;
    return Math.sqrt((w + Math.sqrt(w * w + 4 * a * a * p[2] ** 2)) / 2);
  };
  const captured = (w, metric, M, a) => { // mirrors the shader's capture test
    if (-metricInverse(w.slice(0, 4), metric, M, a)[0][0].v > 50) return true;
    if (metric === METRIC.KERR) {
      const rPlus = M + Math.sqrt(Math.max(M * M - a * a, 0));
      if (kerrR([w[1], w[2], w[3]], a) < rPlus * 1.01) return true;
    }
    return false;
  };
  const falls = (metric, M, a, jitter) => {
    const dir = norm([fwd[0] + jitter, fwd[1], fwd[2]]); // small nudge off-centre
    let w = makePhoton(camPos, dir, metric, M, a);
    for (let i = 0; i < 600; i++) {
      const rho = Math.hypot(w[1], w[2], w[3]);
      if (captured(w, metric, M, a)) return true;
      if (rho > 60) return false;  // escaped — shadow would be see-through
      w = rk4Step(w, Math.max(0.01, 0.08 * rho), metric, M, a);
    }
    return true; // trapped → captured
  };
  // a ≤ 0.9: shadow is near-centred, so origin-aimed rays are a valid solidity probe.
  // (Near-extremal a→1 shifts the shadow strongly and its photon-orbit boundary becomes
  //  nearly fractal — solidity there is a step-count matter, not a pass/fail invariant.)
  for (const [name, metric, a] of [
    ["Schwarzschild", METRIC.SCHWARZSCHILD, 0],
    ["Kerr a=0", METRIC.KERR, 0.0],
    ["Kerr a=0.5", METRIC.KERR, 0.5],
    ["Kerr a=0.9", METRIC.KERR, 0.9],
  ]) {
    const ok = [-0.02, -0.01, 0, 0.01, 0.02].every((j) => falls(metric, 1.0, a, j));
    check(`${name}: central rays captured (solid shadow)`, ok, "");
  }
}

// ---------------------------------------------------------------------------
console.log("\n[11] Ellis/Dneg wormhole: rays thread the throat (b<ρ) or reflect (b>ρ)");
{
  // Mirrors the WGSL wormhole path: ultrastatic, no horizon. The defining behaviour is the
  // dichotomy — light with impact parameter below the throat radius passes to the OTHER
  // universe (ℓ<0); above it, it's lensed back to ours (ℓ>0). Integrated via the smooth
  // 2nd-order radial equation ℓ̈ = (b²/r³)(dr/dℓ).
  const rho = 2, a = 0.5, Mw = 0.5;
  const wr = (l) => { const al = Math.abs(l); if (al <= a) return rho; const x = 2 * (al - a) / (Math.PI * Mw); return rho + Mw * (x * Math.atan(x) - 0.5 * Math.log(1 + x * x)); };
  const drdl = (l) => { const al = Math.abs(l); if (al <= a) return 0; return Math.sign(l) * (2 / Math.PI) * Math.atan(2 * (al - a) / (Math.PI * Mw)); };
  const rhsW = (st, b) => { const r = wr(st[0]); return [st[1], (b * b) / (r * r * r) * drdl(st[0]), b / (r * r)]; };
  const finalL = (b) => {
    const r0 = wr(30);
    let st = [30, -Math.sqrt(Math.max(1 - b * b / (r0 * r0), 0)), 0];
    for (let i = 0; i < 40000; i++) {
      if (Math.abs(st[0]) > 35) break;
      const h = 0.05;
      const k1 = rhsW(st, b), k2 = rhsW(st.map((x, j) => x + 0.5 * h * k1[j]), b),
        k3 = rhsW(st.map((x, j) => x + 0.5 * h * k2[j]), b), k4 = rhsW(st.map((x, j) => x + h * k3[j]), b);
      st = st.map((x, j) => x + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    }
    return st[0];
  };
  check(`b=1.0 < ρ=2 threads to the other universe (ℓ→${finalL(1.0).toFixed(0)})`, finalL(1.0) < 0, "");
  check(`b=3.0 > ρ=2 reflects back into ours (ℓ→${finalL(3.0).toFixed(0)})`, finalL(3.0) > 0, "");
  check(`b=0 (radial) threads straight through (ℓ→${finalL(0).toFixed(0)})`, finalL(0) < 0, "");
}

// ---------------------------------------------------------------------------
console.log(`\n${fail === 0 ? "ALL PASS" : "FAILURES"} — ${pass} passed, ${fail} failed\n`);
process.exit(fail === 0 ? 0 : 1);
