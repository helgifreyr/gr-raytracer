// shadow_probe.mjs — reproduce the shader's trace in JS and print an ASCII shadow map,
// to diagnose the "see-through Kerr shadow". Mirrors index.html trace() exactly.
import { metricInverse, rk4Step, makePhoton, METRIC } from "./engine.mjs";

const M = 1.0, D = 26, stepScale = 0.08, hmin = 0.01, maxSteps = 512;
const escapeR = Math.max(60, D * 1.5), hmax = Math.max(4, escapeR * 0.05);

function captureScalars(w, a) {
  const g = metricInverse(w.slice(0, 4), METRIC.KERR, M, a);
  const p = [w[1], w[2], w[3]];
  const r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
  let radial = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) radial += g[i + 1][j + 1].v * p[i] * p[j];
  const w2 = r2 - a * a;
  const kerrR = Math.sqrt(Math.max(0.5 * (w2 + Math.sqrt(w2 * w2 + 4 * a * a * p[2] * p[2])), 0));
  return { redshift: -g[0][0].v, radial: radial / r2, kerrR };
}

// Classify one ray: 'B' captured, '.' escaped to background, 'o' disk, '#' NaN.
function traceRay(camPos, dir, a) {
  let w = makePhoton(camPos, dir, METRIC.KERR, M, a);
  for (let i = 0; i < maxSteps; i++) {
    const pos = [w[1], w[2], w[3]];
    const rho = Math.hypot(...pos);
    if (!isFinite(rho)) return "#";
    const { redshift, radial, kerrR } = captureScalars(w, a);
    const rPlus = M + Math.sqrt(Math.max(M * M - a * a, 0));
    if (redshift > 50 || kerrR < rPlus * 1.01) return "B";
    if (rho > escapeR) return ".";
    const refine = 0.25 + 0.75 * Math.min(Math.max(radial, 0), 1); // strong-field refinement
    const h = Math.min(Math.max(stepScale * rho * refine, hmin), hmax);
    const wN = rk4Step(w, h, METRIC.KERR, M, a);
    // disk in z=0 plane, rIn=2.6 rOut=14
    if (w[3] * wN[3] < 0) {
      const t = w[3] / (w[3] - wN[3]);
      const hx = w[1] + t * (wN[1] - w[1]), hy = w[2] + t * (wN[2] - w[2]);
      const rr = Math.hypot(hx, hy);
      if (rr >= 2.6 && rr <= 14) return "o";
    }
    w = wN;
  }
  return "B"; // fallthrough → trapped
}

function shadowMap(a, pitch = 0.32) {
  // camera like the app: orbit at distance D, given pitch, yaw=0
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const pos = [D * cp, 0, D * sp];
  const norm = (v) => { const l = Math.hypot(...v); return v.map((c) => c / l); };
  const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const fwd = norm([-pos[0], -pos[1], -pos[2]]);
  const right = norm(cross(fwd, [0, 0, 1]));
  const up = cross(right, fwd);
  const tanHalf = Math.tan(55 * Math.PI / 360);
  const NX = 71, NY = 31, fov = 0.45; // small fov, zoomed on the hole
  let out = `\n=== a=${a}  (B=captured  .=sky  o=disk  #=NaN) ===\n`;
  for (let iy = 0; iy < NY; iy++) {
    const v = (1 - 2 * iy / (NY - 1)) * fov;
    let row = "";
    for (let ix = 0; ix < NX; ix++) {
      const u = (2 * ix / (NX - 1) - 1) * fov * (NX / NY) * 0.5;
      const dir = norm([
        fwd[0] + u * tanHalf * right[0] + v * tanHalf * up[0],
        fwd[1] + u * tanHalf * right[1] + v * tanHalf * up[1],
        fwd[2] + u * tanHalf * right[2] + v * tanHalf * up[2],
      ]);
      row += traceRay(pos, dir, a);
    }
    out += row + "\n";
  }
  return out;
}

for (const a of [0.0, 0.7, 0.998]) console.log(shadowMap(a));
