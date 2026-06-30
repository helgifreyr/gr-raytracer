// Smoke-test: extract the WGSL from index.html and parse it. Catches syntax /
// structural errors (not full type-checking — the browser does the rest).
import { readFileSync } from "node:fs";
import { WgslReflect } from "wgsl_reflect/wgsl_reflect.module.js";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const m = html.match(/const WGSL = \/\* wgsl \*\/`([\s\S]*?)`;/);
if (!m) { console.error("Could not find WGSL block"); process.exit(1); }
const code = m[1];

try {
  const r = new WgslReflect(code);
  const fns = r.functions.map((f) => f.name).join(", ");
  console.log("WGSL parsed OK");
  console.log("  functions:", fns);
  console.log("  entry points:", [...r.entry.vertex, ...r.entry.fragment].map((e) => e.name).join(", "));
  console.log("  uniforms:", r.uniforms.map((u) => u.name).join(", "));
} catch (e) {
  console.error("WGSL PARSE ERROR:\n" + (e && e.message ? e.message : e));
  process.exit(1);
}
