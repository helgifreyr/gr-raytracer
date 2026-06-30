// Smoke-test: extract BOTH WGSL modules from index.html and parse them. Catches syntax /
// structural errors (not full type-checking — the browser does the rest).
import { readFileSync } from "node:fs";
import { WgslReflect } from "wgsl_reflect/wgsl_reflect.module.js";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

function parse(label, re) {
  const m = html.match(re);
  if (!m) { console.error(`Could not find ${label} block`); process.exit(1); }
  try {
    const r = new WgslReflect(m[1]);
    console.log(`${label} parsed OK — ${r.functions.length} fns, entry points: ` +
      [...r.entry.vertex, ...r.entry.fragment].map((e) => e.name).join(", "));
  } catch (e) {
    console.error(`${label} PARSE ERROR:\n` + (e && e.message ? e.message : e));
    process.exit(1);
  }
}

parse("WGSL (scene)", /const WGSL = \/\* wgsl \*\/`([\s\S]*?)`;/);
parse("POST_WGSL (bloom)", /const POST_WGSL = \/\* wgsl \*\/`([\s\S]*?)`;/);
