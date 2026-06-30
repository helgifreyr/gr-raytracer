// Smoke-test: parse both WGSL shader files. Catches syntax / structural errors
// (not full type-checking — the browser's WebGPU driver does the rest).
import { readFileSync } from "node:fs";
import { WgslReflect } from "wgsl_reflect/wgsl_reflect.module.js";

function parse(label, path) {
  const code = readFileSync(new URL(path, import.meta.url), "utf8");
  try {
    const r = new WgslReflect(code);
    console.log(`${label} parsed OK — ${r.functions.length} fns, entry points: ` +
      [...r.entry.vertex, ...r.entry.fragment].map((e) => e.name).join(", "));
  } catch (e) {
    console.error(`${label} PARSE ERROR:\n` + (e && e.message ? e.message : e));
    process.exit(1);
  }
}

parse("scene.wgsl", "../src/shaders/scene.wgsl");
parse("post.wgsl", "../src/shaders/post.wgsl");
