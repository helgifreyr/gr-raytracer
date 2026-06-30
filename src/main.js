// Host module: WebGPU setup, camera, UI, render loop. The two WGSL shaders are loaded
// from src/shaders/*.wgsl (so they need serving over http — Pages, or `npm run serve`).
// ===========================================================================
//  WGSL: the metric-agnostic geodesic engine, transliterated line-for-line
//  from test/engine.mjs (which the Node suite validates against GR).
//  Forward-mode autodiff (dual numbers carrying the full 4-gradient) lets ONE
//  metric evaluation produce every partial the Hamiltonian needs. The metric
//  is the ONLY physics-specific code — everything below `metricInverse` is generic.
// ===========================================================================
const WGSL = await fetch(new URL('./shaders/scene.wgsl', import.meta.url)).then((r) => r.text());

// ===========================================================================
//  Post-processing: HDR bloom. The scene renders to an rgba16float texture; we
//  bright-pass + separably blur it (half-res), then composite scene+bloom and tonemap.
// ===========================================================================
const POST_WGSL = await fetch(new URL('./shaders/post.wgsl', import.meta.url)).then((r) => r.text());

// ===========================================================================
//  Host: WebGPU setup, camera, UI, render loop.
// ===========================================================================
(async function () {
  const fail = (msg) => {
    document.getElementById("err").style.display = "grid";
    if (msg) document.getElementById("errmsg").textContent = msg;
  };
  if (!navigator.gpu) return fail("navigator.gpu is undefined.");

  let adapter;
  try { adapter = await navigator.gpu.requestAdapter(); }
  catch (e) { return fail(String(e)); }
  if (!adapter) return fail("No suitable GPU adapter.");

  let device;
  try { device = await adapter.requestDevice(); }
  catch (e) { return fail("requestDevice failed: " + String(e)); }
  let deviceLost = false;
  device.lost.then((info) => { deviceLost = true; fail("GPU device lost: " + (info && info.message || info && info.reason || "")); });
  device.addEventListener("uncapturederror", (e) => console.error("WebGPU error:", e.error));

  const canvas = document.getElementById("gpu");
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const HDR = "rgba16float";
  async function makeModule(code, label) {
    const m = device.createShaderModule({ code });
    const inf = await m.getCompilationInfo();
    const e = inf.messages.filter((x) => x.type === "error");
    if (e.length) { fail(label + " shader error: " + e.map((x) => x.message).join(" | ")); return null; }
    return m;
  }
  const sceneModule = await makeModule(WGSL, "scene");
  const postModule = await makeModule(POST_WGSL, "post");
  if (!sceneModule || !postModule) return;

  // Scene → HDR; blur (bright-pass + Gaussian) → HDR; composite (+tonemap) → swapchain.
  const scenePipeline = device.createRenderPipeline({
    layout: "auto", vertex: { module: sceneModule, entryPoint: "vs" },
    fragment: { module: sceneModule, entryPoint: "fs", targets: [{ format: HDR }] },
    primitive: { topology: "triangle-list" },
  });
  const blurPipeline = device.createRenderPipeline({
    layout: "auto", vertex: { module: postModule, entryPoint: "vs" },
    fragment: { module: postModule, entryPoint: "fsBlur", targets: [{ format: HDR }] },
    primitive: { topology: "triangle-list" },
  });
  const compositePipeline = device.createRenderPipeline({
    layout: "auto", vertex: { module: postModule, entryPoint: "vs" },
    fragment: { module: postModule, entryPoint: "fsComposite", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const UBO_FLOATS = 40; // 10 * vec4
  const ubuf = device.createBuffer({ size: UBO_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // 3-level bloom: 6 separable-blur passes (H+V per octave) → 6 small uniform buffers.
  const blurBufs = Array.from({ length: 6 }, () =>
    device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const compBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const sampler = device.createSampler({   // sky sampler: longitude wraps, clamp at poles
    magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge",
  });
  const postSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" }); // clamp
  const TEX_USAGE = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
  let skyTexture = device.createTexture({ size: [1, 1], format: "rgba8unorm-srgb", usage: TEX_USAGE });
  device.queue.writeTexture({ texture: skyTexture }, new Uint8Array([12, 14, 22, 255]), { bytesPerRow: 4 }, [1, 1]);
  let skyTexture2 = device.createTexture({ size: [1, 1], format: "rgba8unorm-srgb", usage: TEX_USAGE });
  device.queue.writeTexture({ texture: skyTexture2 }, new Uint8Array([30, 16, 14, 255]), { bytesPerRow: 4 }, [1, 1]);
  let image2Loaded = 0;

  let sceneBind;
  function rebuildSceneBind() {
    sceneBind = device.createBindGroup({
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ubuf } },
        { binding: 1, resource: skyTexture.createView() },
        { binding: 2, resource: sampler },
        { binding: 3, resource: skyTexture2.createView() },
      ],
    });
  }
  rebuildSceneBind();

  // Offscreen targets (recreated on resize): full-res HDR scene + a 3-octave bloom
  // pyramid (½, ¼, ⅛ res). Each octave has an A texture (after the horizontal pass)
  // and a B texture (after the vertical pass); blurTex = [A0,B0, A1,B1, A2,B2].
  let texHDR, blurTex, blurBinds, compBind;
  function rebuildTargets(w, h) {
    const dims = [
      [Math.max(1, w >> 1), Math.max(1, h >> 1)],
      [Math.max(1, w >> 2), Math.max(1, h >> 2)],
      [Math.max(1, w >> 3), Math.max(1, h >> 3)],
    ];
    [texHDR, ...(blurTex || [])].forEach((t) => t && t.destroy());
    texHDR = device.createTexture({ size: [w, h], format: HDR, usage: TEX_USAGE });
    blurTex = [];
    dims.forEach(([lw, lh]) => {
      blurTex.push(device.createTexture({ size: [lw, lh], format: HDR, usage: TEX_USAGE })); // A
      blurTex.push(device.createTexture({ size: [lw, lh], format: HDR, usage: TEX_USAGE })); // B
    });
    const A = (i) => blurTex[i * 2], B = (i) => blurTex[i * 2 + 1];
    // Pass chain (src → dst): bright-pass+H at L0, then V; each coarser octave blurs the
    // previous octave's blurred output, downsampling via the smaller target + linear filter.
    const src = [texHDR, A(0), B(0), A(1), B(1), A(2)];   // dst is blurTex[i] = [A0,B0,A1,B1,A2,B2]
    blurBinds = src.map((s, i) => device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: postSampler }, { binding: 1, resource: s.createView() }, { binding: 2, resource: { buffer: blurBufs[i] } }],
    }));
    compBind = device.createBindGroup({ layout: compositePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: postSampler }, { binding: 1, resource: texHDR.createView() },
      { binding: 2, resource: B(0).createView() }, { binding: 3, resource: B(1).createView() },
      { binding: 4, resource: B(2).createView() }, { binding: 5, resource: { buffer: compBuf } }] });
    // Blur offsets in uv, stepped in TARGET-res texels so each octave's blur is isotropic.
    // Only the very first (L0 horizontal) pass applies the bright-pass threshold (0.7);
    // every later pass just blurs (threshold -1) the already-extracted highlights.
    const spread = 2.5;
    for (let i = 0; i < 6; i++) {
      const [tw, th] = dims[i >> 1];
      const horizontal = (i % 2) === 0;
      const thr = i === 0 ? 1.2 : -1.0;   // only genuinely bright gas/core blooms, not the mid disk
      device.queue.writeBuffer(blurBufs[i], 0,
        new Float32Array([horizontal ? spread / tw : 0, horizontal ? 0 : spread / th, thr, 0]));
    }
  }
  const u = new Float32Array(UBO_FLOATS);

  // ---- camera --------------------------------------------------------------
  const cam = { dist: 26, yaw: 0.6, pitch: 0.32, fov: 55 * Math.PI / 180 };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

  // ---- UI ------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const ui = {
    metric: $("metric"), mass: $("mass"), spin: $("spin"), disk: $("disk"),
    wr: $("wr"), wa: $("wa"), ww: $("ww"),
    rin: $("rin"), rout: $("rout"), dthick: $("dthick"), dbright: $("dbright"), doppler: $("doppler"),
    dnoise: $("dnoise"), dswirl: $("dswirl"),
    anim: $("anim"), bg: $("bg"), steps: $("steps"), ss: $("ss"), res: $("res"), exp: $("exp"),
    bloom: $("bloom"), bloomamt: $("bloomamt"),
  };

  // Show only the controls relevant to the selected spacetime (rows carry data-m="...").
  function updateMetricUI() {
    const m = ui.metric.value;
    document.querySelectorAll("#panel .row[data-m]").forEach((row) => {
      row.style.display = row.getAttribute("data-m").split(",").includes(m) ? "" : "none";
    });
  }

  // Render on demand: only redraw when something changes. `dirty` is set by any
  // interaction; the loop otherwise idles so the GPU isn't pegged on a still frame.
  let dirty = true;
  const invalidate = () => { dirty = true; };
  let overlayOpen = false;          // when a writeup is shown, pause the sim behind it
  function syncLabels() {
    $("massv").textContent = (+ui.mass.value).toFixed(2);
    $("spinv").textContent = (+ui.spin.value).toFixed(2);
    $("wrv").textContent = (+ui.wr.value).toFixed(1);
    $("wav").textContent = (+ui.wa.value).toFixed(1);
    $("wwv").textContent = (+ui.ww.value).toFixed(2);
    $("rinv").textContent = (+ui.rin.value).toFixed(1);
    $("routv").textContent = (+ui.rout.value).toFixed(1);
    $("dthickv").textContent = (+ui.dthick.value).toFixed(2);
    $("dbrightv").textContent = (+ui.dbright.value).toFixed(1);
    $("dnoisev").textContent = (+ui.dnoise.value).toFixed(1);
    $("dswirlv").textContent = (+ui.dswirl.value).toFixed(1);
    $("stepsv").textContent = ui.steps.value;
    $("ssv").textContent = (+ui.ss.value).toFixed(2);
    $("resv").textContent = (+ui.res.value).toFixed(1) + "x";
    $("expv").textContent = (+ui.exp.value).toFixed(1);
    $("bloomamtv").textContent = (+ui.bloomamt.value).toFixed(2);
  }
  Object.values(ui).forEach((el) => el.addEventListener("input", () => { syncLabels(); updateMetricUI(); resize(); invalidate(); }));
  syncLabels();
  updateMetricUI();

  // Load a local image as the celestial sphere (equirectangular / 2:1 works best).
  $("imgfile").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      const bmp = await createImageBitmap(file, { colorSpaceConversion: "none" });
      skyTexture.destroy();
      skyTexture = device.createTexture({ size: [bmp.width, bmp.height], format: "rgba8unorm-srgb", usage: TEX_USAGE });
      device.queue.copyExternalImageToTexture({ source: bmp }, { texture: skyTexture }, [bmp.width, bmp.height]);
      bmp.close();
      rebuildSceneBind();
      ui.bg.value = "3"; // switch Background → Image
      invalidate();
    } catch (err) {
      console.error("sky image load failed:", err);
    }
  });

  // Second image = the wormhole's other universe.
  $("imgfile2").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      const bmp = await createImageBitmap(file, { colorSpaceConversion: "none" });
      skyTexture2.destroy();
      skyTexture2 = device.createTexture({ size: [bmp.width, bmp.height], format: "rgba8unorm-srgb", usage: TEX_USAGE });
      device.queue.copyExternalImageToTexture({ source: bmp }, { texture: skyTexture2 }, [bmp.width, bmp.height]);
      bmp.close();
      image2Loaded = 1;
      rebuildSceneBind();
      invalidate();
    } catch (err) {
      console.error("second image load failed:", err);
    }
  });

  // ---- Physics / Engineering writeups (overlay over the sim) ----------------
  const overlayEl = $("overlay"), overlayContent = $("overlay-content");
  const contentCache = {};

  // KaTeX, vendored in src/vendor/katex/ (woff2 fonts included) and loaded lazily the first
  // time a writeup is opened — so equations render offline, no CDN. Only the docs overlay uses
  // it; the simulation itself stays dependency-free.
  const KB = new URL("./vendor/katex/", import.meta.url).href;
  let katexReady = null;
  function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); }); }
  function loadKaTeX() {
    if (katexReady) return katexReady;
    const css = document.createElement("link"); css.rel = "stylesheet"; css.href = KB + "katex.min.css"; document.head.appendChild(css);
    katexReady = loadScript(KB + "katex.min.js").then(() => loadScript(KB + "auto-render.min.js"));
    return katexReady;
  }
  async function typeset(el) {
    try {
      await loadKaTeX();
      window.renderMathInElement(el, {
        delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
        throwOnError: false,
      });
    } catch (e) { /* offline / CDN blocked → equations fall back to readable source text */ }
  }

  async function showView(view) {
    document.querySelectorAll("#nav span[data-view]").forEach((s) => s.classList.toggle("active", s.dataset.view === view));
    if (view === "sim") { overlayEl.style.display = "none"; overlayOpen = false; invalidate(); return; }
    if (!contentCache[view]) {
      contentCache[view] = await fetch(new URL(`./content/${view}.html`, import.meta.url))
        .then((r) => r.text()).catch(() => "<p>Could not load this section.</p>");
    }
    overlayContent.innerHTML = contentCache[view];
    overlayEl.scrollTop = 0;
    overlayEl.style.display = "block";
    overlayOpen = true;
    typeset(overlayContent);
  }
  document.querySelectorAll("#nav span[data-view]").forEach((s) => s.addEventListener("click", () => showView(s.dataset.view)));
  overlayEl.addEventListener("click", (e) => { if (e.target === overlayEl || e.target.id === "overlay-close") showView("sim"); });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && overlayOpen) showView("sim"); });

  // ---- pointer interaction -------------------------------------------------
  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    cam.yaw -= (e.clientX - lx) * 0.006;
    cam.pitch += (e.clientY - ly) * 0.006;
    cam.pitch = Math.max(-1.48, Math.min(1.48, cam.pitch));
    lx = e.clientX; ly = e.clientY;
    invalidate();
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist = Math.max(6, Math.min(120, cam.dist * Math.exp(e.deltaY * 0.0009)));
    invalidate();
  }, { passive: false });

  // ---- resize --------------------------------------------------------------
  function resize() {
    const scale = +ui.res.value * Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(innerWidth * scale));
    const h = Math.max(1, Math.floor(innerHeight * scale));
    if (canvas.width !== w || canvas.height !== h || !texHDR) {
      canvas.width = w; canvas.height = h;
      rebuildTargets(w, h);   // offscreen HDR + bloom buffers track the render size
    }
  }
  addEventListener("resize", () => { resize(); invalidate(); });
  resize();

  // ---- render loop ---------------------------------------------------------
  const fpsEl = $("fps");
  let t0 = performance.now(), frames = 0, last = t0, lastDraw = 0, frozenTime = 0, idleShown = false;
  const ANIM_FPS = 30;
  // Re-anchor the clock when animation is toggled on, so the disk phase continues from where
  // it was frozen instead of jumping to (now − pageload).
  ui.anim.addEventListener("change", () => { if (ui.anim.checked) t0 = performance.now() - frozenTime * 1000; });
  function frame(now) {
    if (deviceLost) return;                  // stop submitting to a dead device
    if (overlayOpen) { requestAnimationFrame(frame); return; }  // paused behind a writeup
    // Render only when needed: on interaction (dirty), or — if animating — at a
    // capped rate. Otherwise idle, so the GPU isn't redrawing a static image.
    const animate = ui.anim.checked;
    const due = animate && (now - lastDraw) >= 1000 / ANIM_FPS;
    if (!dirty && !due) {
      if (!idleShown) { fpsEl.textContent = "idle · " + canvas.width + "x" + canvas.height; idleShown = true; }
      requestAnimationFrame(frame); return;
    }
    idleShown = false;

    const time = animate ? (now - t0) / 1000 : frozenTime;
    frozenTime = time;

    const cp = cam.pitch, cy = cam.yaw;
    const pos = [
      cam.dist * Math.cos(cp) * Math.cos(cy),
      cam.dist * Math.cos(cp) * Math.sin(cy),
      cam.dist * Math.sin(cp),
    ];
    const fwd = norm([-pos[0], -pos[1], -pos[2]]);
    const right = norm(cross(fwd, [0, 0, 1]));
    const up = cross(right, fwd);
    const M = +ui.mass.value;
    const metric = +ui.metric.value;
    const spin = metric === 2 ? (+ui.spin.value) * M : 0; // a = (a/M)·M, z-axis
    const tanHalf = Math.tan(cam.fov / 2);
    const aspect = canvas.width / canvas.height;
    // Escape sphere must always enclose the camera, or rays "escape" on step 0 and
    // the hole vanishes. Far-field step cap scales with it (flat space → big steps OK).
    const escapeR = Math.max(60, cam.dist * 1.5);
    const hmax = Math.max(4, escapeR * 0.05);

    u.set([pos[0], pos[1], pos[2], tanHalf], 0);
    u.set([right[0], right[1], right[2], aspect], 4);
    u.set([up[0], up[1], up[2], M], 8);
    u.set([fwd[0], fwd[1], fwd[2], time], 12);
    const rIn = (+ui.rin.value) * M, rOut = Math.max((+ui.rout.value) * M, rIn + 0.5);
    u.set([rIn, rOut, escapeR, spin], 16);                                  // rIn,rOut,escapeR,spin
    u.set([+ui.steps.value, +ui.ss.value, metric, ui.disk.checked ? 1 : 0], 20);
    u.set([+ui.bg.value, +ui.exp.value, 0.01, hmax], 24);
    u.set([ui.doppler.checked ? 1 : 0, +ui.dbright.value, (+ui.dthick.value) * M, 0], 28); // dopplerOn, diskBrightness, halfThickness
    u.set([+ui.wr.value, (+ui.wa.value) * 0.5, +ui.ww.value, image2Loaded], 32);            // wormhole: ρ, a, lensing, image2Loaded
    u.set([+ui.dnoise.value, +ui.dswirl.value, 0, 0], 36);                                  // disk: noiseScale, swirlSpeed
    device.queue.writeBuffer(ubuf, 0, u);
    device.queue.writeBuffer(compBuf, 0, new Float32Array([ui.bloom.checked ? +ui.bloomamt.value : 0, 0, 0, 0]));

    const target = (view) => ({ colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    const enc = device.createCommandEncoder();
    const draw = (descView, pipe, bg) => {
      const p = enc.beginRenderPass(target(descView));
      p.setPipeline(pipe); p.setBindGroup(0, bg); p.draw(3); p.end();
    };
    draw(texHDR.createView(), scenePipeline, sceneBind);                      // ray-trace → HDR
    for (let i = 0; i < 6; i++) draw(blurTex[i].createView(), blurPipeline, blurBinds[i]); // 3-octave bloom pyramid
    draw(ctx.getCurrentTexture().createView(), compositePipeline, compBind);  // composite + ACES tonemap
    device.queue.submit([enc.finish()]);

    dirty = false;
    lastDraw = now;
    frames++;
    if (now - last > 400) {
      const fps = (frames * 1000 / (now - last)).toFixed(0);
      fpsEl.textContent = fps + " fps · " + canvas.width + "x" + canvas.height;
      frames = 0; last = now;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
