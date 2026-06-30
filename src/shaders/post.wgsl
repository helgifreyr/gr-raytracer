struct VO { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VO;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv = p[vi] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);  // clip → texture uv (Y flipped)
  return o;
}

// --- separable Gaussian blur, with an optional bright-pass on the first (H) pass ---
struct Blur { dir : vec2<f32>, threshold : f32, pad : f32 };
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var<uniform> bl : Blur;
fn bright(c : vec3<f32>) -> vec3<f32> {
  if (bl.threshold >= 0.0) { return max(c - vec3<f32>(bl.threshold), vec3<f32>(0.0)); }
  return c;
}
@fragment
fn fsBlur(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let w = array<f32, 5>(0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);
  var col = bright(textureSampleLevel(src, samp, uv, 0.0).rgb) * w[0];
  for (var i = 1; i < 5; i = i + 1) {
    let o = bl.dir * f32(i);
    col = col + bright(textureSampleLevel(src, samp, uv + o, 0.0).rgb) * w[i];
    col = col + bright(textureSampleLevel(src, samp, uv - o, 0.0).rgb) * w[i];
  }
  return vec4<f32>(col, 1.0);
}

// --- composite: scene HDR + bloom, then tonemap + gamma to the swapchain ---
struct Comp { bloom : f32, pad0 : f32, pad1 : f32, pad2 : f32 };
@group(0) @binding(0) var csamp : sampler;
@group(0) @binding(1) var hdrTex : texture_2d<f32>;
@group(0) @binding(2) var bloomTex : texture_2d<f32>;
@group(0) @binding(3) var<uniform> cp : Comp;

// Hue-preserving ACES filmic tonemap (adapted from the cuneus black-hole shader,
// Enes Altun, MIT — github.com/altunenes/cuneus). The ACES curve is applied to the
// luminance *peak* and the chroma ratio is preserved, so bright disk highlights roll
// off to white smoothly without the hue shifts plain per-channel Reinhard produces.
fn acesScalar(x : f32) -> f32 {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
fn tonemapHue(c : vec3<f32>) -> vec3<f32> {
  let peak = max(max(c.r, c.g), max(c.b, 1e-5));
  let ratio = c / peak;                                     // chroma, preserved
  let tp = acesScalar(peak);                                // tonemap the luminance peak
  let desat = clamp(pow(tp, 4.0) * 0.55, 0.0, 0.55);        // wash hot highlights toward white
  return mix(ratio, vec3<f32>(1.0), desat) * tp;
}
@fragment
fn fsComposite(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  var col = textureSampleLevel(hdrTex, csamp, uv, 0.0).rgb;
  col = col + textureSampleLevel(bloomTex, csamp, uv, 0.0).rgb * cp.bloom;
  col = tonemapHue(col);                                    // hue-preserving ACES
  col = pow(col, vec3<f32>(1.0 / 2.2));                     // gamma
  return vec4<f32>(col, 1.0);
}
