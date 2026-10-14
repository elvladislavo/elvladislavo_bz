/* ═══════════════════════════════════════════════════════════════════════
   elvladislavo — Belousov–Zhabotinsky background
   ───────────────────────────────────────────────────────────────────────
   A three-species reaction model run on the GPU: black ground, deep red
   bodies, pale fronts along their leading edge.

     a' = a + a(αb − γc)
     b' = b + b(βc − αa)
     c' = c + c(γa − βb)

   Each species is the 3×3 average of its neighbourhood, so the lattice
   organises itself into travelling fronts and spirals from noise alone.
   α=1.1, β=1.2, γ=1.2 were swept numerically: they hold their structure
   over tens of thousands of steps instead of collapsing into the
   absorbing a=0, b=0, c=1 state.

   The lattice is deliberately coarse — a few dozen cells across the
   viewport — so the structures read large, and it advances about twice a
   second. Painting interpolates between the last two states, so the
   motion stays smooth while being far too slow to pull the eye away from
   the text.

   The canvas sits behind everything, never takes pointer events, sleeps
   when the tab is hidden, and falls back to a small CPU lattice where
   WebGL is missing. With reduced motion it settles to a still frame.
   ═══════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  var host = document.getElementById('bz');
  if (!host) return;

  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var FPS       = 24;      // painting cadence
  var STEP_MS   = 820;     // one reaction step per this many ms (~14 px/s)
  var SEED_STEPS = 320;    // warm-up so the first frame already has structure
  var NUCLEUS_MS = 42000;  // a fresh spiral centre now and then

  /* Palette. Black ground rather than ink so the red reads as deep as the
     reference, with the site's own red and bone on top of it. */
  var GROUND = [0.000, 0.000, 0.000];
  var RED    = [0.690, 0.063, 0.035];
  var REDD   = [0.330, 0.024, 0.016];
  var BONE   = [1.000, 0.945, 0.933];

  var canvas = document.createElement('canvas');
  canvas.className = 'bz__c';
  host.appendChild(canvas);

  /* Lattice size. Cells are tens of pixels across, which is what makes the
     spirals read as large shapes instead of filigree. */
  function grid() {
    var w = window.innerWidth || 1, h = window.innerHeight || 1;
    var gw = Math.max(38, Math.min(118, Math.round(w / 11)));
    return { w: gw, h: Math.max(18, Math.round(gw * h / w)) };
  }

  function noise(n) {
    var d = new Uint8Array(n * 4);
    for (var i = 0; i < n; i++) {
      d[i * 4]     = Math.random() * 255;
      d[i * 4 + 1] = Math.random() * 255;
      d[i * 4 + 2] = Math.random() * 255;
      d[i * 4 + 3] = 255;
    }
    return d;
  }

  /* ── GPU path ───────────────────────────────────────────────────────── */

  var VERT =
    'attribute vec2 p;varying vec2 uv;' +
    'void main(){uv=p*0.5+0.5;gl_Position=vec4(p,0.0,1.0);}';

  var STEP =
    'precision highp float;varying vec2 uv;' +
    'uniform sampler2D prev;uniform vec2 texel;' +
    'void main(){' +
    ' vec3 s=vec3(0.0);' +
    ' for(int i=-1;i<=1;i++){for(int j=-1;j<=1;j++){' +
    '  s+=texture2D(prev,uv+vec2(float(i),float(j))*texel).rgb;}}' +
    ' s/=9.0;' +
    ' float a=s.r+s.r*(1.10*s.g-1.20*s.b);' +
    ' float b=s.g+s.g*(1.20*s.b-1.10*s.r);' +
    ' float c=s.b+s.b*(1.20*s.r-1.20*s.g);' +
    ' gl_FragColor=vec4(clamp(vec3(a,b,c),0.0,1.0),1.0);}';

  /* Painting does three things: smooth the upsample of a very coarse
     lattice, cross-fade the two most recent states, and map the result
     onto the palette. */
  var PAINT =
    'precision highp float;varying vec2 uv;' +
    'uniform sampler2D cur,old;uniform vec2 texel;uniform float mixT;' +
    'uniform vec3 ground,red,redd,bone;' +
    /* Hardware bilinear leaves visible creases on a lattice this coarse,
       so the fractional part is smoothstepped before sampling. */
    'vec2 smoothUV(vec2 q){' +
    ' vec2 p=q/texel-0.5; vec2 f=fract(p); f=f*f*(3.0-2.0*f);' +
    ' return (floor(p)+0.5+f)*texel;}' +
    'float field(vec2 q){' +
    ' vec2 s=smoothUV(q);' +
    ' float v=mix(texture2D(old,s).g,texture2D(cur,s).g,mixT);' +
    ' return v;}' +
    'void main(){' +
    ' float v=field(uv);' +
    /* soften the edge a little in screen space */
    ' float e=0.6; vec2 ex=vec2(texel.x*e,0.0), ey=vec2(0.0,texel.y*e);' +
    ' v=0.40*v' +
    '  +0.10*(field(uv+ex)+field(uv-ex)+field(uv+ey)+field(uv-ey))' +
    '  +0.05*(field(uv+ex+ey)+field(uv-ex+ey)+field(uv+ex-ey)+field(uv-ex-ey));' +
    /* Red fills the body of a wave; the pale front is a narrow ridge
       riding its boundary, not the whole plateau. */
    /* The body ramps slowly from black into a deep red and only brightens
       close to the front, which is where the pale ridge rides. */
    ' vec3 col=mix(ground,redd,smoothstep(0.08,0.52,v));' +
    ' col=mix(col,red,smoothstep(0.50,0.70,v));' +
    ' float ridge=smoothstep(0.28,0.50,v)*(1.0-smoothstep(0.54,0.82,v));' +
    ' col=mix(col,bone,ridge*0.92);' +
    ' gl_FragColor=vec4(col,1.0);}';

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  }

  function program(gl, fs) {
    var v = compile(gl, gl.VERTEX_SHADER, VERT);
    var f = compile(gl, gl.FRAGMENT_SHADER, fs);
    if (!v || !f) return null;
    var p = gl.createProgram();
    gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
    return gl.getProgramParameter(p, gl.LINK_STATUS) ? p : null;
  }

  function startGL() {
    var opts = { alpha: false, antialias: false, depth: false, stencil: false,
                 preserveDrawingBuffer: false, powerPreference: 'low-power' };
    var gl = canvas.getContext('webgl', opts) || canvas.getContext('experimental-webgl', opts);
    if (!gl) return null;

    var pStep = program(gl, STEP), pPaint = program(gl, PAINT);
    if (!pStep || !pPaint) return null;

    var quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

    function bindQuad(prog) {
      var loc = gl.getAttribLocation(prog, 'p');
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    }

    var tex = [], fbo = [], g = { w: 0, h: 0 }, cur = 0;

    function makeTarget(i, data) {
      if (!tex[i]) { tex[i] = gl.createTexture(); fbo[i] = gl.createFramebuffer(); }
      gl.bindTexture(gl.TEXTURE_2D, tex[i]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      /* WebGL1 only allows REPEAT on power-of-two textures; the lattice is
         sized to the viewport, so it has to clamp. */
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, g.w, g.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data || null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex[i], 0);
    }

    function seed() {
      g = grid();
      makeTarget(0, noise(g.w * g.h));
      makeTarget(1, null);
      cur = 0;
      for (var i = 0; i < SEED_STEPS; i++) step();
    }

    function step() {
      var src = cur, dst = 1 - cur;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[dst]);
      gl.viewport(0, 0, g.w, g.h);
      gl.useProgram(pStep);
      bindQuad(pStep);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex[src]);
      gl.uniform1i(gl.getUniformLocation(pStep, 'prev'), 0);
      gl.uniform2f(gl.getUniformLocation(pStep, 'texel'), 1 / g.w, 1 / g.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      cur = dst;
    }

    /* Drop one excited cell into the current state. Rare, and the only
       thing standing between the lattice and an eventual flat field. */
    function nucleus() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[cur]);
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(Math.floor(Math.random() * g.w), Math.floor(Math.random() * g.h), 1, 1);
      gl.clearColor(0.35, 1.0, 0.25, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.SCISSOR_TEST);
    }

    function paint(t) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(pPaint);
      bindQuad(pPaint);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex[cur]);
      gl.uniform1i(gl.getUniformLocation(pPaint, 'cur'), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, tex[1 - cur]);
      gl.uniform1i(gl.getUniformLocation(pPaint, 'old'), 1);
      gl.uniform1f(gl.getUniformLocation(pPaint, 'mixT'), t);
      gl.uniform2f(gl.getUniformLocation(pPaint, 'texel'), 1 / g.w, 1 / g.h);
      gl.uniform3fv(gl.getUniformLocation(pPaint, 'ground'), GROUND);
      gl.uniform3fv(gl.getUniformLocation(pPaint, 'red'), RED);
      gl.uniform3fv(gl.getUniformLocation(pPaint, 'redd'), REDD);
      gl.uniform3fv(gl.getUniformLocation(pPaint, 'bone'), BONE);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    return { seed: seed, step: step, paint: paint, nucleus: nucleus };
  }

  /* ── CPU fallback ───────────────────────────────────────────────────── */

  function startCPU() {
    var ctx = canvas.getContext('2d');
    if (!ctx) return null;
    var g, a, b, c, a2, b2, c2, bOld, img;

    function seed() {
      var full = grid();
      g = { w: full.w, h: full.h };
      var n = g.w * g.h;
      a = new Float32Array(n); b = new Float32Array(n); c = new Float32Array(n);
      a2 = new Float32Array(n); b2 = new Float32Array(n); c2 = new Float32Array(n);
      bOld = new Float32Array(n);
      for (var i = 0; i < n; i++) { a[i] = Math.random(); b[i] = Math.random(); c[i] = Math.random(); }
      canvas.width = g.w; canvas.height = g.h;
      img = ctx.createImageData(g.w, g.h);
      for (var s = 0; s < 160; s++) step();
    }

    function step() {
      var w = g.w, h = g.h;
      bOld.set(b);
      for (var y = 0; y < h; y++) {
        var yu = ((y - 1 + h) % h) * w, yd = ((y + 1) % h) * w, yc = y * w;
        for (var x = 0; x < w; x++) {
          var xl = (x - 1 + w) % w, xr = (x + 1) % w;
          var sa = a[yu+xl]+a[yu+x]+a[yu+xr]+a[yc+xl]+a[yc+x]+a[yc+xr]+a[yd+xl]+a[yd+x]+a[yd+xr];
          var sb = b[yu+xl]+b[yu+x]+b[yu+xr]+b[yc+xl]+b[yc+x]+b[yc+xr]+b[yd+xl]+b[yd+x]+b[yd+xr];
          var sc = c[yu+xl]+c[yu+x]+c[yu+xr]+c[yc+xl]+c[yc+x]+c[yc+xr]+c[yd+xl]+c[yd+x]+c[yd+xr];
          sa /= 9; sb /= 9; sc /= 9;
          var i = yc + x;
          var na = sa + sa * (1.10 * sb - 1.20 * sc);
          var nb = sb + sb * (1.20 * sc - 1.10 * sa);
          var nc = sc + sc * (1.20 * sa - 1.20 * sb);
          a2[i] = na < 0 ? 0 : na > 1 ? 1 : na;
          b2[i] = nb < 0 ? 0 : nb > 1 ? 1 : nb;
          c2[i] = nc < 0 ? 0 : nc > 1 ? 1 : nc;
        }
      }
      var t;
      t = a; a = a2; a2 = t; t = b; b = b2; b2 = t; t = c; c = c2; c2 = t;
    }

    function nucleus() {
      var i = Math.floor(Math.random() * g.w * g.h);
      a[i] = 0.35; b[i] = 1; c[i] = 0.25;
    }

    function smooth(t, e0, e1) {
      var x = (t - e0) / (e1 - e0);
      x = x < 0 ? 0 : x > 1 ? 1 : x;
      return x * x * (3 - 2 * x);
    }

    function paint(mt) {
      var d = img.data, n = g.w * g.h;
      for (var i = 0; i < n; i++) {
        var v = bOld[i] + (b[i] - bOld[i]) * mt, k = i * 4;
        var t1 = smooth(v, 0.08, 0.52), t2 = smooth(v, 0.50, 0.70);
        var t3 = smooth(v, 0.28, 0.50) * (1 - smooth(v, 0.54, 0.82)) * 0.92;
        for (var ch = 0; ch < 3; ch++) {
          var col = GROUND[ch] + (REDD[ch] - GROUND[ch]) * t1;
          col = col + (RED[ch] - col) * t2;
          col = col + (BONE[ch] - col) * t3;
          d[k + ch] = col * 255;
        }
        d[k + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }

    return { seed: seed, step: step, paint: paint, nucleus: nucleus, cpu: true };
  }

  /* ── Driver ─────────────────────────────────────────────────────────── */

  var engine = startGL() || startCPU();
  if (!engine) { host.classList.add('bz--off'); return; }

  /* The lattice cells are tens of pixels wide and the fronts are soft, so
     the canvas is drawn below screen resolution and stretched by CSS. */
  var RENDER_SCALE = 0.5;

  function resize() {
    if (engine.cpu) { engine.seed(); engine.paint(1); return; }
    canvas.width  = Math.max(1, Math.round(window.innerWidth  * RENDER_SCALE));
    canvas.height = Math.max(1, Math.round(window.innerHeight * RENDER_SCALE));
    engine.seed();
    engine.paint(1);
  }

  var raf = 0, running = false;
  var lastPaint = 0, lastStep = 0, lastNucleus = 0;

  function loop(now) {
    raf = 0;
    if (!running) return;

    if (now - lastStep >= STEP_MS) {
      engine.step();
      lastStep = now;
      if (now - lastNucleus >= NUCLEUS_MS) { engine.nucleus(); lastNucleus = now; }
    }
    if (now - lastPaint >= 1000 / FPS) {
      lastPaint = now;
      var t = Math.min(1, (now - lastStep) / STEP_MS);
      engine.paint(t);
    }
    raf = requestAnimationFrame(loop);
  }

  function play() {
    if (running || REDUCED) return;
    running = true;
    lastStep = lastPaint = lastNucleus = performance.now();
    if (!raf) raf = requestAnimationFrame(loop);
  }

  function pause() {
    running = false;
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
  }

  resize();
  host.classList.add('is-on');

  var rt = 0;
  window.addEventListener('resize', function () {
    clearTimeout(rt);
    rt = setTimeout(resize, 220);
  }, { passive: true });

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) pause(); else play();
  });

  canvas.addEventListener('webglcontextlost', function (e) {
    e.preventDefault(); pause();
  });
  canvas.addEventListener('webglcontextrestored', function () {
    engine = startGL() || startCPU();
    if (engine) { resize(); play(); }
  });

  play();
})();
