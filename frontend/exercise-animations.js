/*
 * PhysioAI - exercise tutorial animations
 * =======================================
 *
 * One articulated figure, driven by keyframed joint angles, plus a per-exercise
 * script. No dependencies, no image/video assets, no network.
 *
 * Most exercises are drawn in the sagittal (side) plane, where knee flexion is
 * the visible quantity. The squats are drawn in the coronal (front) plane
 * instead, because their coaching cue is knee tracking - whether the knees stay
 * out over the feet or collapse inward - and a side view hides that entirely.
 * Set `view: 'front'` on a spec's base pose to opt in; see solveFront.
 *
 * Why parametric rather than recorded clips or a licensed clip library:
 * the prescription in this app is angle-capped per patient (see
 * clinical_logic._cap_exercises). A recorded clip always shows textbook range,
 * so a patient capped at 60 degrees would be shown a 120-degree squat. Here the
 * cap is an input - pass angleLimit and the figure bends exactly that far and
 * no further, and the knee marker reports the live angle the same way the
 * webcam tracker does.
 *
 * Usage
 * -----
 *   const anim = PhysioAnimations.mount(el, {
 *     name: "Heel Slides",   // protocol name; matched slug-insensitively
 *     angleLimit: 90,        // this patient's ceiling for THIS exercise
 *     capped: false,         // true -> note says the range was reduced
 *   });
 *   anim.setLimit(60);  anim.pause();  anim.play();  anim.destroy();
 *
 * Geometry conventions
 * --------------------
 * Degrees, screen coords (y grows downward), measured clockwise from +x.
 * The figure faces +x (to the right).
 *
 *   view  'side' (default) or 'front'. See solveFront for the frontal model.
 *   rot   global rotation of the body: 0 standing, -90 lying, -55 reclined.
 *         Side view only - head-on there is nothing to rotate about.
 *   m     mirror: +1 normal, -1 prone (face-down), so knee flexion lifts the
 *         heel instead of dropping it.
 *   h/k/a hip flexion, knee flexion, ankle angle, per leg. All zero = anatomic
 *         standing. k is always POSITIVE flexion - that is what angleLimit
 *         clamps and what the tracker measures.
 *
 * Forward kinematics:
 *   torsoDir = rot - 90 + m*lean
 *   thighDir = rot + 90 - m*h
 *   shankDir = thighDir + m*k
 *   footDir  = shankDir - m*90 + m*a
 */
(function (global) {
  'use strict';

  var SEG = { torso: 40, thigh: 34, shank: 32, foot: 13, head: 8.5, upper: 17, fore: 16 };
  var GROUND = 144;
  var NS = 'http://www.w3.org/2000/svg';
  var DEG = '°';

  /* ------------------------------------------------------------------ math */

  function rad(d) { return d * Math.PI / 180; }
  function deg(r) { return r * 180 / Math.PI; }
  function P(x, y) { return { x: x, y: y }; }
  function go(p, ang, len) {
    return { x: p.x + Math.cos(rad(ang)) * len, y: p.y + Math.sin(rad(ang)) * len };
  }
  function lerp(a, b, u) { return a + (b - a) * u; }
  function ease(u) { return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function pick(a, b, c) {
    if (a !== undefined && a !== null) return a;
    if (b !== undefined && b !== null) return b;
    return c;
  }

  /*
   * Two-link IK: what hip and knee angles put the ankle on `target`?
   * Used by the exercises whose foot path is fixed by equipment (bike crank,
   * leg-press plate) rather than by joint angles. `kneeUp` picks the branch
   * with the knee above the hip-to-ankle line, which is the seated posture.
   * Returns the same {h,k,a} shape a hand-written keyframe would.
   */
  function ik2(pelvis, target, rot, m, kneeUp, l1, l2) {
    l1 = l1 || SEG.thigh;
    l2 = l2 || SEG.shank;
    var dx = target.x - pelvis.x, dy = target.y - pelvis.y;
    var d = clamp(Math.sqrt(dx * dx + dy * dy), Math.abs(l1 - l2) + 0.5, l1 + l2 - 0.5);
    var phi = deg(Math.atan2(dy, dx));
    var a1 = deg(Math.acos(clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1)));
    var sgn = kneeUp ? -1 : 1;
    var thighDir = phi + sgn * a1;
    var interior = deg(Math.acos(clamp((l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2), -1, 1)));
    var kneeFlex = 180 - interior;
    var shankDir = thighDir - sgn * kneeFlex;
    return { h: (rot + 90 - thighDir) / m, k: (shankDir - thighDir) / m, a: 0 };
  }

  /* ------------------------------------------------------------ pose model */

  function leg(o) {
    o = o || {};
    return { h: o.h || 0, k: o.k || 0, a: o.a || 0 };
  }

  // Expand a sparse keyframe against the spec's base pose.
  function norm(kf, base, twoLeg) {
    return {
      t: kf.t,
      rot: pick(kf.rot, base.rot, 0),
      m: pick(kf.m, base.m, 1),
      x: pick(kf.x, base.x, 100),
      y: pick(kf.y, base.y, 78),
      lean: pick(kf.lean, base.lean, 0),
      arm: kf.arm || base.arm || [10, 12],
      near: leg(kf.near || base.near),
      far: twoLeg ? leg(kf.far || base.far) : null,
      ground: pick(kf.ground, base.ground, null),
      view: pick(kf.view, base.view, 'side'),
      stance: pick(kf.stance, base.stance, 17),
      label: pick(kf.label, base.label, '')
    };
  }

  function legLerp(a, b, u) {
    if (!a || !b) return a || b;
    return { h: lerp(a.h, b.h, u), k: lerp(a.k, b.k, u), a: lerp(a.a, b.a, u) };
  }

  function poseLerp(A, B, u) {
    return {
      rot: lerp(A.rot, B.rot, u), m: A.m,
      x: lerp(A.x, B.x, u), y: lerp(A.y, B.y, u),
      lean: lerp(A.lean, B.lean, u),
      arm: [lerp(A.arm[0], B.arm[0], u), lerp(A.arm[1], B.arm[1], u)],
      near: legLerp(A.near, B.near, u),
      far: A.far && B.far ? legLerp(A.far, B.far, u) : null,
      stance: lerp(A.stance, B.stance, u),
      ground: A.ground,      // stepped, not interpolated
      view: A.view,          // stepped, not interpolated
      label: A.label         // stepped, not interpolated
    };
  }

  // Sample the keyframe track at normalised cycle time t in [0,1].
  function sample(kfs, t) {
    for (var i = 0; i < kfs.length - 1; i++) {
      if (t >= kfs[i].t && t <= kfs[i + 1].t) {
        var span = kfs[i + 1].t - kfs[i].t;
        return poseLerp(kfs[i], kfs[i + 1], span > 0 ? ease((t - kfs[i].t) / span) : 0);
      }
    }
    return poseLerp(kfs[kfs.length - 1], kfs[kfs.length - 1], 0);
  }

  /*
   * Resolve a pose into screen points. `limit` is the patient's knee ceiling
   * and is applied HERE, before FK, so every downstream drawing - and the angle
   * readout - sees the clamped value rather than the textbook one.
   */
  function solve(pose, limit, bilateral) {
    if (pose.view === 'front') return solveFront(pose, limit, bilateral);
    var m = pose.m, rot = pose.rot;
    var near = { h: pose.near.h, k: pose.near.k, a: pose.near.a };
    var far = pose.far ? { h: pose.far.h, k: pose.far.k, a: pose.far.a } : null;
    if (limit != null) {
      near.k = Math.min(near.k, limit);
      if (far && bilateral) far.k = Math.min(far.k, limit);
    }

    var pelvis = P(pose.x, pose.y);
    var torsoDir = rot - 90 + m * pose.lean;
    var shoulder = go(pelvis, torsoDir, SEG.torso);
    var head = go(shoulder, torsoDir, SEG.head + 2.5);

    function chain(L) {
      if (!L) return null;
      var td = rot + 90 - m * L.h;
      var knee = go(pelvis, td, SEG.thigh);
      var sd = td + m * L.k;
      var ankle = go(knee, sd, SEG.shank);
      var toe = go(ankle, sd - m * 90 + m * L.a, SEG.foot);
      return { knee: knee, ankle: ankle, toe: toe, flex: L.k };
    }

    var upDir = torsoDir + 180 - m * pose.arm[0];
    var elbow = go(shoulder, upDir, SEG.upper);
    var hand = go(elbow, upDir - m * pose.arm[1], SEG.fore);

    var pts = {
      pelvis: pelvis, shoulder: shoulder, head: head, elbow: elbow, hand: hand,
      near: chain(near), far: chain(far),
      torsoDir: torsoDir, flex: near.k, textbookFlex: pose.near.k
    };

    // Drop the figure so the chosen foot rests on the floor. This is what keeps
    // squats, lunges and gait from floating or sinking as the knees bend, and
    // it produces the pelvis bob of a walking cycle for free.
    if (pose.ground) {
      var cands = [];
      if (pose.ground !== 'far' && pts.near) cands.push(pts.near.ankle.y, pts.near.toe.y);
      if (pose.ground !== 'near' && pts.far) cands.push(pts.far.ankle.y, pts.far.toe.y);
      if (cands.length) shiftY(pts, GROUND - Math.max.apply(null, cands));
    }
    return pts;
  }

  /*
   * Frontal-plane solver, for the exercises whose coaching cue lives in the
   * coronal view rather than the sagittal one. A squat seen from the side hides
   * the thing that matters - whether the knees track out over the feet or
   * collapse inward - so both squats are drawn face-on.
   *
   * The vertical drop of each joint is taken from the SAME sagittal chain the
   * side view uses and then projected onto the frontal plane, so the two views
   * agree on depth exactly: bending the knee foreshortens the leg on screen,
   * which is how squat depth reads from the front. Only the lateral positions
   * are new - hips at a fixed width, feet at stance width, knees tracking
   * outward as flexion increases.
   */
  function solveFront(pose, limit, bilateral) {
    var cx = pose.x, py = pose.y, hipW = 10, stance = pose.stance;
    var nearK = limit != null ? Math.min(pose.near.k, limit) : pose.near.k;
    var farK = pose.far
      ? (limit != null && bilateral ? Math.min(pose.far.k, limit) : pose.far.k)
      : 0;

    var all = [];
    function pp(x, y) { var p = P(x, y); all.push(p); return p; }

    // side: +1 draws to screen-right, -1 to screen-left.
    function limb(L, k, side) {
      if (!L) return null;
      var td = 90 - L.h, sd = td + k;
      var kneeDy = SEG.thigh * Math.sin(rad(td));
      var ankleDy = kneeDy + SEG.shank * Math.sin(rad(sd));
      var splay = 2 + 0.055 * k;                    // knees track out with depth
      var kneeX = cx + side * (hipW + (stance - hipW) * 0.55 + splay);
      var ankle = pp(cx + side * stance, py + ankleDy);
      return {
        hip: pp(cx + side * hipW, py),
        knee: pp(kneeX, py + kneeDy),
        ankle: ankle,
        toe: pp(ankle.x + side * 7, ankle.y + 2.5),
        flex: k
      };
    }

    // Forward lean is invisible head-on, so it shows as a shortened torso.
    var tl = SEG.torso * Math.cos(rad(pose.lean * 0.7));
    var pelvis = pp(cx, py);
    var shoulder = pp(cx, py - tl);
    var head = pp(cx, py - tl - SEG.head - 2.5);
    var shW = 13, flare = 4 + 0.09 * nearK;

    function arm(side) {
      var s = pp(cx + side * shW, shoulder.y);
      return [s, pp(s.x + side * flare * 0.45, shoulder.y + 16),
        pp(s.x + side * flare, shoulder.y + 30)];
    }
    var armR = arm(1), armL = arm(-1);

    var pts = {
      pelvis: pelvis, shoulder: shoulder, head: head,
      elbow: armR[1], hand: armR[2],
      near: limb(pose.near, nearK, 1),
      far: pose.far ? limb(pose.far, farK, -1) : null,
      torsoDir: -90, flex: nearK, textbookFlex: pose.near.k,
      frontal: {
        shoulderL: armL[0], shoulderR: armR[0],
        hipL: pp(cx - hipW, py), hipR: pp(cx + hipW, py),
        armL: armL, armR: armR
      },
      _all: all
    };

    if (pose.ground) {
      var cands = [];
      if (pts.near) cands.push(pts.near.ankle.y, pts.near.toe.y);
      if (pts.far) cands.push(pts.far.ankle.y, pts.far.toe.y);
      if (cands.length) shiftY(pts, GROUND - Math.max.apply(null, cands));
    }
    return pts;
  }

  function shiftY(pts, dy) {
    // The frontal solver tracks every point it made, so one pass covers the
    // shoulder bar, pelvis bar and both arms as well as the limbs.
    if (pts._all) {
      pts._all.forEach(function (p) { p.y += dy; });
      return;
    }
    ['pelvis', 'shoulder', 'head', 'elbow', 'hand'].forEach(function (k) { pts[k].y += dy; });
    ['near', 'far'].forEach(function (g) {
      if (!pts[g]) return;
      ['knee', 'ankle', 'toe'].forEach(function (k) { pts[g][k].y += dy; });
    });
  }

  /* -------------------------------------------------------------- svg bits */

  function svgEl(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) n.setAttribute(k, attrs[k]);
    return n;
  }
  function rr(n) { return Math.round(n * 10) / 10; }
  function path() {
    var d = '';
    for (var i = 0; i < arguments.length; i++) {
      d += (i ? 'L' : 'M') + rr(arguments[i].x) + ' ' + rr(arguments[i].y);
    }
    return d;
  }

  /* ------------------------------------------------------------ shared css */

  var STYLE_ID = 'pxa-styles';
  var CSS = [
    '.pxa{--pxa-body:var(--navy,#132733);--pxa-muted:#b6c5cd;--pxa-accent:var(--teal,#447f98);',
    '--pxa-prop:#9fb4bf;--pxa-warn:#d97706;--pxa-danger:var(--red,#dc2626);',
    'display:flex;flex-direction:column;gap:6px;}',
    '.pxa-stage{position:relative;background:linear-gradient(180deg,#f6fafb 0%,#e6eef3 100%);',
    'border:1px solid var(--border,#dadee1);border-radius:10px;overflow:hidden;}',
    '.pxa-svg{display:block;width:100%;height:auto;}',
    '.pxa-limb{fill:none;stroke-linecap:round;stroke-linejoin:round;}',
    '.pxa-far{stroke:var(--pxa-muted);stroke-width:5.5;}',
    '.pxa-torso{stroke:var(--pxa-body);stroke-width:8.5;}',
    '.pxa-arm{stroke:var(--pxa-body);stroke-width:4.5;}',
    '.pxa-near{stroke:var(--pxa-accent);stroke-width:7;}',
    '.pxa-head{fill:var(--pxa-body);}',
    '.pxa-knee{fill:#fff;stroke:var(--pxa-accent);stroke-width:2.5;transition:stroke .2s;}',
    '.pxa-knee.warn{stroke:var(--pxa-warn);}',
    '.pxa-knee.danger{stroke:var(--pxa-danger);}',
    '.pxa-prop{stroke:var(--pxa-prop);fill:none;stroke-width:2.5;stroke-linecap:round;}',
    '.pxa-prop-dash{stroke:var(--pxa-prop);fill:none;stroke-width:1.5;stroke-dasharray:4 4;}',
    '.pxa-prop-fill{fill:rgba(68,127,152,.07);stroke:none;}',
    '.pxa-fx{fill:none;stroke:var(--pxa-accent);stroke-width:2;stroke-linecap:round;}',
    '.pxa-fx-soft{fill:none;stroke:var(--pxa-accent);stroke-width:1.5;opacity:.5;}',
    '.pxa-fx-fill{fill:var(--pxa-accent);stroke:none;}',
    '.pxa-water{fill:rgba(98,155,182,.16);stroke:none;}',
    '.pxa-badge{position:absolute;top:8px;right:8px;font-family:var(--mono,monospace);',
    'font-size:10px;font-weight:600;',
    'color:var(--navy,#132733);letter-spacing:.02em;}',
    '.pxa-badge.warn{color:var(--pxa-warn);}',
    '.pxa-badge.danger{color:var(--red,#dc2626);}',
    '.pxa-cap{position:absolute;left:8px;bottom:8px;right:8px;font-family:var(--body,sans-serif);',
    'font-size:11px;font-weight:600;color:var(--slate,#5a7380);}',
    // No box, but it sits on the floor line: a halo in the stage colour keeps
    // the line from running through the letters.
    '.pxa-cap span{display:inline-block;text-shadow:0 0 3px #f6f8f6,0 0 3px #f6f8f6,0 0 6px #f6f8f6;}',
    '.pxa-note{font-family:var(--mono,monospace);font-size:9.5px;letter-spacing:.06em;',
    'text-transform:uppercase;color:var(--slate,#5a7380);}',
    '.pxa-note.capped{color:var(--pxa-warn);}',
    '.pxa-toggle{position:absolute;top:8px;left:8px;width:22px;height:22px;border-radius:5px;',
    'border:1px solid var(--border,#dadee1);background:rgba(255,255,255,.9);cursor:pointer;',
    'font-size:10px;line-height:1;color:var(--navy,#132733);padding:0;}'
  ].join('');

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* --------------------------------------------------------- shared clock */
  /* One rAF drives every mounted animation. Instances scrolled out of view
   * stop being drawn rather than each holding their own frame loop. */

  var instances = [];
  var raf = null;

  function tick(now) {
    raf = null;
    for (var i = 0; i < instances.length; i++) {
      var a = instances[i];
      if (a._playing && a._visible) a._draw(now / 1000);
    }
    if (instances.length) raf = requestAnimationFrame(tick);
  }
  function ensureLoop() {
    if (raf == null && instances.length) raf = requestAnimationFrame(tick);
  }

  /* ------------------------------------------------------------ prop scenes */

  var PROPS = {
    mat: '<path class="pxa-prop" d="M10 100H190"/>',
    floor: '<path class="pxa-prop" d="M6 144H194"/>',
    chair:
      '<path class="pxa-prop-fill" d="M44 84H112V90H44Z"/>' +
      '<path class="pxa-prop" d="M44 84H112 M49 84V120 M106 84V120 M46 84V30 M46 30H52"/>' +
      '<path class="pxa-prop" d="M6 120H194"/>',
    step:
      '<path class="pxa-prop-fill" d="M116 124H188V144H116Z"/>' +
      '<path class="pxa-prop" d="M116 144V124H188V144"/>' +
      '<path class="pxa-prop-dash" d="M186 124V70 M132 70H190"/>' +
      '<path class="pxa-prop" d="M6 144H194"/>',
    stairs:
      '<path class="pxa-prop-fill" d="M26 126H106V150H26Z M106 138H176V150H106Z"/>' +
      '<path class="pxa-prop" d="M26 126H106V138H176V150"/>' +
      '<path class="pxa-prop-dash" d="M36 108H184 M184 108V138"/>' +
      '<path class="pxa-prop" d="M6 150H194"/>',
    wallLeft: '<path class="pxa-prop" d="M24 16V144"/><path class="pxa-prop-dash" d="M100 20V144"/>' + '<path class="pxa-prop" d="M6 144H194"/>',
    wallRight: '<path class="pxa-prop" d="M168 16V144"/><path class="pxa-prop" d="M6 144H194"/>',
    bike:
      '<path class="pxa-prop" d="M74 66H102"/>' +
      '<path class="pxa-prop" d="M88 68L126 104"/>' +
      '<path class="pxa-prop" d="M126 104L140 56 M132 52H160"/>' +
      '<circle class="pxa-prop-dash" cx="126" cy="104" r="12"/>' +
      '<path class="pxa-prop" d="M96 144H160 M126 116V144"/>' +
      '<path class="pxa-prop" d="M6 144H194"/>',
    legPress:
      '<path class="pxa-prop" d="M30 78L74 108"/>' +
      '<path class="pxa-prop" d="M56 118H96"/>' +
      '<path class="pxa-prop-dash" d="M96 92L136 74"/>' +
      '<path class="pxa-prop" d="M6 130H194"/>' +
      '<path class="pxa-prop" d="M44 108V130 M92 118V130"/>',
    pool:
      '<path class="pxa-water" d="M6 52H194V150H6Z"/>' +
      '<path class="pxa-prop" d="M6 52H194"/>' +
      '<path class="pxa-prop" d="M6 144H194"/>',
    ground: '<path class="pxa-prop" d="M6 144H194"/><path class="pxa-prop-dash" d="M6 132H194"/>'
  };

  /* --------------------------------------------------------- gait generator */

  /*
   * A walking cycle as four keyframes per stride. Amplitude scales every joint
   * angle at once: 0.8 reads as a cautious pool stride, 1.0 as a normal walk,
   * 1.7 as a jog. Pelvis height is left to the ground-snap in solve(), which
   * is what gives the cycle its vertical bob without hand-tuning.
   */
  function gait(t0, t1, cycles, amp, label) {
    var A = amp, out = [], n = cycles * 4;
    var seq = [
      { near: { h: 25 * A, k: 6, a: 0 }, far: { h: -16 * A, k: 12 * A, a: -10 * A }, arm: 30 * A },
      { near: { h: 5 * A, k: 14 * A, a: 0 }, far: { h: 6 * A, k: 38 * A, a: 5 }, arm: 2 },
      { near: { h: -16 * A, k: 12 * A, a: -10 * A }, far: { h: 25 * A, k: 6, a: 0 }, arm: -25 * A },
      { near: { h: 6 * A, k: 38 * A, a: 5 }, far: { h: 5 * A, k: 14 * A, a: 0 }, arm: 2 }
    ];
    for (var i = 0; i <= n; i++) {
      var p = seq[i % 4];
      out.push({
        t: t0 + (t1 - t0) * i / n,
        near: p.near, far: p.far,
        arm: [p.arm, 14],
        lean: 4 + 6 * (A - 1),
        ground: 'lowest',
        label: label
      });
    }
    return out;
  }

  /* ------------------------------------------------- equipment-driven specs */

  // Bike: the foot is bolted to a crank, so the leg angles come from IK on the
  // pedal position rather than from hand-written joint angles.
  function bikeFrames(label) {
    var pelvis = P(88, 64), c = P(126, 104), r = 12, n = 8, out = [];
    for (var i = 0; i <= n; i++) {
      var th = 360 * i / n;
      var near = ik2(pelvis, go(c, th, r), 0, 1, true);
      var far = ik2(pelvis, go(c, th + 180, r), 0, 1, true);
      out.push({
        t: i / n, x: 88, y: 64, lean: 40, arm: [92, 8],
        near: near, far: far, label: label
      });
    }
    return out;
  }

  // Leg press: the foot rides a plate on a fixed rail; same IK reasoning.
  function legPressFrames() {
    var pelvis = P(62, 104);
    function at(u, label) {
      var target = P(lerp(98, 132, u), lerp(90, 75, u));
      var L = ik2(pelvis, target, -55, 1, true);
      return { t: 0, x: 62, y: 104, rot: -55, lean: 0, arm: [-65, 45], near: L, far: L, label: label };
    }
    var a = at(0, 'Set the machine so the knee starts at 90' + DEG);
    var b = at(1, 'Press through both heels to full extension');
    var c = at(1, 'Hold the top briefly');
    var d = at(0, 'Return slowly - do not let the stack drop');
    a.t = 0; b.t = 0.42; c.t = 0.55; d.t = 1;
    return [a, b, c, d];
  }

  /* ------------------------------------------------------------- fx helpers */

  function fxCircle(p, r, cls) {
    return '<circle class="' + (cls || 'pxa-fx') + '" cx="' + rr(p.x) + '" cy="' + rr(p.y) + '" r="' + rr(r) + '"/>';
  }
  function fxArrow(from, ang, len) {
    var tip = go(from, ang, len);
    var l = go(tip, ang + 150, 5), rgt = go(tip, ang - 150, 5);
    return '<path class="pxa-fx" d="' + path(from, tip) + '"/>' +
      '<path class="pxa-fx" d="' + path(l, tip, rgt) + '"/>';
  }
  function kneeTrackingGuides(pts) {
    var out = '';
    ['near', 'far'].forEach(function (g) {
      var L = pts[g];
      if (!L) return;
      out += '<path class="pxa-fx-soft" d="M' + rr(L.ankle.x) + ' ' + rr(L.ankle.y) +
        'V' + rr(L.knee.y - 7) + '"/>';
    });
    return out;
  }

  function fxTarget(p, label) {
    return '<circle class="pxa-fx-soft" cx="' + rr(p.x) + '" cy="' + rr(p.y) + '" r="7"/>' +
      (label ? '<text x="' + rr(p.x + 11) + '" y="' + rr(p.y + 3) +
        '" font-family="monospace" font-size="9" fill="currentColor" opacity=".65">' + label + '</text>' : '');
  }

  /* ------------------------------------------------------------ the library */
  /* Names below mirror backend/exercise_protocols.py exactly. Lookup is by
   * slug (lowercased, alphanumerics only), so en-dash vs hyphen and bracketed
   * qualifiers cannot break the match. */

  var SUPINE = { rot: -90, x: 70, y: 88, arm: [0, 0] };
  var PRONE = { rot: -90, m: -1, x: 70, y: 88, arm: [0, 0] };
  var SEATED = { rot: 0, x: 74, y: 88, arm: [18, 14], near: { h: 90, k: 90 }, far: { h: 90, k: 86 } };
  var STAND = { rot: 0, x: 100, y: 78, lean: 4, arm: [8, 12], ground: 'lowest' };

  var SPECS = [

    /* 1 ------------------------------------------------------- Ankle Pumps */
    {
      name: 'Ankle Pumps', dur: 2.6, twoLeg: true, props: PROPS.mat, base: SUPINE,
      cue: 'One point-and-flex cycle is one rep',
      kf: [
        { t: 0, near: { a: 70 }, far: { a: 62 }, label: 'Point the foot away from you' },
        { t: 0.5, near: { a: -45 }, far: { a: -38 }, label: 'Flex the foot back toward you' },
        { t: 1, near: { a: 70 }, far: { a: 62 }, label: 'Point the foot away from you' }
      ],
      fx: function (pts) {
        return fxArrow(pts.near.ankle, -160, 16) + fxArrow(pts.near.ankle, -20, 16);
      }
    },

    /* 2 -------------------------------------------- Quad Sets (Isometric) */
    {
      name: 'Quad Sets (Isometric)', dur: 6, twoLeg: true, props: PROPS.mat, base: SUPINE,
      cue: 'The knee does not move - only the thigh tightens',
      kf: [
        { t: 0, near: { a: -20 }, far: { a: -22 }, label: 'Relax, knee flat on the floor' },
        { t: 0.14, near: { a: -26 }, far: { a: -22 }, label: 'Tighten the thigh - kneecap draws up' },
        { t: 0.82, near: { a: -26 }, far: { a: -22 }, label: 'Hold 10 seconds, keep breathing' },
        { t: 1, near: { a: -20 }, far: { a: -22 }, label: 'Fully relax' }
      ],
      fx: function (pts, pose, t) {
        if (t < 0.12 || t > 0.9) return '';
        var pulse = 0.55 + 0.45 * Math.sin(t * 34);
        var mid = P((pts.pelvis.x + pts.near.knee.x) / 2, (pts.pelvis.y + pts.near.knee.y) / 2);
        return '<ellipse class="pxa-fx" cx="' + rr(mid.x) + '" cy="' + rr(mid.y) +
          '" rx="17" ry="' + rr(6 + 2.5 * pulse) + '" opacity="' + rr(0.35 + 0.5 * pulse) + '"/>' +
          fxArrow(go(pts.near.knee, -90, 4), -90, 13);
      }
    },

    /* 3 -------------------------------------------------- Straight Leg Raise */
    {
      name: 'Straight Leg Raise', dur: 5, twoLeg: true, props: PROPS.mat,
      base: { rot: -90, x: 70, y: 88, arm: [0, 0], far: { h: 48, k: 82 } },
      cue: 'The knee stays locked straight the whole way',
      kf: [
        { t: 0, near: { h: 0, k: 0, a: -25 }, label: 'Tighten the quad to lock the knee' },
        { t: 0.34, near: { h: 42, k: 0, a: -25 }, label: 'Raise to the height of the other knee' },
        { t: 0.5, near: { h: 42, k: 0, a: -25 }, label: 'Hold 2 seconds at the top' },
        { t: 1, near: { h: 0, k: 0, a: -25 }, label: 'Lower slowly - do not let it drop' }
      ],
      fx: function (pts) {
        return '<path class="pxa-fx-soft" d="M' + rr(pts.far.knee.x - 26) + ' ' + rr(pts.far.knee.y) +
          'H' + rr(pts.far.knee.x + 34) + '"/>' + fxTarget(pts.far.knee, '');
      }
    },

    /* 4 ------------------------------------------------------- Heel Slides */
    {
      name: 'Heel Slides', dur: 5.5, twoLeg: true, props: PROPS.mat,
      base: { rot: -90, x: 70, y: 88, arm: [0, 0], far: { h: 0, k: 0, a: -22 } },
      cue: 'The heel stays on the floor throughout',
      kf: [
        { t: 0, near: { h: 0, k: 0, a: -20 }, label: 'Start with the leg straight' },
        { t: 0.42, near: { h: 45, k: 90, a: -8 }, label: 'Slide the heel toward your buttocks' },
        { t: 0.56, near: { h: 45, k: 90, a: -8 }, label: 'Hold briefly at end range' },
        { t: 1, near: { h: 0, k: 0, a: -20 }, label: 'Slide back out to straight' }
      ],
      fx: function (pts) {
        return '<path class="pxa-fx-soft" d="M' + rr(pts.near.ankle.x) + ' ' + rr(pts.near.ankle.y + 9) +
          'H' + rr(pts.pelvis.x + 64) + '"/>' + fxArrow(go(pts.near.ankle, 180, 8), 180, 12);
      }
    },

    /* 5 ------------------------------------------- Patellar Mobilisation */
    {
      name: 'Patellar Mobilisation', dur: 8, twoLeg: true, props: PROPS.mat,
      base: { rot: -90, x: 76, y: 92, lean: 30, arm: [0, 0], near: { k: 0, a: -18 }, far: { k: 0, a: -24 } },
      cue: 'Gentle glides only - never force the kneecap',
      kf: [
        { t: 0, label: 'Glide the kneecap upward' },
        { t: 0.25, label: 'Glide the kneecap downward' },
        { t: 0.5, label: 'Glide it toward the inside' },
        { t: 0.75, label: 'Glide it toward the outside' },
        { t: 1, label: 'Glide the kneecap upward' }
      ],
      fx: function (pts, pose, t) {
        // Hand orbits the kneecap: one direction per quarter of the cycle,
        // matching the four glides in the protocol instructions.
        var dirs = [-90, 90, 200, 20];
        var q = Math.min(3, Math.floor(t * 4));
        var u = (t * 4) % 1;
        var reach = 7 * Math.sin(u * Math.PI);
        var hand = go(pts.near.knee, dirs[q], reach);
        // Elbow: midpoint of shoulder-to-hand pushed outward, which reads as a
        // bent arm without needing a second IK solve for a two-frame detail.
        var mid = P((pts.shoulder.x + hand.x) / 2, (pts.shoulder.y + hand.y) / 2);
        var span = Math.atan2(hand.y - pts.shoulder.y, hand.x - pts.shoulder.x);
        var elbow = go(mid, deg(span) - 90, 11);
        return '<path class="pxa-limb pxa-arm" d="' + path(pts.shoulder, elbow, hand) + '"/>' +
          fxCircle(hand, 4.5, 'pxa-fx-fill') +
          '<circle class="pxa-fx-soft" cx="' + rr(pts.near.knee.x) + '" cy="' + rr(pts.near.knee.y) + '" r="9"/>' +
          fxArrow(pts.near.knee, dirs[q], 15);
      }
    },

    /* 6 --------------------------------------------------------- Mini Squat */
    {
      name: 'Mini Squat', dur: 4, twoLeg: true, bilateral: true, props: PROPS.floor,
      base: { view: 'front', x: 100, y: 78, stance: 17, ground: 'lowest' },
      cue: 'Knees track out over the toes - never let them fall inward',
      kf: [
        { t: 0, lean: 0, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Feet shoulder-width apart' },
        { t: 0.4, lean: 22, near: { h: 48, k: 60 }, far: { h: 48, k: 60 }, label: 'Bend slowly to 45-60' + DEG },
        { t: 0.55, lean: 22, near: { h: 48, k: 60 }, far: { h: 48, k: 60 }, label: 'Hold 2 seconds' },
        { t: 1, lean: 0, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Press through the heels to stand' }
      ],
      fx: kneeTrackingGuides
    },

    /* 7 ---------------------------------- Stationary Bike (Low Resistance) */
    {
      name: 'Stationary Bike (Low Resistance)', dur: 3.4, twoLeg: true, props: PROPS.bike,
      base: { rot: 0 },
      cue: 'Seat set so the knee bends to ~90' + DEG + ' at the bottom',
      kf: bikeFrames('Half revolutions first, then full - low resistance'),
      fx: function (pts, pose, t) {
        var c = P(126, 104), th = 360 * t;
        return '<path class="pxa-fx" d="' + path(c, go(c, th, 12)) + '"/>' +
          '<path class="pxa-fx" d="' + path(c, go(c, th + 180, 12)) + '"/>' +
          fxCircle(go(c, th, 12), 3, 'pxa-fx-fill');
      }
    },

    /* 8 ------------------------------------------ Step Up (4-6 inch step) */
    {
      name: 'Step Up (4-6 inch step)', dur: 5, twoLeg: true, props: PROPS.step,
      base: { rot: 0, arm: [8, 12] },
      cue: 'Up with the surgical leg, down with it last',
      kf: [
        { t: 0, x: 96, y: 78, lean: 4, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Stand at the base of the step' },
        { t: 0.26, x: 98, y: 75, lean: 8, near: { h: 58, k: 70 }, far: { h: 0, k: 2 }, label: 'Place the surgical foot on the step' },
        { t: 0.5, x: 112, y: 60, lean: 15, near: { h: 16, k: 22 }, far: { h: -30, k: 36 }, label: 'Press up through that leg' },
        { t: 0.68, x: 118, y: 58, lean: 4, near: { h: 0, k: 2 }, far: { h: 8, k: 10 }, label: 'Bring the other foot up to meet it' },
        { t: 0.85, x: 110, y: 62, lean: 12, near: { h: 14, k: 26 }, far: { h: -34, k: 42 }, label: 'Step down with the surgical leg last' },
        { t: 1, x: 96, y: 78, lean: 4, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Back to the floor, under control' }
      ]
    },

    /* 9 ------------------------------------------- Prone Hamstring Curl */
    {
      name: 'Prone Hamstring Curl', dur: 4.5, twoLeg: true, props: PROPS.mat,
      base: { rot: -90, m: -1, x: 70, y: 88, arm: [-140, 15], far: { h: 0, k: 6 } },
      cue: 'Hips stay flat on the mat',
      kf: [
        { t: 0, near: { h: 0, k: 0 }, label: 'Lie face-down, both legs straight' },
        { t: 0.42, near: { h: 0, k: 90 }, label: 'Bend the knee toward your buttocks' },
        { t: 0.55, near: { h: 0, k: 90 }, label: 'Stop at 90' + DEG + ' or at discomfort' },
        { t: 1, near: { h: 0, k: 0 }, label: 'Lower slowly under control' }
      ]
    },

    /* 10 ---------------------------------------------- Full Squat (Progressed) */
    {
      name: 'Full Squat (Progressed)', dur: 4.6, twoLeg: true, bilateral: true,
      props: PROPS.floor,
      base: { view: 'front', x: 100, y: 78, stance: 20, ground: 'lowest' },
      cue: 'Add depth week by week - never chase depth through pain',
      kf: [
        { t: 0, lean: 0, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Feet shoulder-width, toes slightly out' },
        { t: 0.42, lean: 32, near: { h: 95, k: 120 }, far: { h: 95, k: 120 }, label: 'Lower slowly, chest tall' },
        { t: 0.56, lean: 32, near: { h: 95, k: 120 }, far: { h: 95, k: 120 }, label: 'Pause at your working depth' },
        { t: 1, lean: 0, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Drive through the heels to return' }
      ],
      fx: kneeTrackingGuides
    },

    /* 11 ---------------------------- Stationary Bike (Moderate Resistance) */
    {
      name: 'Stationary Bike (Moderate Resistance)', dur: 2.2, twoLeg: true,
      props: PROPS.bike + '<circle class="pxa-prop" cx="146" cy="66" r="5"/><path class="pxa-prop" d="M146 66L149 62"/>',
      base: { rot: 0 },
      cue: '20-30 min at 60-80 RPM',
      kf: bikeFrames('Moderate resistance, steady cadence'),
      fx: function (pts, pose, t) {
        var c = P(126, 104), th = 360 * t;
        return '<path class="pxa-fx" d="' + path(c, go(c, th, 12)) + '"/>' +
          '<path class="pxa-fx" d="' + path(c, go(c, th + 180, 12)) + '"/>' +
          fxCircle(go(c, th, 12), 3, 'pxa-fx-fill');
      }
    },

    /* 12 ------------------------------------------------------ Forward Lunge */
    {
      name: 'Forward Lunge', dur: 4.6, twoLeg: true, props: PROPS.floor, base: STAND,
      cue: 'Front shin vertical - the knee must not pass the toes',
      kf: [
        { t: 0, x: 100, lean: 4, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Stand tall, support within reach' },
        { t: 0.4, x: 92, lean: 10, near: { h: 90, k: 90 }, far: { h: -30, k: 60, a: -75 }, label: 'Step forward, lower the back knee' },
        { t: 0.55, x: 92, lean: 10, near: { h: 90, k: 90 }, far: { h: -30, k: 60, a: -75 }, label: 'Back knee just above the floor' },
        { t: 1, x: 100, lean: 4, near: { h: 0, k: 0 }, far: { h: 0, k: 0 }, label: 'Push back through the front heel' }
      ],
      fx: function (pts) {
        return '<path class="pxa-fx-soft" d="M' + rr(pts.near.ankle.x) + ' ' + rr(GROUND - 2) +
          'V' + rr(pts.near.knee.y - 8) + '"/>';
      }
    },

    /* 13 -------------------------------------------------- Single-Leg Balance */
    {
      name: 'Single-Leg Balance', dur: 6, twoLeg: true, props: PROPS.wallLeft,
      base: {
        rot: 0, y: 78, ground: 'near', arm: [72, 18],
        near: { h: 0, k: 6 }, far: { h: 38, k: 78 }
      },
      cue: 'Wall or chair always within touching distance',
      kf: [
        { t: 0, x: 100, lean: 0, label: 'Stand on the surgical leg only' },
        { t: 0.25, x: 102, lean: 3.5, label: 'Hold 30 seconds without swaying' },
        { t: 0.5, x: 98, lean: -2, label: 'Keep the trunk quiet' },
        { t: 0.75, x: 101, lean: 3, label: 'Progress to eyes closed once stable' },
        { t: 1, x: 100, lean: 0, label: 'Stand on the surgical leg only' }
      ]
    },

    /* 14 ----------------------------------------------------------- Leg Press */
    {
      name: 'Leg Press', dur: 4, twoLeg: true, bilateral: true, props: PROPS.legPress,
      base: { rot: -55 },
      cue: 'Both heels drive - the stack never drops',
      kf: legPressFrames(),
      fx: function (pts) {
        // Plate sits square across the rail at whatever height the feet reached.
        var a = pts.near.ankle;
        var n1 = go(a, -68, 17), n2 = go(a, 112, 17);
        return '<path class="pxa-fx" d="' + path(n1, n2) + '" stroke-width="4"/>';
      }
    },

    /* 15 ------------------------------------------- Walk-to-Run Programme */
    {
      name: 'Walk-to-Run Programme', dur: 4.8, twoLeg: true, props: PROPS.ground,
      base: STAND,
      cue: 'Flat surfaces only for the first 2 weeks',
      kf: gait(0, 0.5, 1, 1.0, 'Walk 1 minute').concat(
        gait(0.5, 1, 1, 1.7, 'Jog 1 minute').slice(1))
    },

    /* 16 ------------------------------------------------- Seated Knee Bend */
    {
      name: 'Seated Knee Bend', dur: 5, twoLeg: true, props: PROPS.chair, base: SEATED,
      // The protocol lists start_angle 90 and angle_limit 90: sitting already
      // puts the knee at its ceiling. So the demo opens with the foot forward
      // of neutral and slides back INTO the ceiling, which is the movement the
      // patient actually performs.
      cue: 'Gravity does the work - assist gently with the other foot',
      kf: [
        { t: 0, near: { h: 90, k: 58 }, label: 'Sit with the foot forward of the knee' },
        { t: 0.42, near: { h: 90, k: 90 }, label: 'Slide the foot back under the chair' },
        { t: 0.58, near: { h: 90, k: 90 }, label: 'Hold 5 seconds at end range' },
        { t: 1, near: { h: 90, k: 58 }, label: 'Slide forward to release' }
      ],
      fx: function (pts) {
        return fxArrow(go(pts.near.ankle, 175, 10), 175, 13);
      }
    },

    /* 17 ------------------------------------------ Reciprocal Stair Descent */
    {
      name: 'Reciprocal Stair Descent', dur: 5.4, twoLeg: true, props: PROPS.stairs,
      base: { rot: 0, arm: [70, 25] },
      cue: 'Step over step, not step to step - hold the rail',
      kf: [
        { t: 0, x: 88, y: 60, lean: 4, near: { h: 0, k: 2 }, far: { h: 0, k: 5 }, label: 'Stand on the step, hold the rail' },
        { t: 0.36, x: 92, y: 64, lean: 12, near: { h: 22, k: 40 }, far: { h: 34, k: 24 }, label: 'Bend the surgical knee to lower' },
        { t: 0.58, x: 96, y: 71, lean: 16, near: { h: 20, k: 64 }, far: { h: 28, k: 16 }, label: 'Lower the other foot to the next step' },
        { t: 0.78, x: 98, y: 71, lean: 9, near: { h: 20, k: 64 }, far: { h: 10, k: 0 }, label: 'That needs 80-100' + DEG + ' of controlled flexion' },
        { t: 1, x: 88, y: 60, lean: 4, near: { h: 0, k: 2 }, far: { h: 0, k: 5 }, label: 'Return to the step above' }
      ]
    },

    /* 18 ---------------------------------------------- Standing Knee Bend */
    {
      name: 'Standing Knee Bend', dur: 4.5, twoLeg: true, props: PROPS.wallRight,
      base: { rot: 0, y: 78, ground: 'far', arm: [88, 12], far: { h: 0, k: 3 } },
      cue: 'Fingertips on the wall for balance only',
      kf: [
        { t: 0, x: 132, lean: 4, near: { h: -2, k: 9 }, label: 'Stand facing the wall' },
        { t: 0.4, x: 132, lean: 4, near: { h: -8, k: 90 }, label: 'Lift the foot up behind you' },
        { t: 0.56, x: 132, lean: 4, near: { h: -8, k: 90 }, label: 'Hold 5 seconds at the top' },
        { t: 1, x: 132, lean: 4, near: { h: -2, k: 9 }, label: 'Lower slowly' }
      ]
    },

    /* 19 --------------------------------------------- Walking Programme */
    {
      name: 'Walking Programme', dur: 3.6, twoLeg: true, props: PROPS.ground, base: STAND,
      cue: 'Aim for a normal heel-to-toe pattern',
      kf: gait(0, 1, 2, 1.0, 'Flat, even ground - 20 to 40 min daily')
    },

    /* 20 ------------------------ Seated Knee Extension (Gravity Only) */
    {
      name: 'Seated Knee Extension (Gravity Only)', dur: 5, twoLeg: true,
      props: PROPS.chair, base: SEATED,
      // angle_limit is 90 here (the seated ceiling) while hold_angle is 45 (the
      // extension target). The two are different numbers and the animation has
      // to honour both: bend no further than 90, reach and hold at 45.
      cue: 'Gravity only - no ankle weights in this phase',
      kf: [
        { t: 0, near: { h: 90, k: 90 }, label: 'Sit with the knee at 90' + DEG },
        { t: 0.38, near: { h: 90, k: 45 }, label: 'Extend the knee as far as comfortable' },
        { t: 0.58, near: { h: 90, k: 45 }, label: 'Hold 5 seconds at end range' },
        { t: 1, near: { h: 90, k: 90 }, label: 'Lower slowly back down' }
      ],
      fx: function (pts, pose) {
        var td = 90 - pose.near.h;
        var target = go(go(pts.pelvis, td, SEG.thigh), td + 45, SEG.shank);
        return fxTarget(target, '45' + DEG);
      }
    },

    /* 21 --------------------------------------- Pool Walking (if available) */
    {
      name: 'Pool Walking (if available)', dur: 4.4, twoLeg: true, props: PROPS.pool, base: STAND,
      cue: 'Chest-deep water removes about 75% of body weight',
      kf: gait(0, 1, 2, 0.8, 'Walk 15-20 min at a comfortable pace'),
      fx: function (pts, pose, t) {
        var out = '';
        for (var i = 0; i < 3; i++) {
          var w = 8 + 14 * (((t * 1.4 + i / 3) % 1));
          var op = 0.5 * (1 - ((t * 1.4 + i / 3) % 1));
          out += '<ellipse class="pxa-fx-soft" cx="' + rr(pts.pelvis.x) + '" cy="52" rx="' +
            rr(w) + '" ry="' + rr(w * 0.22) + '" opacity="' + rr(op) + '"/>';
        }
        return out;
      }
    }
  ];

  /* ------------------------------------------------------------- registry */

  function slug(s) { return String(s).toLowerCase().replace(/[^a-z0-9]+/g, ''); }

  var BY_SLUG = {};
  SPECS.forEach(function (spec) {
    spec.slug = slug(spec.name);
    spec.twoLeg = spec.twoLeg !== false;
    spec.kfs = spec.kf.map(function (k) { return norm(k, spec.base || {}, spec.twoLeg); });
    BY_SLUG[spec.slug] = spec;
  });

  function find(name) { return BY_SLUG[slug(name)] || null; }

  /* -------------------------------------------------------------- instance */

  function mount(container, opts) {
    opts = opts || {};
    var spec = find(opts.name);
    if (!spec) return null;
    injectStyles();

    var wrap = document.createElement('div');
    wrap.className = 'pxa';

    var stage = document.createElement('div');
    stage.className = 'pxa-stage';

    var svg = svgEl('svg', { viewBox: '0 0 200 150', class: 'pxa-svg', 'aria-hidden': 'true' });
    var gProps = svgEl('g', {});
    gProps.innerHTML = spec.props || '';

    var farLeg = svgEl('path', { class: 'pxa-limb pxa-far' });
    var torso = svgEl('path', { class: 'pxa-limb pxa-torso' });
    var arm = svgEl('path', { class: 'pxa-limb pxa-arm' });
    var head = svgEl('circle', { class: 'pxa-head', r: SEG.head });
    var nearLeg = svgEl('path', { class: 'pxa-limb pxa-near' });
    var knee = svgEl('circle', { class: 'pxa-knee', r: 4.2 });
    var gFx = svgEl('g', {});

    [gProps, farLeg, torso, arm, head, nearLeg, knee, gFx].forEach(function (n) { svg.appendChild(n); });

    var badge = document.createElement('div');
    badge.className = 'pxa-badge';
    var cap = document.createElement('div');
    cap.className = 'pxa-cap';
    var capSpan = document.createElement('span');
    cap.appendChild(capSpan);

    stage.appendChild(svg);
    if (opts.showAngle !== false) stage.appendChild(badge);
    if (opts.showCaption !== false) stage.appendChild(cap);
    wrap.appendChild(stage);

    var note = document.createElement('div');
    note.className = 'pxa-note' + (opts.capped ? ' capped' : '');
    note.textContent = opts.capped
      ? 'Range reduced to ' + opts.angleLimit + DEG + ' by your X-ray'
      : (spec.cue || '');
    if (opts.showNote !== false) wrap.appendChild(note);

    container.appendChild(wrap);

    var inst = {
      spec: spec,
      _limit: opts.angleLimit != null ? opts.angleLimit : null,
      _playing: opts.autoplay !== false,
      _visible: true,
      _t0: null,
      el: wrap
    };

    inst._draw = function (nowSec) {
      if (inst._t0 == null) inst._t0 = nowSec;
      var t = ((nowSec - inst._t0) / spec.dur) % 1;
      inst._frame(t);
    };

    inst._frame = function (t) {
      inst._lastT = t;
      var pose = sample(spec.kfs, t);
      var pts = solve(pose, inst._limit, spec.bilateral);

      // Head-on, the legs hang off separate hip points and the body needs a
      // shoulder bar, a pelvis bar and two arms instead of one of each.
      var F = pts.frontal;
      if (pts.far) farLeg.setAttribute('d', path(pts.far.hip || pts.pelvis, pts.far.knee, pts.far.ankle, pts.far.toe));
      else farLeg.setAttribute('d', '');
      torso.setAttribute('d', F
        ? path(pts.pelvis, pts.shoulder) + path(F.shoulderL, F.shoulderR) + path(F.hipL, F.hipR)
        : path(pts.pelvis, pts.shoulder));
      arm.setAttribute('d', F
        ? path(F.armL[0], F.armL[1], F.armL[2]) + path(F.armR[0], F.armR[1], F.armR[2])
        : path(pts.shoulder, pts.elbow, pts.hand));
      head.setAttribute('cx', rr(pts.head.x));
      head.setAttribute('cy', rr(pts.head.y));
      nearLeg.setAttribute('d', path(pts.near.hip || pts.pelvis, pts.near.knee, pts.near.ankle, pts.near.toe));
      knee.setAttribute('cx', rr(pts.near.knee.x));
      knee.setAttribute('cy', rr(pts.near.knee.y));

      // Same three-band colouring the tracker HUD uses, so the tutorial and the
      // live session read identically.
      var cls = 'pxa-knee';
      if (inst._limit != null) {
        if (pts.flex >= inst._limit - 0.5) cls += ' danger';
        else if (pts.flex > inst._limit * 0.85) cls += ' warn';
      }
      knee.setAttribute('class', cls);

      if (opts.showAngle !== false) {
        badge.textContent = Math.round(pts.flex) + DEG +
          (inst._limit != null ? ' / ' + inst._limit + DEG : '');
        badge.className = 'pxa-badge' +
          (inst._limit != null && pts.flex >= inst._limit - 0.5 ? ' danger' : '');
      }
      if (opts.showCaption !== false) capSpan.textContent = pose.label || '';
      gFx.innerHTML = spec.fx ? spec.fx(pts, pose, t, inst) : '';
    };

    inst.setLimit = function (v) {
      inst._limit = v == null ? null : v;
      inst._frame(inst._lastT || (spec.still || 0.45));
      return inst;
    };
    inst.play = function () { inst._playing = true; inst._t0 = null; ensureLoop(); return inst; };
    inst.pause = function () { inst._playing = false; return inst; };
    inst.destroy = function () {
      var i = instances.indexOf(inst);
      if (i >= 0) instances.splice(i, 1);
      if (io) io.disconnect();
      if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
    };

    // Respect reduced-motion: freeze on a frame that shows the working range,
    // rather than looping motion at someone who asked for none.
    var reduce = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      inst._playing = false;
      inst._frame(spec.still || 0.48);
    } else {
      inst._frame(0);
      instances.push(inst);
      ensureLoop();
    }

    var io = null;
    if (!reduce && global.IntersectionObserver) {
      io = new IntersectionObserver(function (entries) {
        inst._visible = entries[0].isIntersecting;
      }, { threshold: 0.05 });
      io.observe(stage);
    }

    return inst;
  }

  global.PhysioAnimations = {
    mount: mount,
    has: function (name) { return !!find(name); },
    get: find,
    names: function () { return SPECS.map(function (s) { return s.name; }); },
    _specs: SPECS,
    // Exposed so the geometry can be checked headlessly (no DOM), which is how
    // the poses were kept inside the 200x150 viewBox.
    _debug: { sample: sample, solve: solve, ik2: ik2, GROUND: GROUND, SEG: SEG }
  };
})(typeof window !== 'undefined' ? window : this);
