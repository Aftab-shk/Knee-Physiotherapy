/*
 * Tests for the camera-view gate in assets/pose-gate.js.
 *
 * The regression these exist to prevent:
 *
 *   calcKneeAngle() measures a projected angle - the angle of the leg as it
 *   appears in the image. Square to the camera, femur and tibia project onto
 *   nearly the same line, so a knee genuinely bent to 60° reads 0°. The HUD
 *   shows a small number, the safe ceiling is never crossed, and the alarm that
 *   is supposed to stop the patient bending too far stays silent for the entire
 *   session. Nothing about it looks broken.
 *
 *   Under-reading is the dangerous direction, and it is the direction a bad
 *   camera angle fails in. So the gate has one job: never let a view through
 *   whose measurement is materially low.
 *
 * Run:  node --test frontend/tests/        (or: npm --prefix frontend test)
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  POSE_GATE,
  assessView,
  calcKneeAngle,
  isHardFail,
  landmarkIndices,
} from "../assets/pose-gate.js";

// ── A synthetic body, projected orthographically ────────────────────────────
// theta is rotation about the vertical axis:
//   90° = perfectly side-on (sagittal - the only view the 2D maths is valid in)
//    0° = square to the camera, where knee flexion is invisible
// Segment lengths are adult averages in metres; S maps metres to normalised
// image units for a frame covering roughly 2 m of height.
const S = 0.5;
const GROUND = 0.95;
const KNEE_H = 0.50;
const FEMUR = 0.42;

function makeBody(thetaDeg, flexDeg) {
  const t = (thetaDeg * Math.PI) / 180;
  const f = (flexDeg * Math.PI) / 180;
  const ct = Math.cos(t);
  const st = Math.sin(t);

  // [lateral, height, anterior-posterior] per landmark. Flexion swings the
  // tibia backwards in the sagittal plane, which is the movement that vanishes
  // when the camera is square-on.
  const shinH = KNEE_H - FEMUR * Math.cos(f);
  const shinD = -FEMUR * Math.sin(f);
  const spec = {
    11: [-0.19, 1.40, 0], 12: [0.19, 1.40, 0],           // shoulders
    23: [-0.095, 0.92, 0], 24: [0.095, 0.92, 0],         // hips
    25: [-0.095, KNEE_H, 0], 26: [0.095, KNEE_H, 0],     // knees
    27: [-0.095, shinH, shinD], 28: [0.095, shinH, shinD], // ankles
    31: [-0.095, shinH - 0.06, shinD + 0.1],             // foot index
    32: [0.095, shinH - 0.06, shinD + 0.1],
  };

  const lm = new Array(33).fill(null);
  const world = new Array(33).fill(null);
  for (const [i, [l, h, d]] of Object.entries(spec)) {
    lm[i] = { x: 0.5 + (l * ct + d * st) * S, y: GROUND - h * S, visibility: 0.99 };
    // World landmarks are metric and hip-centred; depth is the axis the camera
    // cannot see.
    world[i] = { x: l * ct + d * st, y: -(h - 0.92), z: -l * st + d * ct };
  }
  return { lm, world };
}

const IDX = landmarkIndices("right");
const view = (b, ceiling = POSE_GATE.SPREAD_OK) =>
  assessView(b.lm, b.world, IDX, { spreadCeiling: ceiling });
const reading = (b) => calcKneeAngle(b.lm[IDX.hip], b.lm[IDX.knee], b.lm[IDX.ankle]);

// ── The failure the gate exists for ─────────────────────────────────────────

test("square-on, a 60° knee reads as almost straight", () => {
  const b = makeBody(0, 60);
  assert.ok(reading(b) < 15, `reads ${reading(b).toFixed(1)}°, expected it to collapse toward 0°`);
});

test("square-on is rejected, so that reading is never used", () => {
  const v = view(makeBody(0, 60));
  assert.equal(v.ok, false);
  assert.equal(v.code, "not-sagittal");
});

// ── Valid views work ────────────────────────────────────────────────────────

for (const flex of [0, 30, 60, 90]) {
  test(`side-on at ${flex}° flexion is accepted and measures accurately`, () => {
    const b = makeBody(90, flex);
    assert.equal(view(b).ok, true, `rejected with code=${view(b).code}`);
    assert.ok(
      Math.abs(reading(b) - flex) < 2,
      `reads ${reading(b).toFixed(1)}° for a true ${flex}°`,
    );
  });
}

// ── The safety property ─────────────────────────────────────────────────────

test("every accepted view measures within 5° of the truth", () => {
  // Swept finely rather than at a few points: the guarantee has to hold across
  // the whole accepted region, not just where it was convenient to check.
  const TRUE_FLEX = 60;
  let worst = { off: null, under: 0 };

  for (let theta = 90; theta >= 0; theta -= 0.5) {
    const b = makeBody(theta, TRUE_FLEX);
    if (!view(b).ok) continue;
    const under = TRUE_FLEX - reading(b);
    if (under > worst.under) worst = { off: 90 - theta, under };
  }

  assert.ok(
    worst.under < 5,
    `an accepted view under-read by ${worst.under.toFixed(1)}° at ${worst.off}° off-axis`,
  );
});

test("the gate closes before the error grows past a few degrees", () => {
  // The complement of the test above: confirm rejection actually kicks in, so
  // the guarantee is not vacuously true by accepting nothing.
  assert.equal(view(makeBody(90 - 25, 60)).ok, true, "25° off-axis should still be usable");
  assert.equal(view(makeBody(90 - 30, 60)).ok, false, "30° off-axis should be refused");
});

// ── Framing and visibility guards ───────────────────────────────────────────

test("a low-confidence landmark is refused", () => {
  const b = makeBody(90, 30);
  b.lm[IDX.knee].visibility = 0.2;
  assert.equal(view(b).code, "low-visibility");
});

test("a joint clipped at the frame edge is refused", () => {
  const b = makeBody(90, 30);
  b.lm[IDX.ankle].y = 0.995;
  assert.equal(view(b).code, "out-of-frame");
});

test("a patient too far away is refused", () => {
  const b = makeBody(90, 30);
  for (let i = 0; i < b.lm.length; i++) {
    if (b.lm[i]) b.lm[i].y = 0.5 + (b.lm[i].y - 0.5) * 0.15;
  }
  assert.equal(view(b).code, "too-far");
});

test("a missing landmark is refused rather than throwing", () => {
  const b = makeBody(90, 30);
  b.lm[23] = null;
  assert.equal(view(b).code, "no-pose");
});

test("no landmarks at all is refused rather than throwing", () => {
  assert.equal(assessView(null, null, IDX, {}).code, "no-pose");
});

// ── Ankle Pumps counts off the foot, so the foot has to be visible ──────────

test("an unseen foot only blocks the exercise that counts off it", () => {
  const b = makeBody(90, 10);
  b.lm[IDX.foot].visibility = 0.1;

  assert.equal(
    assessView(b.lm, b.world, IDX, { trackedJoint: "knee" }).ok,
    true,
    "a knee-counted exercise does not need the foot",
  );
  assert.equal(
    assessView(b.lm, b.world, IDX, { trackedJoint: "ankle" }).code,
    "low-visibility",
    "ankle pumps cannot be counted without the foot",
  );
});

// ── Override behaviour ──────────────────────────────────────────────────────

test("the overridable failures are exactly the view-quality ones", () => {
  // A patient may override the side-on checks; they may not override a frame
  // with no usable angle in it at all.
  assert.equal(isHardFail("no-pose"), true);
  assert.equal(isHardFail("low-visibility"), true);
  assert.equal(isHardFail("out-of-frame"), true);
  assert.equal(isHardFail("not-sagittal"), false);
  assert.equal(isHardFail("foreshortened"), false);
  assert.equal(isHardFail("too-far"), false);
});

// ── Hysteresis ──────────────────────────────────────────────────────────────

test("there is a band where tracking holds but would not have started", () => {
  // Without this the session bounces in and out of suspension at the threshold,
  // mid-rep.
  assert.ok(POSE_GATE.SPREAD_RECOVER > POSE_GATE.SPREAD_OK);

  let found = null;
  for (let theta = 90; theta >= 30; theta -= 0.5) {
    const b = makeBody(theta, 45);
    if (!view(b, POSE_GATE.SPREAD_OK).ok && view(b, POSE_GATE.SPREAD_RECOVER).ok) {
      found = 90 - theta;
      break;
    }
  }
  assert.notEqual(found, null, "no pose exists that holds tracking but would not start it");
});

// ── Landmark selection ──────────────────────────────────────────────────────

test("landmark indices follow the knee being treated", () => {
  assert.deepEqual(landmarkIndices("left"), { hip: 23, knee: 25, ankle: 27, foot: 31, side: "left" });
  assert.deepEqual(landmarkIndices("right"), { hip: 24, knee: 26, ankle: 28, foot: 32, side: "right" });
  // 'both' has no single tracked leg. It must resolve to a real one, and the
  // guidance text shown to the patient reads from the same `side` field.
  assert.deepEqual(landmarkIndices("both"), { hip: 24, knee: 26, ankle: 28, foot: 32, side: "right" });
});

// ── Frame shape must not change the reading ─────────────────────────────────
// MediaPipe normalises x by frame width and y by frame height. Treating those
// as the same unit made a true 60° knee read 44° on a 1280×720 webcam - under
// -reading, which is the direction that lets a patient past their ceiling in
// silence. Every body above is built square, so nothing here caught it.

/** The same physical leg, normalised the way MediaPipe would for a W×H frame. */
function legInFrame(flexDeg, W, H) {
  const f = (flexDeg * Math.PI) / 180;
  const thigh = 200, shank = 200;                       // pixels
  const hip   = { x: 300, y: 100 };
  const knee  = { x: 300, y: 100 + thigh };
  const ankle = { x: knee.x - Math.sin(f) * shank, y: knee.y + Math.cos(f) * shank };
  const n = p => ({ x: p.x / W, y: p.y / H, visibility: 0.99 });
  return { hip: n(hip), knee: n(knee), ankle: n(ankle), aspect: W / H };
}

test("the knee angle is the same whatever shape the frame is", () => {
  for (const trueDeg of [20, 45, 60, 90, 120]) {
    for (const [W, H] of [[600, 600], [640, 480], [1280, 720], [720, 1280]]) {
      const leg = legInFrame(trueDeg, W, H);
      const got = calcKneeAngle(leg.hip, leg.knee, leg.ankle, leg.aspect);
      assert.ok(
        Math.abs(got - trueDeg) < 0.5,
        `${trueDeg}° in a ${W}x${H} frame read as ${got.toFixed(1)}°`,
      );
    }
  }
});

test("without the frame shape, a 16:9 webcam under-reads a bent knee", () => {
  // Guards the fix by pinning what it was fixing: the uncorrected call is still
  // wrong, so anyone who drops the argument at a call site gets a failing test
  // rather than a silently low reading.
  const leg = legInFrame(60, 1280, 720);
  const uncorrected = calcKneeAngle(leg.hip, leg.knee, leg.ankle);
  assert.ok(uncorrected < 50, `expected a low reading, got ${uncorrected.toFixed(1)}°`);
  assert.ok(Math.abs(calcKneeAngle(leg.hip, leg.knee, leg.ankle, leg.aspect) - 60) < 0.5);
});

test("the side-on test survives a non-square frame", () => {
  // spread ÷ torso is a ratio of a sideways length to a mostly-vertical one, so
  // it moved with the frame shape too: a square-on patient could pass the gate
  // on one camera and fail on another.
  // makeBody() builds square coordinates (both axes divided by the same
  // number). In a W×H frame MediaPipe would divide y by H instead, which is the
  // square value times the aspect. Scaling that back to fit inside the frame is
  // a uniform zoom, so it changes neither angles nor the spread ÷ torso ratio -
  // only the anisotropy under test survives it.
  const squash = (b, aspect) => {
    const pts = b.lm.filter(Boolean);
    const ys = pts.map(p => p.y * aspect);
    const lo = Math.min(...ys), hi = Math.max(...ys);
    const zoom = Math.min(1, 0.86 / (hi - lo));
    const cy = (lo + hi) / 2;
    return {
      lm: b.lm.map(p => (p ? {
        ...p,
        x: 0.5 + (p.x - 0.5) * zoom,
        y: 0.5 + (p.y * aspect - cy) * zoom,
      } : p)),
      world: b.world,
    };
  };
  for (const aspect of [4 / 3, 16 / 9, 9 / 16]) {
    const sideOn = squash(makeBody(90, 60), aspect);
    const squareOn = squash(makeBody(0, 60), aspect);
    assert.equal(
      assessView(sideOn.lm, sideOn.world, IDX, { aspect }).ok, true,
      `side-on rejected at aspect ${aspect.toFixed(2)}`,
    );
    assert.equal(
      assessView(squareOn.lm, squareOn.world, IDX, { aspect }).code, "not-sagittal",
      `square-on accepted at aspect ${aspect.toFixed(2)}`,
    );
  }
});
