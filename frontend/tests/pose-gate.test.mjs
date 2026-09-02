/*
 * Tests for the camera-view gate in assets/pose-gate.js.
 *
 * The regression these exist to prevent:
 *
 *   calcKneeAngle() measures a projected angle — the angle of the leg as it
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
//   90° = perfectly side-on (sagittal — the only view the 2D maths is valid in)
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
