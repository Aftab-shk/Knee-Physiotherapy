/*
 * Tests for assets/form-check.js.
 *
 * The tracker measures how far the knee bent. A physiotherapist would be
 * watching whether the movement came from the right place - a straight leg raise
 * performed by rocking the trunk hits the same knee angle as a proper one and
 * does almost none of the same work.
 *
 * Two things these have to get right, and they pull against each other:
 *
 *   * Catch a rep genuinely being worked around.
 *   * Stay quiet otherwise. Someone corrected every few seconds stops listening,
 *     and then the corrections that matter land on deaf ears.
 *
 * Everything is measured against a baseline taken while the patient is set up
 * and still, because "leaning" only means anything relative to how they started.
 *
 * Run:  node --test        (from frontend/)
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  FORM,
  assessForm,
  captureBaseline,
  createFormMonitor,
  torsoLength,
  trunkAngle,
} from "../assets/form-check.js";

// ── A synthetic body, side-on ───────────────────────────────────────────────
// Only the four trunk landmarks matter here; the knee angle is passed in
// separately, because the tracker has already computed it.
const S = 0.5;

function body({ leanDeg = 0, hipRise = 0, driftX = 0 } = {}) {
  const torsoM = 0.48;                       // shoulder-to-hip, metres
  const hipY = 0.60 - hipRise * torsoM * S;  // image y grows downward
  const rad = (leanDeg * Math.PI) / 180;

  const hip = { x: 0.5 + driftX, y: hipY };
  const shoulder = {
    x: hip.x + Math.sin(rad) * torsoM * S,
    y: hip.y - Math.cos(rad) * torsoM * S,
  };

  const lm = new Array(33).fill(null);
  // Side-on, so left and right project onto the same point.
  lm[11] = { ...shoulder }; lm[12] = { ...shoulder };
  lm[23] = { ...hip };      lm[24] = { ...hip };
  return lm;
}

const upright = body();
const base = captureBaseline(upright);

const codes = (result) => result.faults.map(f => f.code);


// ── Geometry ────────────────────────────────────────────────────────────────

test("an upright trunk reads as vertical", () => {
  assert.ok(Math.abs(trunkAngle(upright)) < 0.01);
});

test("trunk lean is signed, so forward and back are distinguishable", () => {
  assert.ok(trunkAngle(body({ leanDeg: 20 })) > 15);
  assert.ok(trunkAngle(body({ leanDeg: -20 })) < -15);
});

test("torso length is the scale everything else uses", () => {
  assert.ok(torsoLength(upright) > 0.1);
});

test("a baseline needs a patient close enough to measure", () => {
  const tiny = body();
  for (const i of [11, 12, 23, 24]) {
    tiny[i] = { x: 0.5, y: 0.5 + (tiny[i].y - 0.5) * 0.05 };
  }
  assert.equal(captureBaseline(tiny), null);
});

test("a baseline needs the landmarks it is built from", () => {
  const missing = body();
  missing[23] = null;
  assert.equal(captureBaseline(missing), null);
});


// ── Quiet when the exercise is done properly ────────────────────────────────

test("a still, upright patient raises nothing", () => {
  assert.deepEqual(codes(assessForm(upright, base)), []);
});

test("small shifts are not corrections", () => {
  // Breathing, settling, a slight adjustment.
  assert.deepEqual(codes(assessForm(body({ leanDeg: 6 }), base)), []);
  assert.deepEqual(codes(assessForm(body({ hipRise: 0.04 }), base)), []);
});

test("no baseline means no assessment, rather than a guess", () => {
  assert.deepEqual(assessForm(upright, null).faults, []);
  assert.deepEqual(assessForm(null, base).faults, []);
});

test("a reclined starting position is not itself a fault", () => {
  // Someone propped on their elbows for a quad set. Their trunk is permanently
  // tilted, and calling that a fault every frame would make this pure noise.
  const reclined = body({ leanDeg: -35 });
  const ownBaseline = captureBaseline(reclined);
  assert.deepEqual(codes(assessForm(reclined, ownBaseline)), []);
});


// ── The faults ──────────────────────────────────────────────────────────────

test("rocking the trunk backwards is caught", () => {
  const result = assessForm(body({ leanDeg: -25 }), base);
  const fault = result.faults.find(f => f.code === "trunk_lean");
  assert.ok(fault);
  assert.match(fault.message, /leaning back/i);
  assert.ok(Math.abs(fault.value) > FORM.TRUNK_LEAN_DEG);
});

test("leaning forward gets its own wording", () => {
  const fault = assessForm(body({ leanDeg: 25 }), base).faults
    .find(f => f.code === "trunk_lean");
  assert.match(fault.message, /leaning forward/i);
});

test("lifting the hips off the mat is caught", () => {
  const result = assessForm(body({ hipRise: 0.2 }), base);
  const fault = result.faults.find(f => f.code === "hip_lift");
  assert.ok(fault, "the classic straight-leg-raise cheat");
  assert.match(fault.message, /hips/i);
});

test("hips settling downwards is not a lift", () => {
  assert.ok(!codes(assessForm(body({ hipRise: -0.2 }), base)).includes("hip_lift"));
});

test("a straight hold done with a bent knee is caught", () => {
  const result = assessForm(upright, base, { holdTarget: "straight", kneeAngle: 25 });
  const fault = result.faults.find(f => f.code === "knee_not_locked");
  assert.ok(fault, "the timer would otherwise run through a quad set nobody is doing");
  assert.equal(fault.value, 25);
});

test("a properly locked knee passes", () => {
  const result = assessForm(upright, base, { holdTarget: "straight", kneeAngle: 3 });
  assert.ok(!codes(result).includes("knee_not_locked"));
});

test("a bent-target hold is not asked to be straight", () => {
  const result = assessForm(upright, base, { holdTarget: "flexed", kneeAngle: 85 });
  assert.ok(!codes(result).includes("knee_not_locked"));
});

test("drifting is only a fault where standing still is the exercise", () => {
  const drifting = body({ driftX: 0.12 });
  assert.ok(codes(assessForm(drifting, base, { exerciseName: "Single-Leg Balance" }))
    .includes("sway"));
  // During a squat, moving is the point.
  assert.ok(!codes(assessForm(drifting, base, { exerciseName: "Mini Squat" }))
    .includes("sway"));
});

test("every fault carries the number it fired on", () => {
  const result = assessForm(body({ leanDeg: -30, hipRise: 0.25 }), base,
                            { holdTarget: "straight", kneeAngle: 30 });
  assert.ok(result.faults.length >= 3);
  for (const fault of result.faults) {
    assert.equal(typeof fault.value, "number", fault.code);
    assert.equal(typeof fault.threshold, "number", fault.code);
    assert.ok(fault.message, fault.code);
  }
});

test("the measurements come back whether or not anything fired", () => {
  const clean = assessForm(upright, base);
  assert.equal(typeof clean.metrics.trunkLeanDeg, "number");
  assert.equal(typeof clean.metrics.hipLiftFraction, "number");
  assert.equal(typeof clean.metrics.swayFraction, "number");
});


// ── Staying quiet: the sustain guard ────────────────────────────────────────

test("a momentary wobble is not corrected", () => {
  const monitor = createFormMonitor(15);
  const fault = [{ code: "trunk_lean" }];

  for (let i = 0; i < 5; i++) assert.deepEqual(monitor.update(fault), []);
  assert.deepEqual(monitor.update([]), [], "it went away before it mattered");
});

test("a sustained fault is mentioned once, not every frame", () => {
  const monitor = createFormMonitor(15);
  const fault = [{ code: "trunk_lean", message: "…" }];

  let announcements = 0;
  for (let i = 0; i < 90; i++) announcements += monitor.update(fault).length;

  assert.equal(announcements, 1, "three seconds of leaning is one correction");
});

test("a fault that returns is mentioned again", () => {
  const monitor = createFormMonitor(5);
  const fault = [{ code: "hip_lift" }];

  for (let i = 0; i < 10; i++) monitor.update(fault);
  for (let i = 0; i < 10; i++) monitor.update([]);          // corrected

  let again = 0;
  for (let i = 0; i < 10; i++) again += monitor.update(fault).length;
  assert.equal(again, 1, "doing it again is a new mistake");
});

test("different faults are tracked separately", () => {
  const monitor = createFormMonitor(3);
  const both = [{ code: "trunk_lean" }, { code: "hip_lift" }];

  monitor.update(both);
  monitor.update(both);
  const ripe = monitor.update(both);
  assert.deepEqual(ripe.map(f => f.code).sort(), ["hip_lift", "trunk_lean"]);
});

test("a new set starts listening from scratch", () => {
  const monitor = createFormMonitor(3);
  const fault = [{ code: "trunk_lean" }];

  for (let i = 0; i < 10; i++) monitor.update(fault);
  monitor.reset();

  let announced = 0;
  for (let i = 0; i < 10; i++) announced += monitor.update(fault).length;
  assert.equal(announced, 1, "the same mistake in the next set is worth saying again");
});

test("active() reports what is currently held", () => {
  const monitor = createFormMonitor(3);
  for (let i = 0; i < 5; i++) monitor.update([{ code: "sway" }]);
  assert.deepEqual(monitor.active(), ["sway"]);

  monitor.update([]);
  assert.deepEqual(monitor.active(), []);
});
