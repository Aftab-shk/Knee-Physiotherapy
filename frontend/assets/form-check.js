/*
 * form-check.js - is the exercise being done, or worked around?
 * =============================================================
 *
 * The tracker measures how far the knee bent. A physiotherapist watching would
 * be looking at something else entirely: whether the movement came from the
 * right place. A straight leg raise performed by rocking the trunk backwards
 * hits the same knee angle as one performed properly and does almost none of
 * the same work - and the person doing it has no idea, because the counter went
 * up either way.
 *
 * What is NOT here, and why
 * -------------------------
 * Knee valgus and hip hiking are the two faults most often asked for, and
 * neither is measurable here. Both are **frontal-plane** movements: the knee
 * falling inward, the pelvis rising on one side. Seeing them needs a camera in
 * front of the patient.
 *
 * The tracker requires the opposite. A knee flexion angle from 2D landmarks is
 * only valid side-on (see pose-gate.js - square to the camera, a 60-degree knee
 * reads 0), so the session will not even start until the camera is in the one
 * position from which valgus cannot be seen.
 *
 * That is a genuine conflict, not an oversight. Detecting valgus properly means
 * a second camera position and a separate assessment mode, which is a different
 * feature. Guessing at it from MediaPipe's depth estimate would be worse than
 * not offering it: this module's z-axis is exactly the noisy one pose-gate.js
 * already refuses to trust with a safety decision.
 *
 * So what follows is the sagittal-plane faults, measured properly, rather than
 * every fault, measured hopefully.
 */

// Landmark indices, as in pose-gate.js: 11/12 shoulder, 23/24 hip.
const L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24;

export const FORM = {
  // Degrees the trunk may drift from where it started before it counts as
  // compensation. Generous: people breathe, shift and settle, and a coach that
  // corrects posture every few seconds gets ignored.
  TRUNK_LEAN_DEG: 12,

  // A "straight" hold performed with the knee this far bent is not the exercise.
  // Quad sets and straight leg raises carry a 5-degree limit, so this is a real
  // failure to lock out rather than measurement slop.
  KNEE_NOT_LOCKED_DEG: 10,

  // Pelvis rising, as a fraction of torso length. Lifting the hips off the mat
  // to swing the leg up is the classic straight-leg-raise cheat.
  HIP_LIFT_FRACTION: 0.09,

  // Trunk wander during a balance hold, as a fraction of torso length. Some
  // sway is the exercise working; a lot of it is a patient about to put the
  // other foot down.
  SWAY_FRACTION: 0.14,

  // Frames a fault must persist before it is worth mentioning. At ~30fps this
  // is about half a second - long enough to skip a wobble, short enough to
  // catch a rep being performed wrongly.
  SUSTAIN_FRAMES: 15,
};

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * The trunk's tilt, in degrees from vertical.
 *
 * Signed, so leaning back and leaning forward are distinguishable - a
 * comparison against a baseline needs to know which way it moved.
 */
export function trunkAngle(lm) {
  const shoulders = mid(lm[L_SHOULDER], lm[R_SHOULDER]);
  const hips = mid(lm[L_HIP], lm[R_HIP]);
  // Image y grows downward, so an upright trunk has shoulders above hips.
  return Math.atan2(shoulders.x - hips.x, hips.y - shoulders.y) * 180 / Math.PI;
}

/** Shoulder-to-hip distance: the scale everything else is measured against. */
export function torsoLength(lm) {
  return dist(mid(lm[L_SHOULDER], lm[R_SHOULDER]), mid(lm[L_HIP], lm[R_HIP]));
}

/**
 * A resting reference, taken while the patient is set up and still.
 *
 * Everything here is relative: "leaning" only means anything against how this
 * person was positioned to begin with. Someone doing a quad set propped on their
 * elbows has a permanently reclined trunk, and calling that a fault every frame
 * would make the whole feature noise.
 */
export function captureBaseline(lm) {
  if (!lm || !lm[L_SHOULDER] || !lm[R_SHOULDER] || !lm[L_HIP] || !lm[R_HIP]) return null;
  const torso = torsoLength(lm);
  if (torso < 0.06) return null;   // too far away to measure against
  return {
    trunkAngle: trunkAngle(lm),
    hipY: mid(lm[L_HIP], lm[R_HIP]).y,
    shoulderX: mid(lm[L_SHOULDER], lm[R_SHOULDER]).x,
    torso,
  };
}

/**
 * Check one frame against the baseline.
 *
 * @param {Array}  lm       normalised 2D landmarks
 * @param {Object} baseline from captureBaseline(), or null
 * @param {Object} opts     { holdTarget, kneeAngle, exerciseName }
 * @returns {{faults: Array, metrics: Object}} faults carry a code, a message and
 *          the measurement they fired on, so nothing is asserted without a number.
 */
export function assessForm(lm, baseline, opts = {}) {
  const empty = { faults: [], metrics: {} };
  if (!lm || !baseline) return empty;
  if (!lm[L_SHOULDER] || !lm[R_SHOULDER] || !lm[L_HIP] || !lm[R_HIP]) return empty;

  const { holdTarget = null, kneeAngle = null, exerciseName = "" } = opts;
  const torso = baseline.torso;
  const hips = mid(lm[L_HIP], lm[R_HIP]);
  const shoulders = mid(lm[L_SHOULDER], lm[R_SHOULDER]);

  const lean = trunkAngle(lm) - baseline.trunkAngle;
  // Negative y is upward in image coordinates, so a rising pelvis is positive.
  const hipLift = (baseline.hipY - hips.y) / torso;
  const sway = Math.abs(shoulders.x - baseline.shoulderX) / torso;

  const metrics = {
    trunkLeanDeg: Math.round(lean * 10) / 10,
    hipLiftFraction: Math.round(hipLift * 1000) / 1000,
    swayFraction: Math.round(sway * 1000) / 1000,
  };

  const faults = [];

  if (Math.abs(lean) > FORM.TRUNK_LEAN_DEG) {
    faults.push({
      code: "trunk_lean",
      message: lean < 0
        ? "You are leaning back. Keep your upper body still and let the leg do the work."
        : "You are leaning forward. Keep your upper body still and let the leg do the work.",
      value: metrics.trunkLeanDeg,
      threshold: FORM.TRUNK_LEAN_DEG,
    });
  }

  if (hipLift > FORM.HIP_LIFT_FRACTION) {
    faults.push({
      code: "hip_lift",
      message: "Your hips are lifting. Keep them down and raise the leg from the thigh.",
      value: metrics.hipLiftFraction,
      threshold: FORM.HIP_LIFT_FRACTION,
    });
  }

  // A hold that is supposed to be straight, performed bent, is a different
  // exercise doing different work - and the timer would happily run through it.
  if (holdTarget === "straight" && kneeAngle !== null && kneeAngle > FORM.KNEE_NOT_LOCKED_DEG) {
    faults.push({
      code: "knee_not_locked",
      message: "Straighten the knee fully. Push the back of it down.",
      value: Math.round(kneeAngle * 10) / 10,
      threshold: FORM.KNEE_NOT_LOCKED_DEG,
    });
  }

  // Only for a balance exercise: everywhere else, moving is the point.
  if (/balance/i.test(exerciseName) && sway > FORM.SWAY_FRACTION) {
    faults.push({
      code: "sway",
      message: "You are drifting. Fix your eyes on one spot and steady yourself.",
      value: metrics.swayFraction,
      threshold: FORM.SWAY_FRACTION,
    });
  }

  return { faults, metrics };
}

/**
 * Tracks how long each fault has persisted, so a wobble is not a correction.
 *
 * The tracker feeds every frame in and gets back only the faults that have held
 * long enough to be worth saying - the same shape of guard the camera-view gate
 * uses, and for the same reason.
 */
export function createFormMonitor(sustainFrames = FORM.SUSTAIN_FRAMES) {
  const streaks = new Map();
  const announced = new Set();

  return {
    /** @returns {Array} faults newly worth mentioning this frame. */
    update(faults) {
      const present = new Set(faults.map(f => f.code));

      for (const code of [...streaks.keys()]) {
        if (!present.has(code)) {
          streaks.delete(code);
          // Cleared: it becomes sayable again if it comes back, because it is
          // then a new mistake rather than the same one still happening.
          announced.delete(code);
        }
      }

      const ripe = [];
      for (const fault of faults) {
        const streak = (streaks.get(fault.code) || 0) + 1;
        streaks.set(fault.code, streak);
        if (streak >= sustainFrames && !announced.has(fault.code)) {
          announced.add(fault.code);
          ripe.push(fault);
        }
      }
      return ripe;
    },

    /** A new set: start listening from scratch. */
    reset() {
      streaks.clear();
      announced.clear();
    },

    /** Codes currently held, whether or not they have been announced. */
    active() {
      return [...streaks.keys()].filter(code => streaks.get(code) >= sustainFrames);
    },
  };
}
