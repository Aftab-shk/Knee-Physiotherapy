/*
 * pose-gate.js — joint geometry and camera-view validation for the tracker.
 * ==========================================================================
 *
 * Extracted from tracker.html so the geometry that decides whether a knee angle
 * can be trusted is testable on its own. See frontend/tests/pose-gate.test.mjs.
 *
 * Why the validation exists
 * -------------------------
 * calcKneeAngle() measures a *projected* angle: it uses only the x and y of each
 * landmark, so it is the angle of the leg as it appears in the image, not the
 * angle of the leg in space. The two coincide only when the camera looks along
 * the knee's axis of rotation — that is, when it sees the leg side-on.
 *
 * Square to the camera, femur and tibia project onto nearly the same line: a
 * knee bent to 90 degrees reads close to 0. Nothing about that reading looks
 * wrong. The HUD shows a small number, the ceiling is never crossed, and the
 * alarm meant to protect the joint stays silent for the whole session.
 *
 * Under-reading is the dangerous direction, and it is the direction a bad camera
 * angle fails in. So the session does not start until the view has been checked,
 * and it stops if the view is lost.
 *
 * MediaPipe Pose landmark indices used here:
 *   11/12 shoulder, 23/24 hip, 25/26 knee, 27/28 ankle, 31/32 foot index
 */

export const POSE_GATE = {
  SPREAD_OK:          0.38,   // max(shoulder,hip) spread ÷ torso; side-on ≈ 0.1–0.3
  SPREAD_RECOVER:     0.46,   // hysteresis: harder to lose the lock than to gain it
  // ° of disagreement allowed between the 3D and 2D knee angle. Was 22, which
  // had to be that wide because the 2D angle was measured in unsquared
  // coordinates and disagreed with its own 3D reference by ~16° on a 16:9
  // camera before the leg turned at all. With squareUp() in place a properly
  // side-on leg agrees to within a degree, so the budget can go back to
  // catching what it is for. 12 still passes a patient standing 50° off axis.
  FORESHORTEN_MAX:    12,
  VISIBILITY_MIN:     0.5,    // per-landmark confidence floor
  FRAME_MARGIN:       0.02,   // normalised coords; nearer an edge than this counts as clipped
  TORSO_MIN:          0.06,   // shorter than this ⇒ too far away, or an overhead view
  CALIBRATION_FRAMES: 36,     // ~1.2 s of a good view before tracking starts
  SUSPEND_FRAMES:     12,     // ~0.4 s of a bad view before counting stops
  RESUME_FRAMES:      15,     // ~0.5 s of a good view before it picks up again
  ESCAPE_AFTER_MS:    20000,  // offer the manual override after this long stuck
  // No new camera frame for this long means the feed has stopped, not that the
  // patient is holding still. A webcam delivers 15-30 fps, so 700 ms is over
  // ten missed frames: long enough that a hiccup does not trip it, short enough
  // that a frozen picture cannot pass for a live one while the knee keeps
  // bending. See handleFeedStall() in tracker.html.
  FEED_STALL_MS:      700,
};

// Codes that mean there is no usable angle at all, as opposed to an angle
// measured from a viewpoint we cannot vouch for. A patient who overrides the
// view gate is still held to these.
export const HARD_FAIL_CODES = Object.freeze(['no-pose', 'low-visibility', 'out-of-frame']);

export const isHardFail = (code) => HARD_FAIL_CODES.includes(code);

// ── Geometry ────────────────────────────────────────────────────────────────

/**
 * Undo MediaPipe's anisotropic normalisation.
 *
 * Landmarks come back normalised to 0–1 by frame *width* for x and frame
 * *height* for y. On anything but a square frame those are different real
 * distances, and every piece of geometry below — angles, lengths, ratios —
 * silently reads the leg as though the image had been squashed. A true 60°
 * knee measured this way reports 44° on a 1280×720 webcam, and under-reading
 * is the direction that lets a patient past their ceiling with no alarm.
 *
 * Dividing y by the frame's aspect (width ÷ height) puts it back in the same
 * units as x. `aspect` of 1 is a square frame and leaves everything untouched,
 * which is what the synthetic bodies in the tests use.
 */
const squareUp = (p, aspect) =>
  (!p || aspect === 1) ? p : { ...p, y: p.y / aspect };

const dist2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid2  = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

export function jointAngle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x ** 2 + v1.y ** 2);
  const mag2 = Math.sqrt(v2.x ** 2 + v2.y ** 2);
  if (mag1 === 0 || mag2 === 0) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return Math.acos(cos) * 180 / Math.PI;
}

// The same cosine rule in three dimensions. MediaPipe's world landmarks are
// metric and hip-centred, so this is a true anatomical angle rather than a
// projected one. It is noisier than the 2D value — which is why it does not
// drive the safety check — but it cannot be fooled by foreshortening, which
// makes it a useful cross-check.
export function jointAngle3D(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dp = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 === 0 || m2 === 0) return null;
  return Math.acos(Math.max(-1, Math.min(1, dp / (m1 * m2)))) * 180 / Math.PI;
}

// Knee flexion: straight leg = 0°, fully bent ≈ 135°.
// `aspect` is the frame's width ÷ height — pass it or the reading is distorted
// by the frame's shape rather than by the knee. See squareUp().
export function calcKneeAngle(hip, knee, ankle, aspect = 1) {
  return 180 - jointAngle(squareUp(hip, aspect), squareUp(knee, aspect), squareUp(ankle, aspect));
}

// Signed ankle deviation from neutral: positive = plantarflexion (toes pointed
// away), negative = dorsiflexion (toes pulled up). Kept signed so a pump reads
// as one clean oscillation through zero rather than two peaks. Ankle Pumps is
// the only exercise counted off this.
export const ANKLE_NEUTRAL_DEG = 90;
export function calcAnkleAngle(knee, ankle, footIndex, aspect = 1) {
  return jointAngle(squareUp(knee, aspect), squareUp(ankle, aspect), squareUp(footIndex, aspect))
    - ANKLE_NEUTRAL_DEG;
}

// 'both' has no single tracked leg, so it falls back to the right side. Any
// guidance text shown to the patient has to agree with this choice.
export function landmarkIndices(kneeSide) {
  const useLeft = kneeSide === 'left';
  return {
    hip:   useLeft ? 23 : 24,
    knee:  useLeft ? 25 : 26,
    ankle: useLeft ? 27 : 28,
    foot:  useLeft ? 31 : 32,
    side:  useLeft ? 'left' : 'right',
  };
}

// ── View validation ─────────────────────────────────────────────────────────

/**
 * Decide whether a knee angle measured from this frame can be trusted.
 *
 * @param {Array}  lm     normalised 2D landmarks (MediaPipe `landmarks[0]`)
 * @param {Array}  world  metric 3D landmarks (`worldLandmarks[0]`), or null
 * @param {Object} idx    from landmarkIndices()
 * @param {Object} opts   { spreadCeiling, trackedJoint }
 * @returns {{ok: boolean, code: string, message: string, visibility: number}}
 *          `code` is stable for logic; `message` is the single next thing the
 *          patient should do about it; `visibility` is the weakest landmark the
 *          angle depends on, which is how well the reading can be trusted at
 *          all — the tracker averages it across a set and stores it beside the
 *          measurement.
 */
export function assessView(lm, world, idx, opts = {}) {
  const spreadCeiling = opts.spreadCeiling ?? POSE_GATE.SPREAD_OK;
  const trackedJoint  = opts.trackedJoint ?? 'knee';
  // Frame width ÷ height. Every length and angle below is measured after
  // squaring the coordinates up, so the thresholds mean the same thing on a
  // 16:9 webcam, a 4:3 one and a portrait phone.
  const aspect        = opts.aspect ?? 1;

  if (!lm || [11, 12, 23, 24, idx.hip, idx.knee, idx.ankle].some(i => !lm[i])) {
    return { ok: false, code: 'no-pose', visibility: 0, message: 'Step into frame — we can’t see you yet.' };
  }

  const hip = lm[idx.hip], knee = lm[idx.knee], ankle = lm[idx.ankle];

  // 1. Every joint the angle is built from has to be confidently located. Ankle
  //    Pumps counts off the foot as well, so that is included when it is the
  //    joint being tracked.
  const legVis  = Math.min(hip.visibility ?? 1, knee.visibility ?? 1, ankle.visibility ?? 1);
  const footVis = trackedJoint === 'ankle' ? (lm[idx.foot]?.visibility ?? 0) : 1;
  const visibility = Math.min(legVis, footVis);

  if (visibility < POSE_GATE.VISIBILITY_MIN) {
    return {
      ok: false, code: 'low-visibility', visibility,
      message: footVis < legVis
        ? 'Your foot needs to be clearly visible to count ankle pumps.'
        : `Your ${idx.side} hip, knee and ankle all need to be clearly visible.`,
    };
  }

  // 2. …and sit inside the frame rather than clipped against an edge, where the
  //    landmark is extrapolated and drifts.
  const M = POSE_GATE.FRAME_MARGIN;
  if ([hip, knee, ankle].some(p => p.x < M || p.x > 1 - M || p.y < M || p.y > 1 - M)) {
    return { ok: false, code: 'out-of-frame', visibility, message: 'Move back — your whole leg needs to be in frame.' };
  }

  // 3. Torso length is the scale reference for the test below. A very short one
  //    means the camera is far away, or looking down the length of the body,
  //    and neither gives a measurable knee.
  const sq = p => squareUp(p, aspect);
  const torso = dist2(mid2(sq(lm[23]), sq(lm[24])), mid2(sq(lm[11]), sq(lm[12])));
  if (torso < POSE_GATE.TORSO_MIN) {
    return { ok: false, code: 'too-far', visibility, message: 'Move closer, or bring the camera down to knee height.' };
  }

  // 4. The side-on test. Seen from the side, the left and right hips (and
  //    shoulders) project onto almost the same point; seen square-on they are a
  //    shoulder-width apart. Dividing by torso length makes the ratio
  //    independent of how far away the patient is and of their build.
  const spread = Math.max(dist2(sq(lm[23]), sq(lm[24])), dist2(sq(lm[11]), sq(lm[12]))) / torso;
  if (spread > spreadCeiling) {
    return { ok: false, code: 'not-sagittal', visibility, message: 'Turn side-on to the camera — we need a profile view of your leg.' };
  }

  // 5. Backstop for the oblique angles step 4 lets through: compare the
  //    projected knee angle against the metric one. A large gap means the leg is
  //    pointing towards or away from the camera, and the 2D number — the one the
  //    safety check uses — is reading low.
  if (world && world[idx.hip] && world[idx.knee] && world[idx.ankle]) {
    const raw3d = jointAngle3D(world[idx.hip], world[idx.knee], world[idx.ankle]);
    if (raw3d !== null &&
        Math.abs((180 - raw3d) - calcKneeAngle(hip, knee, ankle, aspect)) > POSE_GATE.FORESHORTEN_MAX) {
      return { ok: false, code: 'foreshortened', visibility, message: 'Your leg is pointing towards the camera. Turn so it lies across the view.' };
    }
  }

  return { ok: true, code: 'ok', visibility, message: 'Hold still…' };
}
