/*
 * Tests for the tracker's spoken cues — assets/voice.js.
 *
 * The failure mode worth designing against is not silence, it is chatter. A
 * coach that calls out every one of thirty reps gets muted, and a muted coach
 * says nothing when the knee goes past its limit — which is the one cue the
 * whole feature exists for.
 *
 * So most of what follows checks that it stays quiet: no counting through long
 * sets, no repeating itself, nothing spoken twice for the same event.
 *
 * The cue functions are pure — they decide *what to say*, not how — so they are
 * tested directly. `speak()` is the thin part that touches the browser.
 *
 * Run:  node --test        (from frontend/)
 */

import assert from "node:assert/strict";
import test from "node:test";

// A minimal browser: enough for the module to load and remember a preference.
const store = new Map();
const sandbox = {
  localStorage: {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

const { readFileSync } = await import("node:fs");
const source = readFileSync(new URL("../assets/voice.js", import.meta.url), "utf8");
new Function("window", "globalThis", "var window=arguments[0];" + source)(sandbox, sandbox);

const V = sandbox.PhysioVoice;

// ── Rep counting ────────────────────────────────────────────────────────────

test("a short set is counted rep by rep", () => {
  const spoken = [];
  for (let rep = 1; rep <= 10; rep++) {
    const cue = V.cueForRep(rep, 10);
    if (cue) spoken.push(cue.text);
  }
  assert.equal(spoken.length, 10, "every rep of ten is worth saying");
  assert.equal(spoken.at(-1), "Set complete.");
});

test("a long set is not counted number by number", () => {
  const spoken = [];
  for (let rep = 1; rep <= 30; rep++) {
    const cue = V.cueForRep(rep, 30);
    if (cue) spoken.push(rep);
  }
  assert.ok(spoken.length < 12, `spoke ${spoken.length} times in a set of 30`);
  // …but the milestones and the closing reps are still called.
  assert.ok(spoken.includes(10) && spoken.includes(20));
  assert.ok(spoken.includes(28) && spoken.includes(29) && spoken.includes(30));
});

test("the last few reps are counted down, not up", () => {
  assert.equal(V.cueForRep(8, 10).text, "2 to go");
  assert.equal(V.cueForRep(5, 10).text, "5");
});

test("finishing the set is announced as such", () => {
  assert.equal(V.cueForRep(12, 12).text, "Set complete.");
});

test("a rep count below one says nothing", () => {
  assert.equal(V.cueForRep(0, 10), null);
  assert.equal(V.cueForRep(-1, 10), null);
});

test("every rep cue carries a key, so nothing is said twice", () => {
  const keys = new Set();
  for (let rep = 1; rep <= 20; rep++) {
    const cue = V.cueForRep(rep, 20);
    if (!cue) continue;
    assert.ok(cue.key, "a cue without a key cannot be de-duplicated");
    assert.ok(!keys.has(cue.key), `duplicate key ${cue.key}`);
    keys.add(cue.key);
  }
});

// ── Holds ───────────────────────────────────────────────────────────────────

test("a hold starts with an instruction, not a number", () => {
  assert.equal(V.cueForHold(10, 10).text, "Hold it.");
});

test("a hold counts down only at the useful moments", () => {
  const spoken = [];
  for (let left = 10; left >= 1; left--) {
    const cue = V.cueForHold(left, 30);
    if (cue) spoken.push(cue.text);
  }
  assert.deepEqual(spoken, ["10", "5", "3", "2", "1"]);
});

test("a finished hold says so", () => {
  assert.match(V.cueForHold(0, 10).text, /complete/i);
});

test("fractional seconds round up, so nothing is skipped mid-tick", () => {
  assert.equal(V.cueForHold(4.6, 30).text, "5");
});

// ── Rest between sets ───────────────────────────────────────────────────────

test("a rest countdown names its units until the final seconds", () => {
  assert.equal(V.cueForRest(30, 2).text, "30 seconds");
  assert.equal(V.cueForRest(10, 2).text, "10 seconds");
  // Bare numbers only at the very end, where the context is obvious.
  assert.equal(V.cueForRest(3, 2).text, "3");
  assert.equal(V.cueForRest(1, 2).text, "1");
});

test("the end of a rest tells the patient to start", () => {
  assert.match(V.cueForRest(0, 1).text, /next set/i);
  assert.match(V.cueForRest(0, 0).text, /begin/i);
});

test("a rest says nothing between its countdown points", () => {
  assert.equal(V.cueForRest(27, 2), null);
  assert.equal(V.cueForRest(15, 2), null);
  assert.equal(V.cueForRest(7, 2), null);
});

// ── Sets ────────────────────────────────────────────────────────────────────

test("finishing a set says which one and how long the rest is", () => {
  const cue = V.cueForSetComplete(1, 3, 45);
  assert.match(cue.text, /set 1 of 3/i);
  assert.match(cue.text, /45 seconds/);
});

test("the last set is not followed by a rest instruction", () => {
  const cue = V.cueForSetComplete(3, 3, 45);
  assert.match(cue.text, /last set/i);
  assert.doesNotMatch(cue.text, /rest/i);
});

// ── Safety ──────────────────────────────────────────────────────────────────

test("a breach interrupts whatever else was being said", () => {
  const cue = V.cueForBreach(60);
  assert.equal(cue.interrupt, true, "a rep count must not bury this");
  assert.match(cue.text, /ease off/i);
  assert.match(cue.text, /60/, "the limit is named, so it means something");
});

test("clearing a breach is acknowledged but does not interrupt", () => {
  const cue = V.cueForBreachCleared();
  assert.ok(cue.text);
  assert.notEqual(cue.interrupt, true);
});

// ── The preference ──────────────────────────────────────────────────────────

test("voice is on by default", () => {
  assert.equal(V.isEnabled(), true);
});

test("turning it off is remembered", () => {
  V.setEnabled(false);
  assert.equal(V.isEnabled(), false);
  assert.equal(store.get("physioai_voice"), "off");

  V.setEnabled(true);
  assert.equal(store.get("physioai_voice"), "on");
});

test("speaking while muted does nothing and does not throw", () => {
  V.setEnabled(false);
  assert.equal(V.speak("hello"), false);
  V.setEnabled(true);
});

test("speaking without browser speech support fails quietly", () => {
  // No speechSynthesis in this sandbox, so support is absent by construction.
  assert.equal(V.supported(), false);
  assert.equal(V.speak("hello"), false, "a session must not depend on this working");
});

test("cancelling without speech support does not throw", () => {
  assert.doesNotThrow(() => V.cancel());
});
