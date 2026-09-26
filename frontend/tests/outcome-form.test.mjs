/*
 * Tests for the KOOS-JR questionnaire - assets/outcome-form.js.
 *
 * The failure mode worth designing against is a form that submits something it
 * should not. KOOS-JR has no published rule for a missing item, so six answers
 * out of seven cannot be scored - and a client that quietly filled the gap would
 * be inventing clinical data that then looks exactly like a real reading, on the
 * same scale a joint registry uses.
 *
 * So most of what follows checks that it refuses: no partial submissions, no
 * padding, no treating a skipped question as a zero (which is "no symptoms at
 * all" - the single most flattering answer on the scale).
 *
 * The decision functions are pure, so they are tested directly. `render()` is
 * the thin part that touches the DOM.
 *
 * Run:  node --test        (from frontend/)
 */

import assert from "node:assert/strict";
import test from "node:test";

const sandbox = {};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

const { readFileSync } = await import("node:fs");
const source = readFileSync(new URL("../assets/outcome-form.js", import.meta.url), "utf8");
new Function("window", "globalThis", "var window=arguments[0];" + source)(sandbox, sandbox);

const F = sandbox.OutcomeForm;

const ITEMS = 7;
const COMPLETE = [1, 2, 0, 3, 1, 2, 1];

// ── Completeness ────────────────────────────────────────────────────────────

test("a fully answered form is complete", () => {
  assert.equal(F.isComplete(COMPLETE, ITEMS), true);
  assert.deepEqual(F.unanswered(COMPLETE, ITEMS), []);
});

test("an untouched form is not complete", () => {
  assert.equal(F.isComplete([null, null, null, null, null, null, null], ITEMS), false);
});

test("a skipped question is reported by position", () => {
  const answers = [0, 1, null, 2, null, 3, 1];
  assert.deepEqual(F.unanswered(answers, ITEMS), [2, 4]);
  assert.equal(F.isComplete(answers, ITEMS), false);
});

test("zero is an answer, not a blank", () => {
  /*
   * "None" scores 0 - the most favourable response on the scale. Treating it as
   * unanswered would block a patient with a good knee from ever submitting; the
   * inverse mistake, treating a blank as 0, would score seven skipped questions
   * as a perfect knee.
   */
  const allNone = [0, 0, 0, 0, 0, 0, 0];
  assert.deepEqual(F.unanswered(allNone, ITEMS), []);
  assert.equal(F.isComplete(allNone, ITEMS), true);
});

test("undefined and non-integers do not count as answers", () => {
  assert.equal(F.isComplete([0, 1, undefined, 2, 3, 4, 0], ITEMS), false);
  assert.equal(F.isComplete([0, 1, "2", 2, 3, 4, 0], ITEMS), false);
  assert.equal(F.isComplete([0, 1, 1.5, 2, 3, 4, 0], ITEMS), false);
});

test("more answers than items does not make a short form complete", () => {
  // Only the first `itemCount` positions are the questionnaire.
  assert.equal(F.isComplete([1, 2, 3], ITEMS), false);
});

// ── Progress wording ────────────────────────────────────────────────────────

test("an untouched form says how long it will take, not what is missing", () => {
  const text = F.progressText([null, null, null, null, null, null, null], ITEMS);
  assert.match(text, /7 questions/);
  assert.match(text, /minute/);
});

test("a part-finished form counts what is left, not what is done", () => {
  // "2 left" is an instruction. "5 of 7" makes the reader do the subtraction to
  // find out whether they can submit yet.
  assert.equal(F.progressText([0, 1, 2, 3, 4, null, null], ITEMS), "2 left");
  assert.equal(F.progressText([0, null, null, null, null, null, null], ITEMS), "6 left");
});

test("a finished form says so", () => {
  assert.equal(F.progressText(COMPLETE, ITEMS), "All answered");
});

// ── Building the request ────────────────────────────────────────────────────

test("a complete form produces the request body the API expects", () => {
  const payload = F.payloadFor(COMPLETE, ITEMS, "left");
  assert.deepEqual(payload.responses, COMPLETE);
  assert.equal(payload.knee_side, "left");
});

test("a partial form throws rather than padding itself out", () => {
  /*
   * The server refuses a partial form too. A client that quietly filled the gaps
   * would be inventing an answer on the patient's behalf, on a scale that is
   * meant to be comparable with a registry's.
   */
  assert.throws(
    () => F.payloadFor([0, 1, 2, null, 4, 0, 1], ITEMS, "right"),
    /All 7 questions/,
  );
});

test("the payload is a copy, so later edits cannot rewrite a sent request", () => {
  const answers = COMPLETE.slice();
  const payload = F.payloadFor(answers, ITEMS, "right");
  answers[0] = 4;
  assert.equal(payload.responses[0], COMPLETE[0]);
});

// ── Describing a change ─────────────────────────────────────────────────────

test("a rising score is better, because higher is better on this scale", () => {
  // The single easiest thing to get backwards: the raw sum runs the other way.
  assert.equal(F.deltaText({ delta: 17, direction: "improved" }), "17 points better");
});

test("a falling score is worse", () => {
  assert.equal(F.deltaText({ delta: -14, direction: "declined" }), "14 points worse");
});

test("a change inside the instrument's noise is not reported as a change", () => {
  /*
   * The server has already decided this one. Rendering it as "3 points worse"
   * would invite a conclusion the questionnaire cannot support - week-to-week
   * movement that small is measurement noise, not a knee getting worse.
   */
  assert.equal(F.deltaText({ delta: -3, direction: "unchanged" }), "about the same");
  assert.equal(F.deltaText({ delta: 2, direction: "unchanged" }), "about the same");
});

test("a first score has nothing to compare against and says nothing", () => {
  assert.equal(F.deltaText({ delta: null, direction: "first" }), "");
  assert.equal(F.deltaText(null), "");
  assert.equal(F.deltaText(undefined), "");
});

// ── The rendered form ───────────────────────────────────────────────────────
//
// The gating lives in render(): submit stays disabled until every question is
// answered, and a failed request has to leave the answers on screen. Neither is
// visible from the pure functions, and both are what a patient actually meets.
//
// Rather than a DOM library - this project has no build step and no
// dependencies - a shim of the handful of methods the module touches. It is
// enough to click through a form, and small enough to read.

function makeDocument() {
  function node(tag) {
    const self = {
      tagName: tag.toUpperCase(),
      children: [],
      ownText: "",
      className: "",
      attrs: {},
      events: {},
      classList: {
        set: new Set(),
        add(c) { this.set.add(c); },
        remove(c) { this.set.delete(c); },
        contains(c) { return this.set.has(c); },
      },
      appendChild(child) { self.children.push(child); return child; },
      append(...kids) { kids.forEach(self.appendChild); },
      setAttribute(k, v) { self.attrs[k] = String(v); },
      getAttribute(k) { return k in self.attrs ? self.attrs[k] : null; },
      addEventListener(type, fn) { (self.events[type] = self.events[type] || []).push(fn); },
      fire(type, event = {}) {
        for (const fn of self.events[type] || []) fn({ preventDefault() {}, ...event });
      },
      get textContent() {
        return self.ownText + self.children.map((c) => c.textContent).join("");
      },
      set textContent(v) { self.children.length = 0; self.ownText = v; },
    };
    return self;
  }
  return { createElement: node };
}

/** Every node in the tree matching `predicate`, depth-first. */
function findAll(root, predicate) {
  const out = [];
  (function walk(n) {
    if (predicate(n)) out.push(n);
    n.children.forEach(walk);
  })(root);
  return out;
}

const byClass = (root, cls) =>
  findAll(root, (n) => String(n.className).split(" ").includes(cls));

const DEFINITION = {
  name: "KOOS-JR",
  recall_period: "the last week",
  response_options: [
    { value: 0, label: "None" }, { value: 1, label: "Mild" }, { value: 2, label: "Moderate" },
    { value: 3, label: "Severe" }, { value: 4, label: "Extreme" },
  ],
  items: [
    { code: "S1", section: "Stiffness", lead_in: "How severe…", prompt: "After first waking" },
    { code: "P2", section: "Pain", lead_in: "What amount of pain…", prompt: "Twisting" },
    { code: "P3", section: "Pain", lead_in: "What amount of pain…", prompt: "Straightening" },
    { code: "P5", section: "Pain", lead_in: "What amount of pain…", prompt: "Stairs" },
    { code: "P6", section: "Pain", lead_in: "What amount of pain…", prompt: "Standing" },
    { code: "A4", section: "Function", lead_in: "What difficulty…", prompt: "Rising from sitting" },
    { code: "A6", section: "Function", lead_in: "What difficulty…", prompt: "Bending to the floor" },
  ],
  caveat: null,
};

function mount(definition = DEFINITION, options = {}) {
  sandbox.document = makeDocument();
  const host = sandbox.document.createElement("div");
  const handle = F.render(host, definition, options);
  return {
    host,
    handle,
    form: byClass(host, "om-form")[0],
    radios: findAll(host, (n) => n.tagName === "INPUT"),
  };
}

const submitButton = (host) =>
  findAll(host, (n) => n.tagName === "BUTTON" && n.type === "submit")[0];

/** Answer question `index` with option `value`, as a click would. */
function answer(radios, index, value) {
  radios[index * 5 + value].fire("change");
}

test("every question is rendered, with every option", () => {
  const { host, radios } = mount();
  assert.equal(byClass(host, "om-item").length, 7);
  assert.equal(radios.length, 7 * 5);
});

test("items sharing a question stem are grouped under one copy of it", () => {
  /*
   * Four of the seven items share "What amount of knee pain have you had
   * when…". Repeating that above each one is how a seven-question form starts
   * feeling like twenty.
   */
  const { host } = mount();
  assert.equal(byClass(host, "om-group").length, 3);   // stiffness, pain, function
});

test("the item wording comes from the server, never from this file", () => {
  const { host } = mount();
  const prompts = byClass(host, "om-prompt").map((n) => n.textContent);
  assert.deepEqual(prompts, DEFINITION.items.map((i) => i.prompt));
});

test("submit is disabled until the last question is answered", () => {
  const { host, radios } = mount();
  const submit = submitButton(host);

  assert.equal(submit.disabled, true, "disabled on an untouched form");
  for (let i = 0; i < 6; i++) {
    answer(radios, i, 2);
    assert.equal(submit.disabled, true, `still disabled with ${i + 1} of 7 answered`);
  }
  answer(radios, 6, 2);
  assert.equal(submit.disabled, false, "enabled once all seven are answered");
});

test("the status line counts down as questions are answered", () => {
  const { host, radios } = mount();
  const status = byClass(host, "om-status")[0];

  assert.match(status.textContent, /7 questions/);
  answer(radios, 0, 0);
  assert.equal(status.textContent, "6 left");
  for (let i = 1; i < 7; i++) answer(radios, i, 0);
  assert.equal(status.textContent, "All answered");
});

test("answering 'None' still counts, though it scores zero", () => {
  // The most favourable answer on the scale. Treating it as unanswered would
  // lock a patient with a good knee out of submitting at all.
  const { host, radios } = mount();
  for (let i = 0; i < 7; i++) answer(radios, i, 0);
  assert.equal(submitButton(host).disabled, false);
});

test("submitting hands over the answers in item order", async () => {
  let received = null;
  const { radios, form } = mount(DEFINITION, {
    kneeSide: "left",
    onSubmit: async (payload) => { received = payload; },
  });

  const chosen = [1, 2, 0, 3, 1, 2, 4];
  chosen.forEach((v, i) => answer(radios, i, v));
  await form.fire("submit");

  assert.deepEqual(received.responses, chosen);
  assert.equal(received.knee_side, "left");
});

test("an incomplete form does not submit even if the event fires", async () => {
  let called = false;
  const { radios, form } = mount(DEFINITION, { onSubmit: async () => { called = true; } });
  answer(radios, 0, 1);
  await form.fire("submit");
  assert.equal(called, false);
});

test("a failed save keeps the answers and lets the patient try again", async () => {
  /*
   * Nobody should have to answer seven questions twice because a request timed
   * out - which is exactly what clearing the form on error would cost them.
   */
  const { host, radios, form } = mount(DEFINITION, {
    onSubmit: async () => { throw new Error("The request timed out."); },
  });

  const chosen = [1, 2, 0, 3, 1, 2, 4];
  chosen.forEach((v, i) => answer(radios, i, v));
  await form.fire("submit");

  assert.equal(byClass(host, "om-item").length, 7, "the questions are still on screen");
  assert.equal(submitButton(host).disabled, false, "and can be submitted again");

  const error = byClass(host, "om-error")[0];
  assert.match(error.textContent, /timed out/);
  assert.equal(error.classList.contains("hidden"), false);
});

test("a caveat about the instrument is shown above the questions", () => {
  const { host } = mount({ ...DEFINITION, caveat: "Not validated after ACL reconstruction." });
  const notice = byClass(host, "notice")[0];
  assert.match(notice.textContent, /ACL reconstruction/);
});

test("no caveat means no notice", () => {
  assert.equal(byClass(mount().host, "notice").length, 0);
});
