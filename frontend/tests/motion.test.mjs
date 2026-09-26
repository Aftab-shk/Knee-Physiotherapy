/*
 * Tests for the scroll-reveal gate - assets/motion.js and assets/theme.css.
 *
 * Scroll reveal works by hiding content and then bringing it back. That is a
 * bargain with a sharp edge: every path where the "bring it back" half fails
 * leaves a patient staring at a blank panel where their grade, their range of
 * motion or their clinician's caution note should be. Nothing about that
 * failure looks like a bug from the inside - the page loads, the console is
 * clean, and the content is simply not there.
 *
 * So the invariants worth defending are the ones that decide whether hidden
 * content can ever get stranded, and they are checked here rather than only in
 * a browser because each has already been wrong once:
 *
 *   1. .reveal hides ONLY under .js-motion, so a failed, blocked or absent
 *      motion.js leaves every section plainly visible.
 *   2. The gate is never set for a reader who asked for reduced motion, or on
 *      a browser with no IntersectionObserver to un-hide with.
 *   3. The observer's rootMargin never shrinks the root's bottom edge. A
 *      negative value there carves out a dead band at the end of the document
 *      that nothing can scroll into, and anything tagged inside it stays
 *      hidden for good.
 *
 * Run:  node --test        (from frontend/)
 */

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

const motionSrc = readFileSync(new URL("../assets/motion.js", import.meta.url), "utf8");
const themeSrc  = readFileSync(new URL("../assets/theme.css", import.meta.url), "utf8");

/* Run motion.js against a stand-in for the browser. Returns what it did to
   <html> plus the options it handed IntersectionObserver. */
function run({ reducedMotion = false, hasIO = true } = {}) {
  const classes = new Set();
  let ioOptions = null;

  const html = {
    classList: {
      add: (c) => classes.add(c),
      contains: (c) => classes.has(c),
    },
  };
  const body = {
    nodeType: 1,
    matches: () => false,
    querySelectorAll: () => [],
  };
  const doc = {
    documentElement: html,
    body,
    readyState: "complete",
    addEventListener() {},
  };

  const win = {
    matchMedia: (q) => ({ matches: reducedMotion && q.includes("reduce") }),
    document: doc,
    MutationObserver: class { observe() {} },
  };
  if (hasIO) {
    win.IntersectionObserver = class {
      constructor(_cb, opts) { ioOptions = opts; }
      observe() {} unobserve() {}
    };
  }

  new Function("window", "document", "MutationObserver", "IntersectionObserver", motionSrc)(
    win, doc, win.MutationObserver, win.IntersectionObserver,
  );

  return { gated: classes.has("js-motion"), ioOptions };
}

// ── The gate decides whether anything is hidden at all ──────────────────────

test("a normal browser gets the gate, so the reveal can run", () => {
  assert.equal(run().gated, true);
});

test("a reader who asked for reduced motion is never gated", () => {
  // Not "animated faster" - never hidden in the first place.
  assert.equal(run({ reducedMotion: true }).gated, false);
});

test("without IntersectionObserver nothing is hidden", () => {
  // There would be nothing left to un-hide it with.
  assert.equal(run({ hasIO: false }).gated, false);
});

// ── Content must never be strandable ────────────────────────────────────────

test("the observer's rootMargin never shrinks the bottom of the root", () => {
  const { ioOptions } = run();
  assert.ok(ioOptions, "IntersectionObserver was constructed with options");

  const parts = String(ioOptions.rootMargin).trim().split(/\s+/);
  assert.equal(parts.length, 4, `expected 4 rootMargin values, got "${ioOptions.rootMargin}"`);

  const bottom = parseFloat(parts[2]);
  assert.ok(
    bottom >= 0,
    `rootMargin bottom is ${parts[2]}; a negative value strands anything ` +
    `sitting in the last stretch of the document, because the page cannot ` +
    `be scrolled far enough to push it into the shrunken root`,
  );
});

test("theme.css hides .reveal only behind the gate", () => {
  // A bare ".reveal { opacity: 0 }" would blank the page whenever motion.js
  // fails to load - the exact failure this whole design is arranged to avoid.
  const rules = themeSrc.match(/(^|\})\s*([^{}]*\.reveal[^{}]*)\{([^}]*)\}/g) || [];
  assert.ok(rules.length, "expected at least one .reveal rule in theme.css");

  for (const rule of rules) {
    const selector = rule.slice(rule.indexOf("}") + 1, rule.indexOf("{")).trim() || rule.slice(0, rule.indexOf("{")).trim();
    const body = rule.slice(rule.indexOf("{") + 1, rule.lastIndexOf("}"));
    if (/opacity\s*:\s*0/.test(body)) {
      assert.ok(
        selector.includes(".js-motion"),
        `"${selector}" hides content without requiring .js-motion`,
      );
    }
  }
});
