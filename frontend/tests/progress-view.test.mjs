/*
 * The injected markup must not bring its own #content.
 *
 * progress.html and share.html each own a <div id="content" class="hidden">
 * and unhide it once data arrives. The view's markup used to open with a
 * second element carrying the same id and the same class, so the page unhid
 * the outer one and every chart stayed inside a hidden inner one — on both
 * pages, with no error anywhere.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("../assets/progress-view.js", import.meta.url), "utf8");
const markup = src.slice(src.indexOf("const MARKUP = `"), src.indexOf("`;", src.indexOf("const MARKUP = `")));

test("the view's markup does not repeat the host page's #content", () => {
  assert.ok(markup.length > 100, "could not find the MARKUP template");
  assert.doesNotMatch(markup, /id="content"/);
});

// The share page passes full timestamps (shared_at, expires_at) to longDate.
// Treating them as "YYYY-MM-DD" printed "Invalid Date" in the banner.
test("longDate reads a plain day and a full timestamp", () => {
  const sandbox = {};
  new Function("window", src)(sandbox);
  const { longDate } = sandbox.ProgressView;

  const day = longDate("2026-09-12");
  assert.doesNotMatch(day, /Invalid|NaN/);
  assert.match(day, /12/);

  const stamp = longDate("2026-09-12T12:00:00Z");
  assert.doesNotMatch(stamp, /Invalid|NaN/);
});
