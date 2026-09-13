/*
 * Tests for the service worker's caching policy — sw.js.
 *
 * The routing rules are the part of a service worker that can quietly do harm.
 * A cached prescription is a cached *movement ceiling*: a clinician lowers a
 * limit, and the patient's phone goes on handing them yesterday's number with
 * nothing to say it is out of date.
 *
 * So the rule these defend is blunt — **nothing clinical is ever served from
 * cache** — and it is tested here rather than only inside a browser, because
 * "we thought it was network-only" is not a thing anyone can check by looking.
 *
 * Run:  node --test        (from frontend/)
 */

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

const ORIGIN = "https://physio.example";

// Load sw.js with a stand-in for the worker globals it registers against.
const sandbox = {
  addEventListener() {},
  skipWaiting() {},
  clients: { claim() {} },
  location: { origin: ORIGIN },
  caches: { open: async () => ({}), keys: async () => [], match: async () => undefined },
  URL,
  fetch: async () => ({}),
};
sandbox.self = sandbox;

const source = readFileSync(new URL("../sw.js", import.meta.url), "utf8");
new Function("self", "caches", "URL", "fetch", source)(
  sandbox, sandbox.caches, URL, sandbox.fetch,
);

const { routeFor } = sandbox;

const get = (url, mode = "cors") => ({ method: "GET", url, mode });
const route = (url, mode) => routeFor(get(url, mode), ORIGIN);


// ── Nothing clinical is cached ──────────────────────────────────────────────

test("a same-origin API call is never served from cache", () => {
  for (const path of [
    "/analyse-xray",
    "/exercises?surgery_type=tkr",
    "/sessions",
    "/sessions/sets",
    "/me/progress",
    "/me/prescriptions/abc",
    "/me/flags",
    "/auth/me",
    "/clinician/patients",
    "/share/sometoken",
    "/health",
  ]) {
    assert.equal(route(ORIGIN + path), "network-only", path);
  }
});

test("an API on another origin is never cached either", () => {
  // The usual deployment: frontend on one host, backend on another.
  assert.equal(route("https://api.physio.example/analyse-xray"), "network-only");
  assert.equal(route("http://127.0.0.1:8000/me/progress"), "network-only");
});

test("the appointment summary is never served from cache", () => {
  // It ends in .pdf, and .pdf would otherwise fall through to the cache-first
  // branch that exists for static assets. The API-path rule catches it first,
  // and this pins that ordering: a summary served from cache would be last
  // month's figures printed under today's date, with nothing to show it.
  for (const path of ["/me/summary.pdf",
                      "/share/abc123/summary.pdf",
                      "/clinician/patients/p1/summary.pdf"]) {
    assert.equal(route(ORIGIN + path), "network-only", path);
  }
});


test("a prescription cannot be served stale", () => {
  // Stated on its own because it is the specific harm: an old prescription is
  // an old safe-angle ceiling, applied by the tracker without comment.
  assert.equal(route(ORIGIN + "/me/prescriptions/xyz"), "network-only");
});

test("nothing but GET is touched", () => {
  for (const method of ["POST", "PATCH", "DELETE", "PUT"]) {
    assert.equal(routeFor({ method, url: ORIGIN + "/upload.html", mode: "cors" }, ORIGIN),
                 "network-only", method);
  }
});


// ── Pages are fresh when they can be ────────────────────────────────────────

test("a page is fetched before it is cached", () => {
  assert.equal(route(ORIGIN + "/tracker.html"), "network-first");
  assert.equal(route(ORIGIN + "/upload.html"), "network-first");
  assert.equal(route(ORIGIN + "/"), "network-first");
});

test("a navigation is network-first whatever its path looks like", () => {
  assert.equal(route(ORIGIN + "/progress", "navigate"), "network-first");
});

test("an old tracker page is a fallback, never a preference", () => {
  // tracker.html carries the safety logic — the camera-view gate, the angle
  // ceiling, the alarm. Serving a cached copy in preference to a live one would
  // run last month's safety rules on this month's prescription.
  assert.equal(route(ORIGIN + "/tracker.html"), "network-first");
});


// ── Static assets are cached ────────────────────────────────────────────────

test("the shell is cache-first", () => {
  for (const path of [
    "/assets/theme.css",
    "/assets/pose-gate.js",
    "/assets/voice.js",
    "/assets/form-check.js",
    "/assets/progress-view.js",
    "/logo.png",
    "/assets/icons/icon-192.png",
  ]) {
    assert.equal(route(ORIGIN + path), "cache-first", path);
  }
});

test("a third-party script is not cached by us", () => {
  // MediaPipe's wasm and Google's fonts are not ours to version, and a stale
  // pose model would change what the tracker measures.
  assert.equal(route("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs"),
               "network-only");
  assert.equal(route("https://fonts.googleapis.com/css2?family=Space+Grotesk"), "network-only");
});


// ── It does not fall over on nonsense ───────────────────────────────────────

test("an unparseable URL goes to the network rather than throwing", () => {
  assert.equal(routeFor({ method: "GET", url: "not a url", mode: "cors" }, ORIGIN), "network-only");
});


// ── The cache is versioned ──────────────────────────────────────────────────

test("the cache name carries a version, so a deploy can invalidate it", () => {
  assert.match(source, /const CACHE = 'physioai-shell-v\d+'/);
});

test("old caches are deleted on activate", () => {
  // Without this a bumped version leaves the previous shell on disk, and a
  // browser that fails to update simply keeps using it.
  assert.match(source, /caches\.delete/);
});

test("every shell entry is a real file", async () => {
  const { existsSync } = await import("node:fs");
  const list = source.split("const SHELL = [")[1].split("];")[0];
  const paths = [...list.matchAll(/'\.\/([^']*)'/g)].map((m) => m[1]).filter(Boolean);

  assert.ok(paths.length > 10, "the shell should not be nearly empty");
  for (const path of paths) {
    assert.ok(
      existsSync(new URL(`../${path}`, import.meta.url)),
      `${path} is listed in the shell but does not exist — install would fail wholesale`,
    );
  }
});
