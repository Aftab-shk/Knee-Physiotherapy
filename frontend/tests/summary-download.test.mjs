/*
 * Tests for the appointment-summary download — the PDF half of assets/api.js.
 *
 * This is the one endpoint the client cannot reach with a plain link. It needs
 * the bearer token, so the file arrives as a blob and the page has to build the
 * download itself, and the two things that then go wrong are silent: a request
 * that forgot the Authorization header comes back 401 and looks like a session
 * problem, and a mangled filename saves as "download" with no error anywhere.
 *
 * So the assertions are about the request that goes out and the name that comes
 * back, tested through the public surface rather than by reaching inside for the
 * header parser.
 *
 * Run:  node --test        (from frontend/)
 */

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

const SOURCE = readFileSync(new URL("../assets/api.js", import.meta.url), "utf8");

/*
 * A fresh api.js with fetch, storage and the DOM bits it touches all stubbed.
 *
 * The browser globals are passed as *parameters* rather than set on the sandbox
 * object. api.js calls `fetch(...)` and `document.createElement(...)` bare, and
 * a bare identifier resolves past the sandbox to Node's own global — so a stub
 * hung on the sandbox is simply never consulted, and the test quietly makes a
 * real network call instead. Shadowing them as arguments is what actually binds.
 */
function load({ response = null, throws = null } = {}) {
  const calls = [];
  const clicked = [];
  const revoked = [];
  const store = new Map();

  const sandbox = {
    PhysioConfig: { API_BASE: "http://api.test", isFileOrigin: false },
    location: { origin: "http://page.test" },
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, v),
      removeItem: (k) => store.delete(k),
    },
    setTimeout: (fn) => { fn(); return 0; },
    clearTimeout: () => {},
    AbortController: class { constructor() { this.signal = {}; } abort() {} },
    URL: {
      createObjectURL: () => "blob:stub",
      revokeObjectURL: (u) => revoked.push(u),
    },
    document: {
      body: { appendChild() {} },
      createElement: () => {
        const el = { href: "", download: "", click() { clicked.push({ ...el }); }, remove() {} };
        return el;
      },
    },
    fetch: async (url, init) => {
      calls.push({ url, init });
      if (throws) throw throws;
      return response;
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;

  const params = ["window", "globalThis", "fetch", "document", "URL",
                  "AbortController", "setTimeout", "clearTimeout"];
  new Function(...params, "var window=arguments[0];" + SOURCE)(
    sandbox, sandbox, sandbox.fetch, sandbox.document, sandbox.URL,
    sandbox.AbortController, sandbox.setTimeout, sandbox.clearTimeout,
  );
  return { api: sandbox.PhysioAPI, calls, clicked, revoked };
}

function pdfResponse(disposition, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    blob: async () => ({ type: "application/pdf", size: 1234 }),
    headers: { get: (name) => (name.toLowerCase() === "content-disposition" ? disposition : null) },
  };
}

const ASCII_ONLY = 'attachment; filename="physio-summary-ada-lovelace-2026-09-04.pdf"';
const BOTH_FORMS = ASCII_ONLY +
  "; filename*=UTF-8''physio-summary-Ren%C3%A9e%20Faur%C3%A9-2026-09-04.pdf";

// ── The request that goes out ───────────────────────────────────────────────

test("the download carries the bearer token", async () => {
  const { api, calls } = load({ response: pdfResponse(ASCII_ONLY) });
  api.setAuthToken("token-123");

  await api.getSummaryPdf();

  assert.equal(calls[0].init.headers.Authorization, "Bearer token-123");
});

test("the selected range and timezone reach the server", async () => {
  const { api, calls } = load({ response: pdfResponse(ASCII_ONLY) });

  await api.getSummaryPdf({ days: 30, tzOffsetMinutes: 330 });

  const url = new URL(calls[0].url);
  assert.equal(url.pathname, "/me/summary.pdf");
  assert.equal(url.searchParams.get("days"), "30");
  assert.equal(url.searchParams.get("tz_offset_minutes"), "330");
});

test("a range is always sent, so the sheet cannot silently mean something else", async () => {
  const { api, calls } = load({ response: pdfResponse(ASCII_ONLY) });

  await api.getSummaryPdf();

  assert.equal(new URL(calls[0].url).searchParams.get("days"), "90");
});

// ── The filename that comes back ────────────────────────────────────────────

test("the server's filename is used rather than one invented here", async () => {
  const { api } = load({ response: pdfResponse(ASCII_ONLY) });

  const { filename } = await api.getSummaryPdf();

  assert.equal(filename, "physio-summary-ada-lovelace-2026-09-04.pdf");
});

test("the UTF-8 filename wins, because it is the one that can carry a name", async () => {
  const { api } = load({ response: pdfResponse(BOTH_FORMS) });

  const { filename } = await api.getSummaryPdf();

  assert.equal(filename, "physio-summary-Renée Fauré-2026-09-04.pdf");
});

test("a malformed encoding falls back to the ASCII name instead of throwing", async () => {
  const broken = ASCII_ONLY + "; filename*=UTF-8''physio-%E0%A4.pdf";
  const { api } = load({ response: pdfResponse(broken) });

  const { filename } = await api.getSummaryPdf();

  assert.equal(filename, "physio-summary-ada-lovelace-2026-09-04.pdf");
});

test("a missing header still yields a usable name", async () => {
  const { api } = load({ response: pdfResponse(null) });

  const { filename } = await api.getSummaryPdf();

  assert.equal(filename, "physio-summary.pdf");
});

// ── Handing it to the browser ───────────────────────────────────────────────

test("the download is triggered under the server's filename", async () => {
  const { api, clicked } = load({ response: pdfResponse(ASCII_ONLY) });

  const name = await api.downloadSummary();

  assert.equal(clicked.length, 1);
  assert.equal(clicked[0].download, "physio-summary-ada-lovelace-2026-09-04.pdf");
  assert.equal(clicked[0].href, "blob:stub");
  assert.equal(name, "physio-summary-ada-lovelace-2026-09-04.pdf");
});

test("the object URL is released, so repeated downloads do not leak", async () => {
  const { api, revoked } = load({ response: pdfResponse(ASCII_ONLY) });

  await api.downloadSummary();

  assert.deepEqual(revoked, ["blob:stub"]);
});

// ── Failure ─────────────────────────────────────────────────────────────────

test("an expired session surfaces as a 401 the page can redirect on", async () => {
  const { api } = load({ response: pdfResponse(null, { ok: false, status: 401 }) });

  await assert.rejects(() => api.getSummaryPdf(), (err) => {
    assert.equal(err.name, "ApiError");
    assert.equal(err.status, 401);
    return true;
  });
});

test("a network failure is an ApiError, not a raw TypeError", async () => {
  const { api } = load({ throws: new TypeError("Failed to fetch") });

  await assert.rejects(() => api.getSummaryPdf(), (err) => {
    assert.equal(err.name, "ApiError");
    assert.ok(err.isNetwork, "a request that never got a reply must read as a network error");
    return true;
  });
});

test("nothing is downloaded when the request fails", async () => {
  const { api, clicked } = load({ response: pdfResponse(null, { ok: false, status: 500 }) });

  await assert.rejects(() => api.downloadSummary());

  assert.equal(clicked.length, 0, "a failed fetch must not save an empty file");
});
