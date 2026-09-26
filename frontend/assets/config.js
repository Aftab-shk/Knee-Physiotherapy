/*
 * config.js - where the frontend finds its backend.
 * =================================================
 *
 * The API base used to be a literal in upload.html, which meant a deployed copy
 * of these pages pointed at the developer's own laptop. Resolution order, first
 * match wins:
 *
 *   1. ?api=http://host:port - per-load override, handy for testing
 *   2. window.PHYSIOAI_API_BASE - set by a <script> before this one loads
 *   3. <meta name="physioai-api"> - per-deployment, no JS needed
 *   4. this page's own origin - the API serves these pages
 *   5. http://127.0.0.1:8000 - local development default
 *
 * Load as a classic script before api.js, matching exercise-animations.js:
 *   <script src="assets/config.js"></script>
 */
(function (global) {
  'use strict';

  const LOCAL_HOSTS = ['localhost', '127.0.0.1', '[::1]', ''];
  const DEFAULT_PORT = '8000';
  const FALLBACK = 'http://127.0.0.1:' + DEFAULT_PORT;

  function trimSlash(url) {
    return String(url).replace(/\/+$/, '');
  }

  // Only http(s) is accepted. A query parameter is attacker-supplyable via a
  // crafted link, and without this check `?api=javascript:…` would be handed
  // straight to fetch().
  function safeUrl(value) {
    if (!value) return null;
    try {
      const u = new URL(value, global.location ? global.location.href : undefined);
      return (u.protocol === 'http:' || u.protocol === 'https:') ? trimSlash(u.href) : null;
    } catch {
      return null;
    }
  }

  function resolveBase() {
    const loc = global.location || {};

    const fromQuery = safeUrl(new URLSearchParams(loc.search || '').get('api'));
    if (fromQuery) return fromQuery;

    const fromGlobal = safeUrl(global.PHYSIOAI_API_BASE);
    if (fromGlobal) return fromGlobal;

    const meta = global.document && global.document.querySelector('meta[name="physioai-api"]');
    const fromMeta = safeUrl(meta && meta.content);
    if (fromMeta) return fromMeta;

    // Served over http(s) by anything: the API is the thing serving these
    // pages, so its origin is this origin.
    //
    // This used to guess port 8000 instead, which broke twice over. A page
    // served on any other port called across origins to a server that might not
    // be there, and now that the API sends a Content-Security-Policy of
    // connect-src 'self' the browser blocks the guess outright - correctly,
    // because a page that talks to another origin for its clinical data is
    // exactly what that header exists to stop.
    //
    // Only file:// falls through: it has no origin to inherit.
    if (loc.protocol === 'http:' || loc.protocol === 'https:') {
      return trimSlash(loc.origin || `${loc.protocol}//${loc.host}`);
    }

    return FALLBACK;
  }

  global.PhysioConfig = {
    API_BASE: resolveBase(),
    /**
     * True when the page was opened straight from disk. The origin is then
     * literally "null", which no CORS allow-list can match.
     */
    isFileOrigin: (global.location || {}).protocol === 'file:',
  };
})(typeof window !== 'undefined' ? window : globalThis);
