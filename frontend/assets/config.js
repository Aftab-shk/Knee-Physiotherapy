/*
 * config.js — where the frontend finds its backend.
 * =================================================
 *
 * The API base used to be a literal in upload.html, which meant a deployed copy
 * of these pages pointed at the developer's own laptop. Resolution order, first
 * match wins:
 *
 *   1. ?api=http://host:port        — per-load override, handy for testing
 *   2. window.PHYSIOAI_API_BASE     — set by a <script> before this one loads
 *   3. <meta name="physioai-api">   — per-deployment, no JS needed
 *   4. same host on port 8000       — when served from a real host
 *   5. http://127.0.0.1:8000        — local development default
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

    // Served from a real host: assume the API sits beside it on the API port.
    // file:// has no hostname, so it falls through to the loopback default.
    if (loc.hostname && !LOCAL_HOSTS.includes(loc.hostname)) {
      return `${loc.protocol}//${loc.hostname}:${DEFAULT_PORT}`;
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
