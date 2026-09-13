/*
 * sw.js — the app shell, cached so it opens without a connection.
 * ==============================================================
 *
 * What is cached, and what is emphatically not:
 *
 *   The shell — HTML, CSS, JS, icons — is cached, so the app opens on a bus.
 *
 *   **Nothing from the API is ever cached.** Not a prescription, not a
 *   progress chart, not a caseload. A stale prescription is a stale *movement
 *   ceiling*: a clinician lowers a limit, and the patient's phone keeps handing
 *   them yesterday's number with no indication anything is out of date. Every
 *   request that is not a same-origin static asset goes to the network and
 *   stays there.
 *
 * HTML is network-first: a cached page is a fallback for being offline, never
 * a reason to run an old version of the safety logic. Static assets are
 * cache-first but versioned, so a deploy invalidates them wholesale.
 */

// Bump on every deploy. Old caches are deleted on activate, which is what makes
// a stale build impossible rather than merely unlikely.
const CACHE = 'physioai-shell-v2';

const SHELL = [
  './',
  './index.html',
  './login.html',
  './upload.html',
  './tracker.html',
  './progress.html',
  './share.html',
  './assets/theme.css',
  './assets/progress-view.css',
  './assets/config.js',
  './assets/api.js',
  './assets/pose-gate.js',
  './assets/voice.js',
  './assets/form-check.js',
  './assets/progress-view.js',
  './assets/outcome-form.js',
  './exercise-animations.js',
  './logo.png',
  './manifest.webmanifest',
  './assets/icons/icon-192.png',
  './assets/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    // addAll fails the whole install if one entry 404s, which would leave the
    // worker uninstalled and the app working normally — the right failure.
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

/**
 * Decide how one request should be handled.
 *
 * Split out and exported so the policy can be tested directly — the routing
 * rules are the part that would silently serve a stale clinical value, and they
 * should not only be verifiable inside a browser.
 */
function routeFor(request, selfOrigin) {
  if (request.method !== 'GET') return 'network-only';

  let url;
  try {
    url = new URL(request.url);
  } catch {
    return 'network-only';
  }

  // Anything on another origin — the API, MediaPipe's CDN, Google Fonts — is
  // never served from cache. The API because its answers are clinical and
  // change; the CDN because a wasm model is not ours to version.
  if (url.origin !== selfOrigin) return 'network-only';

  // A same-origin API call (the backend deployed behind the same host) must not
  // be cached either. Matching on the paths the backend actually serves.
  if (/^\/(analyse-xray|exercises|sessions|share|auth|me|clinician|health)\b/.test(url.pathname)) {
    return 'network-only';
  }

  // Pages: fresh when possible, cached when not. Never the other way round —
  // an old tracker.html is old safety logic.
  if (request.mode === 'navigate' || url.pathname.endsWith('.html') || url.pathname.endsWith('/')) {
    return 'network-first';
  }

  return 'cache-first';
}

self.addEventListener('fetch', (event) => {
  const strategy = routeFor(event.request, self.location.origin);
  if (strategy === 'network-only') return;

  if (strategy === 'network-first') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(event.request, copy)).catch(() => {});
          return response;
        })
        .catch(() => caches.match(event.request).then((hit) => hit || caches.match('./index.html'))),
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((hit) => hit || fetch(event.request).then((response) => {
      if (response.ok) {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy)).catch(() => {});
      }
      return response;
    })),
  );
});

// Exported for the tests; harmless in a worker.
if (typeof self !== 'undefined') self.routeFor = routeFor;
