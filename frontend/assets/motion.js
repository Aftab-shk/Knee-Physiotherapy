/*
 * motion.js — scroll reveal and card transitions, applied product-wide.
 * ====================================================================
 *
 * Load from <head> WITHOUT defer:
 *   <script src="assets/motion.js"></script>
 *
 * The no-defer placement is deliberate. This script's first act is to set
 * .js-motion on <html>, and that has to happen before the first paint —
 * deferred, the page would paint every section visible and then blank them
 * as the observer took over, which is a worse flash than no animation.
 * Nothing here touches the body at parse time, so running early costs a
 * class name.
 *
 * The gate cuts the other way too: assets/theme.css hides .reveal only
 * under .js-motion. If this file 404s, throws, or is blocked, the class is
 * never set and every section renders plainly. Content is never invisible
 * by default — which matters on a page a patient may be reading to decide
 * whether to seek care.
 *
 * Why a MutationObserver and not a single querySelectorAll: the cards on
 * progress, share and upload do not exist at DOMContentLoaded. They are
 * built from template strings once the API answers, so a one-shot scan
 * would animate the landing page and quietly do nothing on the three
 * screens that matter most. New nodes are picked up as they land.
 *
 * Not loaded by tracker.html (a live camera view, nothing to scroll, and
 * its cards update every frame) or login.html (one centred form, above the
 * fold, with its own slideUp entrance).
 */
(function () {
  'use strict';

  var root = document.documentElement;

  // Asked for reduced motion? Then never set the gate. Sections stay
  // visible and unanimated, which is the request itself — not a degraded
  // version of the animation.
  var reduced = window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // IntersectionObserver is what brings a revealed element back. Without
  // it there is nothing to un-hide with, so leave the gate off.
  if (reduced || !('IntersectionObserver' in window)) return;

  root.classList.add('js-motion');

  /* Sections and cards worth revealing, across every page that has them.
     Unknown selectors match nothing, so one list serves all pages. */
  var REVEAL = [
    '.reveal',                                         /* hand-tagged     */
    '.step', '.feature-card', '.safety-callout-inner', /* index           */
    '.card',                                           /* progress, share */
    '.summary-card', '.exercise-card',                 /* upload          */
    '.rationale-box', '.caution-box', '.disclaimer-box'
  ].join(',');

  /* Cards that should lift under the cursor. Deliberately much narrower
     than REVEAL, because a lift is an affordance and not a decoration: it
     promises the card does something. The stat tiles on upload are read,
     not pressed, and its exercise rows are accordions that keep their own
     flatter hover — lifting either would promise a click that is not
     there. */
  var LIFT = ['.feature-card', '.step'].join(',');

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var el = entry.target;

      /* Stagger by position among revealing siblings, so a grid of cards
         arrives as a sequence rather than one block. Capped at 6 so a long
         list never leaves the reader waiting on the last item. */
      var i = 0;
      if (el.parentElement) {
        var sibs = el.parentElement.children;
        for (var n = 0; n < sibs.length; n++) {
          if (sibs[n] === el) break;
          if (sibs[n].classList && sibs[n].classList.contains('reveal')) i++;
        }
      }
      el.style.transitionDelay = (Math.min(i, 6) * 50) + 'ms';

      el.classList.add('visible');
      observer.unobserve(el);   /* reveal once; re-fading on scroll-up nauseates */
    });
  }, {
    /* Fire slightly BEFORE the element reaches the fold, so it is already
       fading as it scrolls up rather than starting once the reader is
       looking at it.

       The sign matters more than the feel. A negative bottom margin shrinks
       the root, which leaves a dead band at the very end of the document:
       once the page is scrolled to its maximum, anything inside that band
       can never intersect, and a card that was hidden on tag would stay
       hidden for good. A positive margin only ever enlarges the root, so
       every element that is on screen intersects — the reveal cannot strand
       content. On a page a patient reads to judge their own recovery, an
       invisible card is not a cosmetic bug. */
    threshold: 0,
    rootMargin: '0px 0px 10% 0px'
  });

  /* Marked on the element itself rather than tracked in a Set, so a node
     that is removed and re-inserted is not re-hidden under the reader. */
  function tag(el) {
    if (el.dataset.motion) return;
    el.dataset.motion = '1';
    el.classList.add('reveal');
    observer.observe(el);
  }

  function scan(node) {
    if (node.nodeType !== 1) return;
    if (node.matches(REVEAL)) tag(node);
    var found = node.querySelectorAll(REVEAL);
    for (var i = 0; i < found.length; i++) tag(found[i]);

    if (node.matches(LIFT)) node.classList.add('card-lift');
    var lifts = node.querySelectorAll(LIFT);
    for (var j = 0; j < lifts.length; j++) lifts[j].classList.add('card-lift');
  }

  function start() {
    scan(document.body);

    /* Cards injected once the API answers get the same treatment. subtree
       is required: progress-view.js replaces the contents of a wrapper, so
       the cards arrive as descendants of an added node, not as the node. */
    new MutationObserver(function (records) {
      records.forEach(function (r) {
        for (var i = 0; i < r.addedNodes.length; i++) scan(r.addedNodes[i]);
      });
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
