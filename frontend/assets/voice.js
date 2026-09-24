/*
 * voice.js — the tracker talking, so the patient does not have to look.
 * =====================================================================
 *
 * Someone doing a straight leg raise is lying on the floor with their head on
 * the carpet. Someone doing a mini squat is watching their own knee. Neither is
 * reading a screen, which means every count, every countdown and — most of all —
 * every "you have gone too far" arrives too late to be useful.
 *
 * Two halves, deliberately separated:
 *
 *   `cueFor…`  pure functions deciding *what should be said*, if anything.
 *              No browser, no speech, testable directly — see
 *              frontend/tests/voice.test.mjs.
 *   `speak`    the thin part that actually makes noise.
 *
 * Rules the cue logic follows, because a coach that never shuts up gets muted
 * and then says nothing at all when it matters:
 *
 *   * Long sets are not counted aloud rep by rep. Twenty numbers in a row is
 *     noise; the milestones and the last few are what a person listens for.
 *   * Nothing is repeated. Every cue is keyed, and a key already spoken in this
 *     set is skipped.
 *   * Safety interrupts. Everything else waits its turn.
 */
(function (global) {
  'use strict';

  const STORAGE_KEY = 'physioai_voice';

  // Speaking every rep is right for a set of ten and wearing for a set of
  // thirty. Above this, only the milestones and the closing reps are called.
  const COUNT_EVERY_REP_UP_TO = 12;
  const MILESTONE_EVERY = 5;
  const FINAL_REPS_ALWAYS_CALLED = 3;

  // Seconds at which a countdown is spoken. Dense at the end, sparse before.
  const COUNTDOWN_AT = [30, 20, 10, 5, 3, 2, 1];

  function supported() {
    return typeof global.speechSynthesis !== 'undefined'
      && typeof global.SpeechSynthesisUtterance !== 'undefined';
  }

  let enabled = true;
  try {
    const stored = global.localStorage && global.localStorage.getItem(STORAGE_KEY);
    if (stored !== null && stored !== undefined) enabled = stored === 'on';
  } catch {
    // Private browsing, or storage disabled. Voice still works; the preference
    // just is not remembered.
  }

  function setEnabled(value) {
    enabled = Boolean(value);
    if (!enabled) cancel();
    try {
      if (global.localStorage) global.localStorage.setItem(STORAGE_KEY, enabled ? 'on' : 'off');
    } catch { /* not remembered; not important enough to surface */ }
  }

  const isEnabled = () => enabled;

  function cancel() {
    if (supported()) {
      try { global.speechSynthesis.cancel(); } catch { /* nothing to cancel */ }
    }
  }

  /**
   * Say something.
   *
   * `interrupt` is for safety only. A rep count arriving on top of "ease off"
   * would bury the one utterance that matters, so the alarm cancels the queue
   * and everything else waits.
   */
  function speak(text, { interrupt = false, rate = 1.05 } = {}) {
    if (!enabled || !text || !supported()) return false;
    try {
      if (interrupt) global.speechSynthesis.cancel();
      const utterance = new global.SpeechSynthesisUtterance(text);
      utterance.rate = rate;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      global.speechSynthesis.speak(utterance);
      return true;
    } catch {
      // Speech is a convenience. A browser refusing it must not stop a session.
      return false;
    }
  }

  // ── What to say ───────────────────────────────────────────────────────────

  /**
   * A rep was counted.
   *
   * Returns {key, text} or null. `key` lets the caller skip anything already
   * spoken, so a jittery rep counter cannot make it stutter.
   */
  function cueForRep(current, target) {
    if (current < 1) return null;

    if (current === target) {
      return { key: `rep-${current}`, text: 'Set complete.' };
    }

    const remaining = target - current;
    const worthSaying =
      target <= COUNT_EVERY_REP_UP_TO
      || current % MILESTONE_EVERY === 0
      || remaining <= FINAL_REPS_ALWAYS_CALLED;

    if (!worthSaying) return null;

    // Near the end, the number left is more use than the number done.
    const text = remaining <= FINAL_REPS_ALWAYS_CALLED && remaining > 0
      ? `${remaining} to go`
      : String(current);
    return { key: `rep-${current}`, text };
  }

  /** A hold is running. Called with whole seconds remaining. */
  function cueForHold(secondsLeft, totalSeconds) {
    const left = Math.ceil(secondsLeft);
    if (left <= 0) return { key: 'hold-done', text: 'Hold complete. Relax.' };
    if (left === totalSeconds) return { key: `hold-${left}`, text: 'Hold it.' };
    if (!COUNTDOWN_AT.includes(left)) return null;
    return { key: `hold-${left}`, text: String(left) };
  }

  /** Resting between sets. */
  function cueForRest(secondsLeft, setsRemaining) {
    const left = Math.ceil(secondsLeft);
    if (left <= 0) {
      return { key: 'rest-go', text: setsRemaining > 0 ? 'Next set. Begin when ready.' : 'Begin.' };
    }
    if (!COUNTDOWN_AT.includes(left)) return null;
    // Only the last three are counted bare; earlier ones say what they are, so
    // a number heard from the next room is not mistaken for a rep count.
    return { key: `rest-${left}`, text: left <= 3 ? String(left) : `${left} seconds` };
  }

  /** A set finished. Said once, before the rest countdown starts. */
  function cueForSetComplete(setNumber, totalSets, restSeconds) {
    if (setNumber >= totalSets) {
      return { key: 'all-done', text: 'That is the last set. Well done.' };
    }
    return {
      key: `set-${setNumber}`,
      text: `Set ${setNumber} of ${totalSets} done. Rest for ${restSeconds} seconds.`,
    };
  }

  /**
   * The knee has gone past its safe angle.
   *
   * The reason this module exists: an alarm someone has to look up to
   * understand is an alarm that has already failed.
   */
  function cueForBreach(limitDegrees) {
    return { key: 'breach', text: `Ease off. Past your ${limitDegrees} degree limit.`, interrupt: true };
  }

  function cueForBreachCleared() {
    return { key: 'breach-clear', text: 'Good. Carry on.' };
  }

  global.PhysioVoice = {
    supported,
    isEnabled,
    setEnabled,
    speak,
    cancel,
    cueForRep,
    cueForHold,
    cueForRest,
    cueForSetComplete,
    cueForBreach,
    cueForBreachCleared,
    COUNT_EVERY_REP_UP_TO,
    COUNTDOWN_AT,
  };
})(typeof window !== 'undefined' ? window : globalThis);
