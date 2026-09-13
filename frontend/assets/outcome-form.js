/*
 * outcome-form.js — the seven questions, and the rules for asking them.
 * =====================================================================
 *
 * Everything else this app shows the patient is something it measured: how far
 * the knee bent, how many sets were finished, how long the safe angle was
 * exceeded. This is the one place the patient answers back — and the only place
 * the question they actually came with gets asked, which is whether the knee is
 * getting better to live with.
 *
 * The instrument is KOOS-JR. Its item wording arrives from GET /outcome-measures
 * and is never written down here. That is not a style preference: a KOOS-JR whose
 * questions have been reworded is not a KOOS-JR, it is a bespoke survey whose
 * scores only look comparable to a registry's. One copy of the wording, served
 * by the thing that scores it.
 *
 * Split the same way voice.js is:
 *
 *   pure      `unanswered`, `isComplete`, `progressText`, `payloadFor`,
 *             `deltaText` — decide what the form is doing, no browser needed.
 *             Tested directly in frontend/tests/outcome-form.test.mjs.
 *   `render`  the part that builds DOM.
 *
 * Two rules the form itself enforces:
 *
 *   * All seven or none. KOOS-JR has no published rule for a missing item, and
 *     scoring six of them produces a number indistinguishable from a real one.
 *     So submit stays disabled, and says how many are left.
 *   * Nothing here is a wall. The card can be dismissed, and dismissing it costs
 *     the patient nothing anywhere else in the app.
 */
(function (global) {
  'use strict';

  /** Indices (0-based) of items still unanswered, in order. */
  function unanswered(answers, itemCount) {
    const out = [];
    for (let i = 0; i < itemCount; i++) {
      const v = answers[i];
      if (!Number.isInteger(v)) out.push(i);
    }
    return out;
  }

  function isComplete(answers, itemCount) {
    return itemCount > 0 && unanswered(answers, itemCount).length === 0;
  }

  /**
   * What to show beside the submit button.
   *
   * Counts what is left rather than what is done. "Two to go" is an instruction;
   * "five of seven" is a status report, and the reader has to do the subtraction
   * themselves to find out whether they can submit.
   */
  function progressText(answers, itemCount) {
    const left = unanswered(answers, itemCount).length;
    if (itemCount === 0) return '';
    if (left === 0) return 'All answered';
    if (left === itemCount) return `${itemCount} questions, about a minute`;
    return `${left} left`;
  }

  /**
   * The request body, or an exception.
   *
   * Throws rather than silently padding: the server refuses a partial form too,
   * and a client that quietly filled the gaps would be inventing clinical data.
   */
  function payloadFor(answers, itemCount, kneeSide) {
    if (!isComplete(answers, itemCount)) {
      throw new Error(`All ${itemCount} questions have to be answered before this can be scored.`);
    }
    return {
      responses: answers.slice(0, itemCount).map(Number),
      knee_side: kneeSide,
    };
  }

  /**
   * A change, written for a chart tooltip or a tile.
   *
   * The sign convention is the thing to get right: the score runs 0-100 with
   * higher better, so a positive delta is an improving knee. Anything the server
   * called "unchanged" is rendered as such — a two-point move is inside the
   * instrument's own noise, and showing it as "−2" invites a conclusion the
   * questionnaire cannot support.
   */
  function deltaText(change) {
    if (!change || change.delta === null || change.delta === undefined) return '';
    if (change.direction === 'unchanged') return 'about the same';
    const rounded = Math.abs(Math.round(change.delta));
    return change.delta > 0 ? `${rounded} points better` : `${rounded} points worse`;
  }

  // ── The form ──────────────────────────────────────────────────────────────

  // textContent everywhere, as in progress-view.js. The bearer token lives in
  // localStorage on this origin, so an innerHTML path fed by API data would be
  // an account-compromise bug rather than a cosmetic one.
  //
  // Reached through `global` rather than as a bare `document`, so the module can
  // be driven by the stand-in document in frontend/tests/outcome-form.test.mjs.
  function el(tag, cls, text) {
    const node = global.document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /**
   * Build the questionnaire into `host`.
   *
   * `definition` is a GET /outcome-measures body. `onSubmit(payload)` is called
   * with a ready request body; it should return a promise, and rejecting leaves
   * the form filled in so nobody has to answer seven questions twice.
   *
   * Returns { destroy } so the caller can take it down again.
   */
  function render(host, definition, { kneeSide = 'right', onSubmit, onCancel } = {}) {
    const items = definition.items || [];
    const options = definition.response_options || [];
    const answers = new Array(items.length).fill(null);
    const formId = `koos-${Math.random().toString(36).slice(2, 8)}`;

    host.textContent = '';

    const form = el('form', 'om-form');
    form.noValidate = true;

    // The caveat, when the instrument is a poor fit for this patient's
    // operation. Above the questions, because it changes how the answers should
    // be read and that is not something to discover afterwards.
    if (definition.caveat) {
      const warn = el('div', 'notice notice-warn');
      warn.appendChild(el('span', null, 'ⓘ'));
      warn.appendChild(el('span', null, definition.caveat));
      form.appendChild(warn);
    }

    // Grouped by the question stem rather than by section: several items share
    // one lead-in ("What amount of knee pain have you had when…"), and repeating
    // it above each of them is how a seven-item form starts feeling like twenty.
    let lastLeadIn = null;
    let group = null;

    items.forEach((item, index) => {
      if (item.lead_in !== lastLeadIn) {
        lastLeadIn = item.lead_in;
        group = el('fieldset', 'om-group');
        group.appendChild(el('legend', 'om-lead', item.lead_in));
        form.appendChild(group);
      }

      const row = el('div', 'om-item');
      row.appendChild(el('div', 'om-prompt', item.prompt));

      const scale = el('div', 'om-scale');
      scale.setAttribute('role', 'radiogroup');
      scale.setAttribute('aria-label', `${item.lead_in} ${item.prompt}`);

      for (const option of options) {
        const id = `${formId}-${index}-${option.value}`;

        const input = global.document.createElement('input');
        input.type = 'radio';
        input.name = `${formId}-${index}`;
        input.id = id;
        input.value = String(option.value);
        input.className = 'om-radio';

        const label = el('label', 'om-option');
        label.setAttribute('for', id);
        label.appendChild(el('span', 'om-option-label', option.label));

        input.addEventListener('change', () => {
          answers[index] = option.value;
          row.classList.add('answered');
          refresh();
        });

        scale.append(input, label);
      }

      row.appendChild(scale);
      group.appendChild(row);
    });

    // ── Foot ────────────────────────────────────────────────────────────────
    const foot = el('div', 'om-foot');

    const status = el('div', 'om-status');
    status.setAttribute('role', 'status');       // announced as it changes
    status.setAttribute('aria-live', 'polite');

    const submit = el('button', 'btn', 'Save answers');
    submit.type = 'submit';
    submit.disabled = true;

    const actions = el('div', 'om-actions');
    if (onCancel) {
      const cancel = el('button', 'om-cancel', 'Not now');
      cancel.type = 'button';
      cancel.addEventListener('click', onCancel);
      actions.appendChild(cancel);
    }
    actions.appendChild(submit);

    const error = el('div', 'om-error hidden');
    error.setAttribute('role', 'alert');

    foot.append(status, actions);
    form.append(error, foot);

    function refresh() {
      const ready = isComplete(answers, items.length);
      submit.disabled = !ready;
      status.textContent = progressText(answers, items.length);
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!isComplete(answers, items.length) || !onSubmit) return;

      submit.disabled = true;
      error.classList.add('hidden');
      try {
        await onSubmit(payloadFor(answers, items.length, kneeSide));
      } catch (err) {
        // The answers stay on screen. Nobody should have to fill in seven
        // questions twice because a request timed out.
        error.textContent = (err && err.message) || 'That could not be saved. Try again.';
        error.classList.remove('hidden');
        submit.disabled = false;
      }
    });

    refresh();
    host.appendChild(form);

    return {
      destroy() { host.textContent = ''; },
      // Exposed for tests and for a caller that wants to know without reaching
      // into the DOM.
      get answers() { return answers.slice(); },
    };
  }

  global.OutcomeForm = {
    unanswered,
    isComplete,
    progressText,
    payloadFor,
    deltaText,
    render,
  };
})(typeof window !== 'undefined' ? window : globalThis);
