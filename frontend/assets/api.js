/*
 * api.js — the one place the frontend talks to the backend.
 * =========================================================
 *
 * Every page goes through here rather than calling fetch() directly, so that
 * request headers, error shaping and the signed-in token have a single home
 * instead of a copy per page.
 *
 * Load as a classic script after config.js, matching exercise-animations.js:
 *   <script src="assets/config.js"></script>
 *   <script src="assets/api.js"></script>
 *
 * Errors always arrive as an ApiError carrying `status` and a `detail` already
 * unwrapped from FastAPI's {"detail": ...} envelope, so callers never have to
 * guess whether a rejection is a network failure or a 422.
 */
(function (global) {
  'use strict';

  const cfg = global.PhysioConfig;
  if (!cfg) throw new Error('assets/api.js requires assets/config.js to be loaded first.');

  const REQUEST_TIMEOUT_MS = 60000;  // a B4 forward pass on CPU is ~300 ms; this is for a stalled socket
  const PROBE_TIMEOUT_MS = 3000;

  class ApiError extends Error {
    constructor(message, { status = 0, detail = null, cause = null } = {}) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.detail = detail;
      this.cause = cause;
    }
    /** True for the errors that mean "the request never got a reply". */
    get isNetwork() { return this.status === 0; }
  }

  // ── The signed-in session ─────────────────────────────────────────────────
  //
  // The token is kept in localStorage rather than an httpOnly cookie, because
  // the API deliberately runs with allow_credentials off (see the CORS note in
  // backend/main.py) — a bearer header is what fits that decision. The trade is
  // real: script running on this origin can read the token, so an XSS bug here
  // is an account compromise. Every value rendered from API data goes through
  // textContent, never innerHTML, for exactly that reason.
  //
  // It survives a refresh and a closed tab on purpose. Rehab is a daily habit
  // over months, and an app that asks for a password every morning is an app
  // people stop opening.
  const TOKEN_KEY = 'physioai_token';

  let authToken = null;
  try {
    authToken = global.localStorage ? global.localStorage.getItem(TOKEN_KEY) : null;
  } catch {
    // Private browsing, or storage disabled entirely. Sign-in still works for
    // the length of the page's life; it just will not be remembered.
    authToken = null;
  }

  function setAuthToken(token) {
    authToken = token || null;
    try {
      if (!global.localStorage) return;
      if (authToken) global.localStorage.setItem(TOKEN_KEY, authToken);
      else global.localStorage.removeItem(TOKEN_KEY);
    } catch { /* storage unavailable; the in-memory token still works */ }
  }

  function authHeaders() {
    return authToken ? { Authorization: `Bearer ${authToken}` } : {};
  }

  function isSignedIn() { return Boolean(authToken); }

  async function request(path, { method = 'GET', body = null, headers = {}, timeout = REQUEST_TIMEOUT_MS } = {}) {
    const url = cfg.API_BASE + path;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeout);

    let response;
    try {
      response = await fetch(url, {
        method,
        body,
        headers: { ...authHeaders(), ...headers },
        signal: ctl.signal,
      });
    } catch (err) {
      // AbortError and a genuine network failure are indistinguishable to the
      // caller in every way that matters: no reply arrived.
      throw new ApiError(
        err.name === 'AbortError'
          ? `The request to ${url} timed out.`
          : 'Failed to fetch',
        { cause: err },
      );
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      // A 500 from an unhandled exception may not be JSON at all, so a failure
      // to parse the body must not mask the status code that explains it.
      let detail = null;
      try {
        detail = (await response.json()).detail ?? null;
      } catch { /* non-JSON error body */ }
      throw new ApiError(detail || `Server error ${response.status}`, {
        status: response.status,
        detail,
      });
    }

    return response.json();
  }

  /*
   * "Failed to fetch" is what the browser reports for BOTH an unreachable
   * server and a CORS rejection, so blaming the server outright sends people off
   * to restart a backend that was never down. Narrow it down first:
   *
   *   file:// page      the origin is literally "null", which no allow-list can
   *                     match. Certain without probing anything.
   *   no-cors probe ok  the server answered, so the request WAS delivered and it
   *                     was the allow-list that rejected the reply. A
   *                     mode:'no-cors' GET skips the allow-origin check and
   *                     resolves opaquely exactly where a normal fetch throws,
   *                     which is what makes the two cases separable at all.
   *   probe fails       genuinely unreachable.
   */
  async function describeFailure(err) {
    if (!(err instanceof ApiError) || !err.isNetwork) {
      return err.message;
    }

    if (cfg.isFileOrigin) {
      return 'This page was opened straight from disk, so the browser sends "null" as its ' +
        'origin and the API rejects it. Serve the frontend over http:// instead: run ' +
        '"python -m http.server 5500 --directory frontend" and open ' +
        'http://127.0.0.1:5500/upload.html. The webcam tracker needs a real origin too.';
    }

    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), PROBE_TIMEOUT_MS);
    try {
      await fetch(`${cfg.API_BASE}/health`, { mode: 'no-cors', signal: ctl.signal });
      return `The backend at ${cfg.API_BASE} is running, but it refused this page's origin ` +
        `(${global.location.origin}). Add that origin to CORS_ORIGINS in backend/main.py, or serve ` +
        `this page from a port that is already allowed: 5500, 8080, 5173 or 3000.`;
    } catch {
      return `Could not reach the backend at ${cfg.API_BASE}. Start it from the backend/ ` +
        `directory with: uvicorn main:app --reload --port 8000`;
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Endpoints ─────────────────────────────────────────────────────────────

  function health() {
    return request('/health', { timeout: PROBE_TIMEOUT_MS });
  }

  /**
   * POST /analyse-xray — KL grade plus a full rehab prescription.
   * `weeksPostOp` is required by the API unless surgeryType is 'none'.
   */
  function analyseXray({ file, kneeSide, surgeryType, weeksPostOp }) {
    const form = new FormData();
    form.append('image', file);
    form.append('knee_side', kneeSide);
    form.append('surgery_type', surgeryType);
    // Sending an empty or null value fails validation with a confusing message;
    // omitting the field entirely is what the endpoint expects.
    if (surgeryType !== 'none' && weeksPostOp !== null && weeksPostOp !== undefined && weeksPostOp !== '') {
      form.append('weeks_post_op', parseInt(weeksPostOp, 10));
    }
    // No Content-Type header: the browser has to set the multipart boundary.
    return request('/analyse-xray', { method: 'POST', body: form });
  }

  /** GET /exercises — the protocol for a surgery type and week, without an X-ray. */
  function getExercises({ surgeryType, weeksPostOp = null, klGrade = 0 }) {
    const q = new URLSearchParams({ surgery_type: surgeryType, kl_grade: String(klGrade) });
    if (surgeryType !== 'none' && weeksPostOp !== null && weeksPostOp !== '') {
      q.set('weeks_post_op', String(parseInt(weeksPostOp, 10)));
    }
    return request(`/exercises?${q}`);
  }

  // ── Accounts ──────────────────────────────────────────────────────────────

  async function register({ email, password, displayName = null }) {
    const body = { email, password };
    if (displayName) body.display_name = displayName;
    const res = await request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' },
    });
    setAuthToken(res.access_token);
    return res;
  }

  async function login({ email, password }) {
    const res = await request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
      headers: { 'Content-Type': 'application/json' },
    });
    setAuthToken(res.access_token);
    return res;
  }

  /**
   * Forget the token.
   *
   * Local only: the server issues stateless JWTs, so a token that has already
   * been copied elsewhere stays valid until it expires. Revocation needs a
   * server-side denylist, which is not built.
   */
  function logout() { setAuthToken(null); }

  /** The signed-in patient, or null if the token is missing or no longer good. */
  async function me() {
    if (!authToken) return null;
    try {
      return await request('/auth/me');
    } catch (err) {
      // 401 means the token has expired or the account is gone. Clearing it
      // here stops every later request retrying with a token already known to
      // be dead.
      if (err instanceof ApiError && err.status === 401) {
        setAuthToken(null);
        return null;
      }
      throw err;
    }
  }

  /**
   * PATCH /me/surgery — record (or clear) the operation being recovered from.
   *
   * Stored once so weeks-post-op stops being retyped on every visit. That number
   * selects the whole rehab protocol, and a plausible wrong week is
   * indistinguishable from a right one. Pass null for either field to clear it;
   * the server derives the week on every read, so nothing goes stale overnight.
   */
  function setSurgery({ surgeryDate = null, surgeryType = null } = {}) {
    return request('/me/surgery', {
      method: 'PATCH',
      body: JSON.stringify({ surgery_date: surgeryDate, surgery_type: surgeryType }),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /** Past analyses, newest first. Requires a signed-in patient. */
  function myPrescriptions({ limit = 20 } = {}) {
    return request(`/me/prescriptions?limit=${encodeURIComponent(limit)}`);
  }

  /**
   * POST /sessions/sets — record one completed set from the tracker.
   *
   * Sent per set rather than once at the end, so a patient who stops after two
   * of three sets still has those two saved. Re-posting the same set is a
   * no-op server-side, which makes a retry safe.
   */
  function recordSet(payload) {
    return request('/sessions/sets', {
      method: 'POST',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /**
   * POST /sessions/report — how a finished session felt.
   *
   * The server merges rather than replaces, so calling this again with only an
   * exertion score will not wipe a pain score already recorded.
   */
  function recordReport({ client_session_id, pain_after = null, rpe = null }) {
    const body = { client_session_id };
    if (pain_after !== null) body.pain_after = pain_after;
    if (rpe !== null) body.rpe = rpe;
    return request('/sessions/report', {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // ── Clinicians ────────────────────────────────────────────────────────────
  //
  // A clinician's token and a patient's are not interchangeable: the server
  // stamps a role into each and refuses one where the other is expected. The
  // same setAuthToken holds whichever this browser signed in as.

  async function clinicianRegister({ email, password, displayName = null, registration = null }) {
    const res = await request('/clinician/register', {
      method: 'POST',
      body: JSON.stringify({
        email, password, display_name: displayName, registration,
      }),
      headers: { 'Content-Type': 'application/json' },
    });
    setAuthToken(res.access_token);
    return res;
  }

  async function clinicianLogin({ email, password }) {
    const res = await request('/clinician/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
      headers: { 'Content-Type': 'application/json' },
    });
    setAuthToken(res.access_token);
    return res;
  }

  /** The signed-in clinician, or null if this token is not one. */
  async function clinicianMe() {
    if (!authToken) return null;
    try {
      return await request('/clinician/me');
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) return null;
      throw err;
    }
  }

  /**
   * POST /clinician/invites — a code for a patient to redeem.
   *
   * Returned once; only its hash is stored. The patient redeeming it is the
   * consent, which is why the code goes to them rather than the other way round.
   */
  function createInvite({ patientLabel = null, days = 7 } = {}) {
    return request('/clinician/invites', {
      method: 'POST',
      body: JSON.stringify({ patient_label: patientLabel, days }),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /** GET /clinician/invites — codes issued but not yet redeemed. */
  function listInvites() {
    return request('/clinician/invites');
  }

  /** GET /clinician/patients — your caseload. */
  function getCaseload() {
    return request('/clinician/patients');
  }

  /** GET /clinician/patients/{id}/progress — one patient, in full. */
  function getPatientProgress(patientId, { days = 90, tzOffsetMinutes = -new Date().getTimezoneOffset() } = {}) {
    const q = new URLSearchParams({ days: String(days), tz_offset_minutes: String(tzOffsetMinutes) });
    return request(`/clinician/patients/${encodeURIComponent(patientId)}/progress?${q}`);
  }

  /** DELETE /clinician/patients/{linkId} — discharge from your caseload. */
  function dischargePatient(linkId) {
    return request(`/clinician/patients/${encodeURIComponent(linkId)}`, { method: 'DELETE' });
  }

  /** POST /me/clinicians/redeem — give a clinician access. The patient's consent. */
  function redeemInvite(code) {
    return request('/me/clinicians/redeem', {
      method: 'POST',
      body: JSON.stringify({ code }),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /** GET /me/clinicians — who can see your progress. */
  function myClinicians() {
    return request('/me/clinicians');
  }

  /** DELETE /me/clinicians/{linkId} — withdraw a clinician's access. */
  function withdrawClinician(linkId) {
    return request(`/me/clinicians/${encodeURIComponent(linkId)}`, { method: 'DELETE' });
  }

  /**
   * GET /me/flags — anything worth raising with your physiotherapist.
   *
   * Triage signals, not findings: they say a human should look, and decide
   * nothing on their own. Findings about the clinician's own workflow are left
   * out of this one.
   */
  function myFlags() {
    return request('/me/flags');
  }

  /** GET /clinician/patients/{id}/flags — the same assessment, with the numbers. */
  function getPatientFlags(patientId) {
    return request(`/clinician/patients/${encodeURIComponent(patientId)}/flags`);
  }

  // ── Clinical review ───────────────────────────────────────────────────────
  //
  // The model drafts; a clinician approves. Raising any limit needs a reason —
  // the server refuses without one, so collect it before submitting rather than
  // letting the request fail.

  /** GET /exercises/catalogue — everything a clinician may add. No auth needed. */
  function getCatalogue() {
    return request('/exercises/catalogue');
  }

  /** GET /clinician/patients/{id}/prescriptions — newest first. */
  function getPatientPrescriptions(patientId) {
    return request(`/clinician/patients/${encodeURIComponent(patientId)}/prescriptions`);
  }

  /** GET /clinician/prescriptions/{id} — draft and effective side by side, plus the audit trail. */
  function getPrescriptionDetail(prescriptionId) {
    return request(`/clinician/prescriptions/${encodeURIComponent(prescriptionId)}`);
  }

  /**
   * POST /clinician/prescriptions/{id}/review — approve, with or without changes.
   *
   * Exercises left out of `decisions` are kept as drafted: a clinician editing
   * one of nine has approved the other eight, not deleted them.
   */
  function reviewPrescription(prescriptionId, { decisions = [], note = null, ceiling = null, ceilingReason = null } = {}) {
    return request(`/clinician/prescriptions/${encodeURIComponent(prescriptionId)}/review`, {
      method: 'POST',
      body: JSON.stringify({ decisions, note, ceiling, ceiling_reason: ceilingReason }),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /** GET /me/prescriptions/{id} — what you should follow, and who stands behind it. */
  function getMyPrescription(prescriptionId) {
    return request(`/me/prescriptions/${encodeURIComponent(prescriptionId)}`);
  }

  // ── Sharing ───────────────────────────────────────────────────────────────

  /**
   * POST /me/share-links — create a read-only link to your progress.
   *
   * The token comes back exactly once; only its hash is stored, so it cannot be
   * recovered afterwards. Show it, then let the patient copy it.
   */
  function createShareLink({ label = null, days = 14 } = {}) {
    return request('/me/share-links', {
      method: 'POST',
      body: JSON.stringify({ label, days }),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /** GET /me/share-links — links you have shared, newest first. Never includes tokens. */
  function listShareLinks() {
    return request('/me/share-links');
  }

  /** DELETE /me/share-links/{id} — stop a link working, immediately. */
  function revokeShareLink(id) {
    return request(`/me/share-links/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  /**
   * GET /share/{token} — read a shared progress view.
   *
   * No account and no token header: the link itself is the credential, which is
   * the entire point. Every failure is a 404, deliberately — an expired link and
   * one that never existed look the same.
   */
  function getSharedProgress(token, { days = 90, tzOffsetMinutes = -new Date().getTimezoneOffset() } = {}) {
    const q = new URLSearchParams({ days: String(days), tz_offset_minutes: String(tzOffsetMinutes) });
    return request(`/share/${encodeURIComponent(token)}?${q}`);
  }

  /** GET /sessions — recent exercise sessions, newest first. */
  function mySessions({ limit = 20 } = {}) {
    return request(`/sessions?limit=${encodeURIComponent(limit)}`);
  }

  /**
   * GET /me/progress — range of motion, adherence and per-exercise history.
   *
   * `tzOffsetMinutes` is minutes east of UTC. Days are bucketed in it so an
   * evening session lands on the evening it happened, rather than the next UTC
   * day — which would split a streak the patient never broke.
   */
  function getProgress({ days = 90, tzOffsetMinutes = -new Date().getTimezoneOffset() } = {}) {
    const q = new URLSearchParams({
      days: String(days),
      tz_offset_minutes: String(tzOffsetMinutes),
    });
    return request(`/me/progress?${q}`);
  }

  global.PhysioAPI = {
    ApiError,
    get baseUrl() { return cfg.API_BASE; },
    isSignedIn,
    setAuthToken,
    register,
    login,
    logout,
    me,
    myPrescriptions,
    setSurgery,
    recordSet,
    recordReport,
    mySessions,
    getProgress,
    clinicianRegister,
    clinicianLogin,
    clinicianMe,
    createInvite,
    listInvites,
    getCaseload,
    getPatientProgress,
    dischargePatient,
    myFlags,
    getPatientFlags,
    getCatalogue,
    getPatientPrescriptions,
    getPrescriptionDetail,
    reviewPrescription,
    getMyPrescription,
    redeemInvite,
    myClinicians,
    withdrawClinician,
    createShareLink,
    listShareLinks,
    revokeShareLink,
    getSharedProgress,
    health,
    analyseXray,
    getExercises,
    describeFailure,
  };
})(typeof window !== 'undefined' ? window : globalThis);
