# Brief: the clinician UI

**Paste this whole file into a new chat to start the work.** It is written for
someone — or something — with no memory of how the rest of the app was built.

---

## The task

Build the missing screens for clinician review. The backend is finished, tested
and documented; **write no new endpoints unless you find one genuinely missing**,
and say so explicitly if you do.

## Why this is worth doing

The project's central claim is *an AI drafts and a clinician approves.* Right now
nothing can move a prescription off `draft`, because there is no screen for it —
so the claim is false in practice.

It shows up on the appointment PDF, which prints on **every sheet ever
generated**:

> This plan has not been reviewed by a clinician — the limits above are the
> model's own draft.

That note is correct. Closing this gap is what makes it stop being correct.

**17 of 42 endpoints (40%) have no page behind them**, and the care link is
unreachable from *both* ends — `redeemInvite`, `myClinicians` and
`withdrawClinician` are patient-facing and also have no screen. A patient cannot
connect to a physiotherapist at all through the UI today.

What *does* already work: a patient can send a share link, and the physio opens
`share.html` with no account and sees live progress plus a PDF download. The
**read** path exists. The **write** path — approve, adjust a ceiling, discharge —
does not.

## Scope, smallest first

Build in this order and stop wherever you run out of appetite; each step is
useful on its own.

1. **Invite redemption for the patient** (~30 lines). A code field on
   `progress.html`, plus the list of linked clinicians and a way to withdraw.
   Without this nothing else can be reached, because the clinician has no
   patients. `redeemInvite(code)`, `myClinicians()`, `withdrawClinician(linkId)`.
2. **`clinician.html` — sign-in and caseload.** One page. Register/login, then
   the patient list sorted worst-flag-first, and invite-code creation.
3. **The review screen.** Per exercise: keep / remove / adjust, plus add from the
   catalogue, plus the overall ceiling. This is the one that matters.
4. *(optional)* Patient detail — progress charts, flags, outcome scores, PDF
   download for one patient.

**The caseload and the review screen are the same screen** in the sense that
neither is useful alone — build 2 and 3 together if you build either.

---

## The API you are building against

Every one of these already exists in `frontend/assets/api.js` as a `PhysioAPI`
method. **Do not call `fetch` directly.**

```js
// Accounts
PhysioAPI.clinicianRegister({ email, password, displayName, registration })
PhysioAPI.clinicianLogin({ email, password })
PhysioAPI.clinicianMe()

// Linking — the clinician issues, the patient redeems
PhysioAPI.createInvite({ patientLabel, days })   // -> { code, invite_hint, expires_at, ... }
PhysioAPI.listInvites()
PhysioAPI.redeemInvite(code)                     // patient side
PhysioAPI.myClinicians()                         // patient side
PhysioAPI.withdrawClinician(linkId)              // patient side
PhysioAPI.dischargePatient(linkId)               // clinician side

// Looking
PhysioAPI.getCaseload()                          // -> { count, patients: [CaseloadEntry] }
PhysioAPI.getPatientProgress(patientId, { days, tzOffsetMinutes })
PhysioAPI.getPatientFlags(patientId)
PhysioAPI.getPatientOutcomeScores(patientId)
PhysioAPI.getPatientPrescriptions(patientId)
PhysioAPI.getPrescriptionDetail(prescriptionId)  // -> PrescriptionDetail, below
PhysioAPI.getCatalogue()                         // exercises available to add

// Acting
PhysioAPI.reviewPrescription(prescriptionId, { decisions, note, ceiling, ceilingReason })
```

### `CaseloadEntry` — what a caseload row already gives you

Designed to be triaged by, so the row itself is the screen; the detail is one
click away.

```
link_id, patient_id, patient_name, patient_label,
surgery_type, weeks_post_op, linked_at,
last_session_at, sessions_last_7_days,
latest_flexion_deg, latest_pain_after, breaches_last_7_days,
latest_outcome_score, outcome_recorded_at, outcome_change,
flags: [{ code, severity, summary, evidence }],   // worst first
worst_flag: "urgent" | "warning" | "info" | null
```

Sort by `worst_flag`. `outcome_change` is negative when the patient's own verdict
has got worse — that is the single best reason to open a record, and it is
invisible in adherence.

### `PrescriptionDetail` — what the review screen renders

```
id, patient_id, patient_name, created_at,
status: "draft" | "clinician_approved" | "clinician_modified",
reviewed_at, reviewed_by, review_note,
kl_grade, max_angle, model_max_angle, model_version, demo_mode,
confidence_band, calibrated,
draft:     {...},   // what the model proposed
effective: {...},   // what the patient actually follows
audit:     [{ exercise_name, field, previous_value, new_value,
              actor, clinician_name, reason, created_at }]
```

Show `draft` beside `effective` and `max_angle` beside `model_max_angle` — the
whole point of the screen is the difference between what the model said and what
a human decided.

### The review payload

```js
decisions: [{
  name,                                    // exercise name, must match the draft or catalogue
  action: "keep" | "remove" | "adjust" | "add",
  angle_limit, target_reps, target_sets, hold_seconds,   // adjust/add only
  reason,                                  // see the rule below
}]
```

---

## The one rule you must not get wrong

**Loosening a restriction requires a written reason. Tightening does not.**

The API enforces this and returns 422 without one — `require_reason()` in
`backend/main.py` (~line 2221). It fires when:

- the overall `ceiling` is raised above the model's, → needs `ceilingReason`
- an exercise's `angle_limit` is raised, → needs that decision's `reason`
- an exercise is *added* above the current ceiling, → needs `reason`

**The UI must ask for the reason before sending, not surface a 422 afterwards.**
A clinician who has just typed out a full review and gets a validation error has
been failed by the screen.

This is the local instance of a rule that runs through the whole codebase:
*never loosen a restriction on a weak signal.* Suspected hardware warns but never
raises a limit; unverified camera views are excluded from every angle but still
count for adherence; the bilateral summary carries the more restrictive knee. If
you find yourself making something less restrictive by default, stop.

---

## How the frontend works here

**No framework, no build step, no bundler.** Plain HTML, classic scripts, one
stylesheet. Match what is there rather than introducing anything.

- **Scripts are classic, not modules**, and attach to a global:
  `PhysioAPI`, `PhysioConfig`, `PhysioVoice`, `ProgressView`, `PhysioAnimations`.
  The exceptions are `pose-gate.js` and `form-check.js`, which are ES modules
  because `tracker.html` is `type="module"`.
- **Load order matters**: `config.js` before `api.js`. `api.js` throws without it.
- `assets/theme.css` holds the design tokens; `assets/progress-view.css` holds
  the progress/share styles. A page-specific block goes in a `<style>` at the top
  of the page, as `login.html` does.
- **`ProgressView.mount(el)` / `render(data)`** already draws every chart from a
  `ProgressResponse`. `share.html` and `progress.html` both use it — the patient
  detail view should too rather than drawing anything new.
- The auth token lives in `localStorage` under `physioai_token`, handled entirely
  inside `api.js`. **A clinician token and a patient token share that one slot**,
  so signing into `clinician.html` signs the patient out in the same browser.
  Decide deliberately what to do about that and write down the decision.
- Every value rendered from API data goes through `textContent`, never
  `innerHTML`. The token is readable by script on this origin, so an XSS bug here
  is an account compromise — this rule is load-bearing, not style.

### Easy to forget

- **Add any new page to `SHELL` in `frontend/sw.js`**, or it will not work
  offline and the offline test will not cover it.
- New same-origin API paths must stay `network-only` in `sw.js`'s `routeFor()` —
  `/clinician` is already matched, so this is only a concern if you add a path
  outside the existing prefixes.
- `.range-btn` is a shared small-button style, not only a range control. Scope
  any `querySelectorAll` for range buttons to `.ranges .range-btn` — an unscoped
  selector caused a real bug.

---

## How work is verified here

Run all of these before reporting done:

```bash
python -m ruff check backend/                 # must be clean
python -m pytest backend/tests -q             # 656 passing as of 2026-09-04
cd frontend && node --test                    # 117 passing
```

Then an **end-to-end check against a live server with the real checkpoint** —
this has caught real bugs the unit tests missed every single time:

```bash
cd backend
DATABASE_URL="sqlite:///$(pwd)/e2e.db" JWT_SECRET="e2e-only" \
  python -m uvicorn main:app --port 8137 --log-level warning
```

Register a clinician, issue an invite, redeem it as a patient, review a
prescription, and confirm the summary PDF stops saying "has not been reviewed".
Use a unique email per run — the rate limiter will refuse repeated runs, and
restarting the server clears it. Delete `e2e.db` and stop the server afterwards.

**Frontend tests load a classic script into a sandbox** — see
`frontend/tests/summary-download.test.mjs` for the pattern. Note the trap
documented there: `api.js` calls `fetch(...)` and `document...` as bare
identifiers, so a stub hung on the sandbox object is never consulted and the test
makes a real network call instead. Shadow browser globals as **parameters** of
`new Function`, not as properties.

Also update:
- The **status table** and **file structure** in `README.md`
- The **test counts** in the README status table

---

## Repo orientation

```
backend/
  main.py              all 42 endpoints; clinician section ~1470-2360
  auth.py              current_patient / current_clinician split by role claim
  schemas.py           ReviewRequest, ExerciseDecision, CaseloadEntry, PrescriptionDetail
  triage.py            the flag rules — torch-free, DB-free, 33 tests
  clinical_logic.py    build_prescription(), merge_bilateral()
  summary_pdf.py       the appointment sheet
  tests/               test_clinicians.py (25), test_review.py (24), test_triage.py (33)
frontend/
  assets/api.js        every endpoint, already written
  assets/theme.css     design tokens
  assets/progress-view.js  charts, shared by progress.html and share.html
  sw.js                offline shell — add new pages to SHELL
docs/
  NEXT-clinician-ui.md this file
```

**Read `backend/tests/test_review.py` before designing the review screen.** It
documents, in prose, every rule the endpoint enforces and why — including several
that are not obvious from the schema.

## Working style that suits this repo

- Read the existing code before designing. Several previous items turned out to
  rest on a false premise, and checking first is what caught them.
- Keep pure logic separate from I/O so it can be unit tested — the pattern behind
  `pose-gate.js`, `voice.js`, `form-check.js`, `triage.py`, `outcome_measures.py`.
- Write tests that pin *why* the code is that way, not just that it runs.
- Report what you got wrong, scoped down, or could not build — plainly, not
  buried at the end. Do not quietly narrow scope.

---

*Written 2026-09-04, after the 20-item roadmap was completed. Facts above were
verified against the working tree that day; re-check anything load-bearing before
relying on it.*
