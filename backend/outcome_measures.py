"""
outcome_measures.py — what the patient says, on a scale someone else can read.

Everything else this app records is something it measured: how far the knee bent,
how many sets were finished, how long the safe angle was exceeded. None of that
answers the question the patient is actually asking, which is whether their knee
is getting better. A knee that flexes to 120° and hurts on every stair is not a
success, and no amount of goniometry says so.

A patient-reported outcome measure does. It is a questionnaire, scored the same
way everywhere, so that "58" here means what "58" means in a hospital audit or a
national registry — which is the entire reason to use a standard instrument
instead of inventing a five-star rating.

The instrument
--------------
**KOOS-JR**, the Knee injury and Osteoarthritis Outcome Score for Joint
Replacement. Lyman S, Lee YY, Franklin PD, Li W, Cross MB, Padgett DE.
"Validation of the KOOS, JR: A Short-form Knee Arthroplasty Outcomes Survey."
*Clin Orthop Relat Res* 474(6):1461-1471, 2016.

Chosen over the Oxford Knee Score for four reasons:

  * It is free to use. The OKS is © Oxford University Innovation and requires a
    licence for anything beyond non-commercial research, so its twelve items
    cannot simply be reproduced in this file. See OXFORD_KNEE_SCORE_NOTE.
  * Seven items rather than twelve. This is a questionnaire meant to be answered
    again and again across a year of rehab, and length is what stops that
    happening.
  * The raw score is Rasch-calibrated onto an interval 0-100 scale. Ten points is
    the same amount of knee at the top of the scale as at the bottom, which is
    what makes a *change* meaningful. The OKS's 0-48 is a sum of ordinal items
    and has no such property — the difference between 20 and 25 there is not the
    same quantity as the difference between 40 and 45.
  * It is the PROM collected by the American Joint Replacement Registry and used
    by CMS for the knee-arthroplasty bundle, so a score recorded here lines up
    with the national comparison a surgeon already sees.

Direction of the scales, because they run opposite ways and mixing them up
inverts every conclusion in the file:

    each item      0 = no symptom  …  4 = extreme      (higher is worse)
    raw sum        0 - 28                              (higher is worse)
    interval score 0 - 100                             (higher is BETTER)

100 is a knee with no symptoms at all; 0 is total knee disability.

What this module is not
-----------------------
It is not a diagnosis, and a score does not change anybody's exercise ceiling.
The ceiling comes from the radiograph and the surgical protocol, and a
questionnaire — however standard — is not evidence about what a joint can safely
withstand. What a falling score does is tell a human to look, which is why the
thresholds below feed triage.py and nothing else.

Kept free of torch, of FastAPI and of any database import, so the scoring can be
tested directly on plain data — the same shape as triage.py and clinical_logic.py.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Response options
# ---------------------------------------------------------------------------
# All seven KOOS-JR items share one 5-point Likert scale. That is a property of
# the items chosen, not a simplification made here: S1 asks about severity of
# stiffness, P2/P3/P5/P6 about the amount of pain during an activity, and A4/A6
# about difficulty — three different questions that happen to take the same
# None → Extreme answers.

RESPONSE_OPTIONS: tuple[dict, ...] = (
    {"value": 0, "label": "None"},
    {"value": 1, "label": "Mild"},
    {"value": 2, "label": "Moderate"},
    {"value": 3, "label": "Severe"},
    {"value": 4, "label": "Extreme"},
)

MIN_RESPONSE = 0
MAX_RESPONSE = 4

# KOOS asks about the last week, every time. The recall window is part of the
# instrument: answering "how has it been since the operation" produces a number
# that is not comparable with anyone else's.
RECALL_PERIOD = "the last week"


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------
# `code` is the item's identifier in the full KOOS, kept so a score recorded here
# can be traced back to the parent instrument. Order is fixed: it is the order
# the raw sum is defined over, and reordering the list would not change the sum
# but would silently change what a stored `responses` array means.

@dataclass(frozen=True)
class Item:
    code:     str
    section:  str
    prompt:   str
    # Shown above the item, because "None … Extreme" is ambiguous on its own —
    # the reader needs to know whether they are rating pain, stiffness or
    # difficulty.
    lead_in:  str


KOOS_JR_ITEMS: tuple[Item, ...] = (
    Item(
        code    = "S1",
        section = "Stiffness",
        lead_in = f"How severe has your knee stiffness been during {RECALL_PERIOD}?",
        prompt  = "After first waking in the morning",
    ),
    Item(
        code    = "P2",
        section = "Pain",
        lead_in = f"What amount of knee pain have you had during {RECALL_PERIOD} when…",
        prompt  = "Twisting or pivoting on your knee",
    ),
    Item(
        code    = "P3",
        section = "Pain",
        lead_in = f"What amount of knee pain have you had during {RECALL_PERIOD} when…",
        prompt  = "Straightening your knee fully",
    ),
    Item(
        code    = "P5",
        section = "Pain",
        lead_in = f"What amount of knee pain have you had during {RECALL_PERIOD} when…",
        prompt  = "Going up or down stairs",
    ),
    Item(
        code    = "P6",
        section = "Pain",
        lead_in = f"What amount of knee pain have you had during {RECALL_PERIOD} when…",
        prompt  = "Standing upright",
    ),
    Item(
        code    = "A4",
        section = "Function",
        lead_in = f"What difficulty have you had during {RECALL_PERIOD} when…",
        prompt  = "Rising from sitting",
    ),
    Item(
        code    = "A6",
        section = "Function",
        lead_in = f"What difficulty have you had during {RECALL_PERIOD} when…",
        prompt  = "Bending to the floor or picking up an object",
    ),
)

ITEM_COUNT = len(KOOS_JR_ITEMS)
MAX_RAW_SUM = ITEM_COUNT * MAX_RESPONSE       # 28


# ---------------------------------------------------------------------------
# Raw sum → interval score
# ---------------------------------------------------------------------------
# The Rasch calibration published with the instrument. This table is the reason
# KOOS-JR is worth using at all: without it the score is an ordinal sum, and the
# difference between 4 and 6 is not the same quantity of knee as the difference
# between 24 and 26.
#
# It is a lookup, not a formula — the spacing is irregular by construction, which
# is the point. Do not "simplify" it into a linear map; that would silently turn
# every reported change into a different number from the one a registry would
# compute for the same answers.
#
# Reproduced from Lyman et al. 2016, Table 4. A deployment that intends to submit
# these scores anywhere should check them against the published table once — a
# transcription error here is invisible, because every value it produces still
# looks like a plausible score.
_KOOS_JR_INTERVAL: tuple[float, ...] = (
    100.000,   # 0
     91.975,   # 1
     84.775,   # 2
     79.414,   # 3
     75.010,   # 4
     71.203,   # 5
     67.833,   # 6
     64.792,   # 7
     62.000,   # 8
     59.394,   # 9
     56.920,   # 10
     54.525,   # 11
     52.161,   # 12
     49.786,   # 13
     47.365,   # 14
     44.876,   # 15
     42.310,   # 16
     39.674,   # 17
     36.984,   # 18
     34.259,   # 19
     31.517,   # 20
     28.762,   # 21
     25.984,   # 22
     23.147,   # 23
     20.184,   # 24
     16.947,   # 25
     13.202,   # 26
      8.291,   # 27
      0.000,   # 28
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Like everything in triage.py, these are judgements about when a change is worth
# a human's attention — not diagnostic cut-offs. Published estimates for KOOS-JR
# vary with the population and the anchor question used, so both numbers below
# sit at the cautious end of the range they are drawn from.

# Minimal clinically important difference: the improvement a patient actually
# notices. Reported around 14 points after knee arthroplasty. Used to say
# "this is real progress", never to say the opposite.
MCID = 14.0

# Minimal detectable change: roughly the width of the instrument's own
# measurement noise. A fall smaller than this is not evidence of anything.
MDC = 10.0

# How the number is described in words. Bands are a reading aid, not a
# classification — a patient at 61 and a patient at 59 do not have different
# knees, and nothing downstream branches on which band they fall in.
_BANDS: tuple[tuple[float, str, str], ...] = (
    (85.0, "very good",  "Little or no knee trouble in daily life."),
    (70.0, "good",       "Some knee trouble, but it is not getting in the way much."),
    (50.0, "fair",       "Knee symptoms are a regular part of your day."),
    (25.0, "poor",       "Knee symptoms are limiting a lot of what you do."),
    (0.0,  "very poor",  "Severe knee symptoms affecting most daily activities."),
)

# How often it is worth asking again.
#
# The honest answer is that KOOS-JR is a registry instrument: it is collected
# before surgery and at fixed milestones afterwards — commonly 6 weeks, 3 months,
# 6 months and a year. A rehab app cannot reliably know where in that schedule
# someone is, so a steady cadence stands in for it.
#
# The floor matters more than the interval. Asked weekly, the score becomes a
# mood reading: real week-to-week variation is smaller than MDC, so most of what
# gets charted is noise, and a patient who answers the same seven questions every
# Monday stops reading them by the third time.
MIN_INTERVAL_DAYS = 14
DUE_AFTER_DAYS = 28


# ---------------------------------------------------------------------------
# The instrument, as served
# ---------------------------------------------------------------------------

KOOS_JR = "koos_jr"

INSTRUMENTS: tuple[str, ...] = (KOOS_JR,)

# How the instrument should appear to a reader — on a chart axis, on a printed
# summary, anywhere the storage key would otherwise leak out as "koos_jr".
DISPLAY_NAMES: dict[str, str] = {KOOS_JR: "KOOS-JR"}


def display_name(instrument: str) -> str:
    return DISPLAY_NAMES.get(instrument, instrument.replace("_", "-").upper())

# KOOS-JR was developed and validated in knee osteoarthritis and knee
# arthroplasty populations. It is not the right instrument for a 22-year-old
# eight weeks after an ACL reconstruction — the full KOOS or the IKDC subjective
# form is, because KOOS-JR contains no item about sport, pivoting under load or
# confidence in the knee, which is most of what has actually gone wrong there.
#
# Surfaced rather than enforced. Refusing to record a score would leave the
# patient with nothing, and a score with a caveat attached beats no score at all
# — as long as the caveat travels with it.
_LIMITED_VALIDATION = ("acl", "meniscus", "arthroscopy")

_APPLICABILITY_CAVEAT = (
    "KOOS-JR was developed for knee osteoarthritis and joint replacement. It asks "
    "nothing about sport, pivoting under load, or confidence in the knee, so after a "
    "ligament or cartilage operation it will miss a great deal of what matters. Track "
    "it if it is useful, but treat a good score as saying less than it would after a "
    "replacement."
)

OXFORD_KNEE_SCORE_NOTE = (
    "The Oxford Knee Score is not offered here. It is © Oxford University Innovation "
    "and needs a licence for anything beyond non-commercial research, so its twelve "
    "items cannot ship in this repository. A deployment that holds a licence can add "
    "it as a second instrument: the scoring is a plain sum of twelve items scored 0-4, "
    "0-48 with 48 best, and everything downstream of `score()` is instrument-agnostic."
)


def applicability_caveat(surgery_type: Optional[str]) -> Optional[str]:
    """
    The warning that has to travel with a score, or None.

    Factored out of `definition()` so that everywhere a score is presented —
    the questionnaire, the printed summary — applies one rule rather than each
    growing its own copy of the list.
    """
    return _APPLICABILITY_CAVEAT if surgery_type in _LIMITED_VALIDATION else None


def definition(instrument: str = KOOS_JR, surgery_type: Optional[str] = None) -> dict:
    """
    The questionnaire, in the shape the frontend renders.

    Item wording lives here and nowhere else. A form that hard-codes its own
    copy of the questions is a form that drifts — and a KOOS-JR whose items have
    been reworded is not a KOOS-JR any more, it is a bespoke survey whose scores
    happen to look comparable to everyone else's.
    """
    _require_known(instrument)

    return {
        "instrument":    KOOS_JR,
        "name":          display_name(KOOS_JR),
        "full_name":     "Knee injury and Osteoarthritis Outcome Score for Joint Replacement",
        "citation":      "Lyman et al., Clin Orthop Relat Res 474(6):1461-1471, 2016",
        "recall_period": RECALL_PERIOD,
        "min_score":     0,
        "max_score":     100,
        "higher_is_better": True,
        "score_meaning": (
            "0 is total knee disability and 100 is a knee with no symptoms at all. "
            "The scale is interval-calibrated, so a 10-point change means the same "
            "thing wherever on the scale it happens."
        ),
        "mcid":               MCID,
        "mdc":                MDC,
        "min_interval_days":  MIN_INTERVAL_DAYS,
        "due_after_days":     DUE_AFTER_DAYS,
        "response_options":   [dict(o) for o in RESPONSE_OPTIONS],
        "items": [
            {
                "code":    item.code,
                "section": item.section,
                "lead_in": item.lead_in,
                "prompt":  item.prompt,
            }
            for item in KOOS_JR_ITEMS
        ],
        "caveat": applicability_caveat(surgery_type),
    }


def _require_known(instrument: str) -> None:
    if instrument not in INSTRUMENTS:
        raise ValueError(
            f"Unknown outcome measure {instrument!r}. Available: {', '.join(INSTRUMENTS)}. "
            + OXFORD_KNEE_SCORE_NOTE
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def band(interval_score: float) -> tuple[str, str]:
    """(label, plain-English sentence) for a 0-100 score."""
    for floor, label, sentence in _BANDS:
        if interval_score >= floor:
            return label, sentence
    return _BANDS[-1][1], _BANDS[-1][2]        # unreachable; the last floor is 0


def score(responses: Sequence[int], instrument: str = KOOS_JR) -> dict:
    """
    Turn seven answers into a score.

    Raises ValueError on anything that is not a complete, in-range set of
    answers. Partial completion is refused rather than imputed: KOOS-JR has no
    published rule for a missing item, and averaging the six that were answered
    would produce a number that looks exactly like a real score and is not one.
    The caller's job is to collect all seven, which is why the form does not let
    you submit without them.
    """
    _require_known(instrument)

    if len(responses) != ITEM_COUNT:
        raise ValueError(
            f"KOOS-JR needs all {ITEM_COUNT} answers; got {len(responses)}. "
            "There is no published way to score a partly completed form."
        )

    for i, value in enumerate(responses):
        # bool is an int subclass, and True would otherwise score as 1.
        #
        # ValueError, not TypeError, despite this being a type problem: every way
        # of getting an answer wrong is one kind of failure to the caller, and
        # POST /me/outcome-scores catches ValueError to turn it into a 422. A
        # TypeError here would escape that and become a 500.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(  # noqa: TRY004
                f"Answer {i + 1} ({KOOS_JR_ITEMS[i].code}) must be a whole number."
            )
        if not MIN_RESPONSE <= value <= MAX_RESPONSE:
            raise ValueError(
                f"Answer {i + 1} ({KOOS_JR_ITEMS[i].code}) is {value}; "
                f"the scale runs {MIN_RESPONSE} to {MAX_RESPONSE}."
            )

    raw = sum(responses)
    interval = _KOOS_JR_INTERVAL[raw]
    label, sentence = band(interval)

    return {
        "instrument":     KOOS_JR,
        "raw_sum":        raw,
        "interval_score": round(interval, 1),
        "band":           label,
        "band_text":      sentence,
    }


def change(current: float, previous: Optional[float]) -> dict:
    """
    Describe the move from one score to the next.

    The single number is the least interesting part of a PROM. A patient at 58
    who was at 41 a month ago and a patient at 58 who was at 72 are in opposite
    situations, and only the second one needs a phone call.

    `direction` is deliberately three-valued. Anything inside ±MDC is reported as
    "unchanged", because a change smaller than the instrument can detect is not a
    small improvement — it is no information.
    """
    if previous is None:
        return {
            "delta":       None,
            "direction":   "first",
            "meaningful":  False,
            "summary":     "First recorded score — the baseline everything after this is read against.",
        }

    delta = round(current - previous, 1)

    if abs(delta) < MDC:
        return {
            "delta":      delta,
            "direction":  "unchanged",
            "meaningful": False,
            "summary": (
                f"{abs(delta):.0f} point{'s' if abs(delta) != 1 else ''} "
                f"{'up' if delta > 0 else 'down'} — within what the questionnaire can reliably "
                "tell apart, so it is best read as no change."
            ),
        }

    if delta > 0:
        return {
            "delta":      delta,
            "direction":  "improved",
            # The MCID is the "you would notice this" threshold; MDC only says
            # the instrument can see it.
            "meaningful": delta >= MCID,
            "summary": (
                f"Up {delta:.0f} points"
                + (" — an improvement large enough that most people feel it."
                   if delta >= MCID else
                   " — a real improvement, though a modest one.")
            ),
        }

    return {
        "delta":      delta,
        "direction":  "declined",
        "meaningful": abs(delta) >= MCID,
        "summary": (
            f"Down {abs(delta):.0f} points"
            + (" — a fall large enough to be worth raising with a physiotherapist."
               if abs(delta) >= MCID else
               " — a real fall, beyond the questionnaire's measurement noise.")
        ),
    }


# ---------------------------------------------------------------------------
# When to ask again
# ---------------------------------------------------------------------------

def _utc(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat those as the UTC they were stored as."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def schedule(last_recorded_at: Optional[datetime], now: Optional[datetime] = None) -> dict:
    """
    Whether it is worth asking again, and when it next will be.

    Three states rather than a boolean, because "not yet" and "too soon to be
    meaningful" are different answers to the patient. The second one is the
    reason `can_record` exists at all: a patient who wants to answer again the
    next morning should be told why that will not tell them anything, not
    silently allowed to fill the chart with noise.
    """
    now = now or datetime.now(timezone.utc)

    if last_recorded_at is None:
        return {
            "due":          True,
            "can_record":   True,
            "next_due_at":  now,
            "days_since":   None,
            "reason":       "No score recorded yet. This first one is the baseline.",
        }

    last = _utc(last_recorded_at)
    days = (now - last).days
    next_due = last + timedelta(days=DUE_AFTER_DAYS)
    can_record = days >= MIN_INTERVAL_DAYS

    if not can_record:
        when = "today" if days == 0 else f"{days} day{'s' if days != 1 else ''} ago"
        return {
            "due":         False,
            "can_record":  False,
            "next_due_at": next_due,
            "days_since":  days,
            "reason": (
                f"Answered {when}. Knees do not change measurably in less than "
                f"{MIN_INTERVAL_DAYS} days, so asking again now would record noise rather "
                "than progress."
            ),
        }

    if days >= DUE_AFTER_DAYS:
        return {
            "due":         True,
            "can_record":  True,
            "next_due_at": next_due,
            "days_since":  days,
            "reason":      f"Last answered {days} days ago.",
        }

    return {
        "due":         False,
        "can_record":  True,
        "next_due_at": next_due,
        "days_since":  days,
        "reason":      f"Last answered {days} days ago; next one due in {DUE_AFTER_DAYS - days}.",
    }
