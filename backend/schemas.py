from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class KneeSide(str, Enum):
    left  = "left"
    right = "right"
    both  = "both"


class SurgeryType(str, Enum):
    acl         = "acl"
    tkr         = "tkr"
    meniscus    = "meniscus"
    arthroscopy = "arthroscopy"
    none        = "none"


class TrackedJoint(str, Enum):
    knee  = "knee"
    ankle = "ankle"


class HoldTarget(str, Enum):
    straight = "straight"
    flexed   = "flexed"


class Exercise(BaseModel):
    name:          str
    description:   str
    target_reps:   int
    target_sets:   int
    angle_limit:   int           = Field(..., description="Effective angle limit in degrees (capped by X-ray severity)")
    hold_seconds:  Optional[int] = Field(None, description="Hold duration in seconds for isometric exercises")
    tracked_joint: TrackedJoint  = Field(TrackedJoint.knee, description="Joint the webcam tracker counts reps/holds from. Safety is always checked at the knee.")
    hold_target:   Optional[HoldTarget] = Field(None, description="Where the hold position sits relative to angle_limit: 'straight' (at or below) or 'flexed' (near it). Null for rep-counted exercises.")
    start_angle:   Optional[int] = Field(None, description="Unavoidable passive knee flexion the exercise starts from (seated/equipment). Null if it starts from extension.")
    hold_angle:    Optional[int] = Field(None, description="Angle to reach and hold, when it differs from angle_limit. Null means derive it from angle_limit.")
    instructions:  List[str]
    cautions:      Optional[str] = None
    angle_capped:  bool          = Field(False, description="True if X-ray severity reduced the standard protocol angle")


class ExcludedExercise(BaseModel):
    """
    An exercise withheld because its starting position exceeds the patient's
    safe ceiling. Reported rather than silently dropped so a physiotherapist can
    see what was removed and prescribe an alternative.
    """
    name:        str
    start_angle: int = Field(..., description="Passive knee flexion the exercise starts from, in degrees")
    max_angle:   int = Field(..., description="The patient's X-ray-derived ceiling that it exceeds")
    reason:      str


class ConfidenceBand(str, Enum):
    low      = "low"
    moderate = "moderate"
    high     = "high"


class SidePrescription(BaseModel):
    """
    One knee's own reading and its own exercise limits.

    Present only for a bilateral analysis. Two knees can differ by two KL grades
    and 45° of permitted flexion, and holding one to the other's limit is either
    unsafe or pointlessly restrictive depending on which way round it is.
    """

    knee_side:           KneeSide
    kl_grade:            int
    health_score:        int
    max_angle:           int
    confidence:          float
    confidence_band:     ConfidenceBand
    calibrated:          bool = False
    grade_probabilities: List[float] = Field(default_factory=lambda: [0.0] * 5)
    within_one_grade:    float = 0.0
    kl_applicable:       bool = True
    hardware_suspected:  bool = False
    ood_suspected:       bool = False
    explanation:         Optional[str] = None
    exercise_list:       List[Exercise] = Field(default_factory=list)
    excluded_exercises:  List[ExcludedExercise] = Field(default_factory=list)


class AnalyseXrayResponse(BaseModel):
    kl_grade:          int            = Field(..., ge=0, le=4,     description="KL Grade 0–4")
    health_score:      int            = Field(..., ge=0, le=100,   description="Joint health score 0–100")
    max_angle:         int            = Field(...,                  description="Safe flexion ceiling in degrees")
    confidence:        float          = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1. Only meaningful as a probability when calibrated is true.")
    confidence_band:   ConfidenceBand = Field(ConfidenceBand.low,  description="Banded confidence. Prefer this over the raw number for display.")
    calibrated:        bool           = Field(False, description="True if the checkpoint carries a fitted temperature. When false, confidence is raw softmax and overstates certainty.")
    ood_suspected:     bool           = Field(False, description="True if the image sits in the tail of the in-distribution energy range — grade with caution.")
    knee_side:         str
    surgery_type:      str
    weeks_post_op:     Optional[int]  = None
    rehab_phase:       str
    rehab_phase_label: str
    rehab_phase_goal:  str
    exercise_list:     List[Exercise]
    excluded_exercises: List[ExcludedExercise] = Field(
        default_factory=list,
        description="Protocol exercises withheld because their starting position exceeds the safe ceiling",
    )
    rationale:         str
    disclaimer:        str            = Field(..., description="Clinical disclaimer")
    model_version:     str
    demo_mode:         bool           = False
    bilateral:         bool           = Field(
        False,
        description="True when two X-rays were assessed. See `sides` for each knee.",
    )
    sides:             List[SidePrescription] = Field(
        default_factory=list,
        description=(
            "One entry per knee, each with its own grade and ceiling. Empty for a "
            "single-knee analysis. The flat fields above carry the MORE RESTRICTIVE "
            "side, so a client that ignores this still gets a safe answer."
        ),
    )
    grade_probabilities: List[float] = Field(
        default_factory=lambda: [0.0] * 5,
        description=(
            "Calibrated probability for each KL grade 0-4, in order. The model was trained "
            "with an ordinal loss, so mass beside the winner is the model saying the answer "
            "is nearby rather than noise — and the difference between grade 2 and grade 3 is "
            "30° of permitted flexion."
        ),
    )
    within_one_grade:  float          = Field(
        0.0,
        ge=0.0, le=1.0,
        description=(
            "Probability mass at the chosen grade or either neighbour. The checkpoint scores "
            "70.3% exact and 95.3% within one grade, so this is the figure that describes the "
            "reading at the scale the ceiling actually moves."
        ),
    )
    explanation:       Optional[str]  = Field(
        None,
        description=(
            "Grad-CAM overlay as a data: URI, when `explain` was requested and a real "
            "model produced one. Shows where the model was looking — not why it decided, "
            "but enough to see whether it was reading the joint or the corner of the film. "
            "Null in demo mode, or if the overlay could not be produced."
        ),
    )
    kl_applicable:     bool           = Field(
        True,
        description=(
            "False when the joint has been replaced. The KL grade is still reported — it is "
            "what the model said — but it set nothing: a grade measures wear on a native "
            "joint surface, and a resurfaced one no longer has that. The limits come from "
            "the surgical protocol instead."
        ),
    )
    hardware_suspected: bool          = Field(
        False,
        description=(
            "The image looks like it may contain a joint replacement that was not declared. "
            "A heuristic, not a classifier: it warns and never loosens a restriction."
        ),
    )
    hardware_note:     Optional[str]  = None
    status:            str            = Field(
        "draft",
        description=(
            "Always 'draft' here — a clinician has not seen it yet. The patient's "
            "history shows whether one later approved or changed it."
        ),
    )
    prescription_id:   Optional[str]  = Field(
        None,
        description=(
            "Identifier of the saved record, if the caller was signed in. Null "
            "for a guest analysis, and also if the save failed — in which case "
            "the result above is still valid, it just was not kept."
        ),
    )


class PrescriptionEffective(BaseModel):
    """
    What the patient should actually follow, and who stands behind it.

    `payload` is the effective version — the clinician's if they changed
    anything, the model's otherwise. `status` is how the patient knows which.
    """

    id:          str
    created_at:  datetime
    status:      str
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_note: Optional[str] = None
    payload:     dict


class ExercisesResponse(BaseModel):
    surgery_type:      str
    weeks_post_op:     Optional[int]
    kl_grade:          int
    rehab_phase:       str
    rehab_phase_label: str
    rehab_phase_goal:  str
    max_angle:         int
    exercise_list:     List[Exercise]
    excluded_exercises: List[ExcludedExercise] = Field(default_factory=list)
    disclaimer:        str


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    demo_mode:     bool
    calibrated:    bool = Field(False, description="True if the checkpoint carries a fitted temperature, so reported confidence is calibrated")
    ood_screening: bool = Field(False, description="True if the checkpoint carries an energy reference, so non-radiograph images can be screened out")

# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email:        EmailStr
    # The floor is enforced here so the API is safe on its own terms, not only
    # when called from a form that happens to validate. MIN_PASSWORD_LEN in
    # auth.py is the same number; this bound is what produces a 422 with a
    # readable message instead of a 500 from the hasher.
    password:     str           = Field(..., min_length=8, max_length=1024)
    display_name: Optional[str] = Field(None, max_length=120)


class LoginRequest(BaseModel):
    email:    EmailStr
    # No min_length: a short password is a failed login, not a malformed
    # request, and saying which one leaks whether the account exists.
    password: str = Field(..., max_length=1024)


class PatientOut(BaseModel):
    id:           str
    email:        EmailStr
    display_name: Optional[str] = None
    created_at:   datetime

    surgery_date: Optional[date]        = None
    surgery_type: Optional[SurgeryType] = None
    # Derived from surgery_date, never stored: a stored week count is wrong by
    # the next morning.
    weeks_post_op: Optional[int] = Field(
        None, description="Whole weeks since surgery_date, floored. Null if no date is on record."
    )

    model_config = ConfigDict(from_attributes=True)


class SurgeryUpdate(BaseModel):
    """
    Record (or clear) the operation this patient is recovering from.

    Both fields are explicitly nullable so that sending null clears them — a
    patient who entered the wrong date has to be able to take it back, and an
    omitted field is not the same as an emptied one.
    """

    surgery_date: Optional[date]        = None
    surgery_type: Optional[SurgeryType] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = Field(..., description="Seconds until the token expires")
    patient:      PatientOut


class PrescriptionSummary(BaseModel):
    """A row in the patient's history. The full stored payload is not included."""

    id:                str
    created_at:        datetime
    kl_grade:          int
    health_score:      int
    max_angle:         int
    knee_side:         str
    surgery_type:      str
    weeks_post_op:     Optional[int] = None
    rehab_phase:       str
    model_version:     str
    demo_mode:         bool
    # How much weight this carries: a draft is a machine's reading that nobody
    # has looked at, and the patient's own history should not present the two
    # as the same thing.
    status:            str = "draft"
    reviewed_at:       Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PrescriptionHistory(BaseModel):
    count:         int
    prescriptions: List[PrescriptionSummary]


# ---------------------------------------------------------------------------
# Exercise sessions
# ---------------------------------------------------------------------------

class SetRecord(BaseModel):
    """
    One completed set, as measured by the webcam tracker.

    Bounds are generous but real: they exist so that a bug in the tracker, or a
    hand-rolled request, cannot write a 4000° flexion into a clinical record.
    """

    # Groups sets from one sitting. Chosen by the browser so the first set does
    # not have to wait for a server round trip to know where it belongs.
    client_session_id: str = Field(..., min_length=8, max_length=64)
    prescription_id:   Optional[str] = Field(None, max_length=32)

    exercise_name: str          = Field(..., min_length=1, max_length=120)
    tracked_joint: TrackedJoint = TrackedJoint.knee
    knee_side:     KneeSide     = KneeSide.right

    angle_limit:  int           = Field(..., ge=0, le=180)
    target_reps:  Optional[int] = Field(None, ge=0, le=500)
    target_sets:  int           = Field(..., ge=1, le=50)
    hold_seconds: Optional[int] = Field(None, ge=0, le=3600)

    set_index:        int   = Field(..., ge=1, le=50)
    reps_completed:   int   = Field(0, ge=0, le=500)
    duration_seconds: float = Field(0.0, ge=0, le=86400)

    peak_flexion_deg: float = Field(0.0, ge=0, le=180)
    breach_count:     int   = Field(0, ge=0, le=10000)
    breach_seconds:   float = Field(0.0, ge=0, le=86400)

    hold_seconds_achieved: Optional[float] = Field(None, ge=0, le=3600)
    mean_visibility:       Optional[float] = Field(None, ge=0.0, le=1.0)
    suspended_seconds:     float           = Field(0.0, ge=0, le=86400)

    # False when the patient overrode the camera-view gate. Angles measured from
    # an unverified viewpoint read low, so anything charting range of motion has
    # to be able to leave them out.
    view_verified: bool = True

    # Numeric Pain Rating Scale, 0 (none) to 10 (worst imaginable), taken before
    # starting. Sent with every set; the server keeps it once, when the session
    # is created. Optional — a rating is asked for, never required.
    pain_before: Optional[int] = Field(None, ge=0, le=10)


class SessionReport(BaseModel):
    """
    How the session felt, recorded once it is over.

    Separate from the set posts because both readings only exist at the end:
    pain afterwards is the second half of a pair whose first half was taken
    before starting, and exertion is a judgement about the whole sitting.
    """

    client_session_id: str = Field(..., min_length=8, max_length=64)
    # NPRS again, so before and after are the same instrument and comparable.
    pain_after: Optional[int] = Field(None, ge=0, le=10)
    # Borg CR10: 0 nothing at all, 10 maximal.
    rpe: Optional[int] = Field(None, ge=0, le=10)


class SetOut(BaseModel):
    set_index:             int
    reps_completed:        int
    duration_seconds:      float
    peak_flexion_deg:      float
    breach_count:          int
    breach_seconds:        float
    hold_seconds_achieved: Optional[float] = None
    mean_visibility:       Optional[float] = None
    suspended_seconds:     float
    recorded_at:           datetime

    model_config = ConfigDict(from_attributes=True)


class SessionOut(BaseModel):
    id:              str
    exercise_name:   str
    tracked_joint:   str
    knee_side:       str
    angle_limit:     int
    target_reps:     Optional[int] = None
    target_sets:     int
    hold_seconds:    Optional[int] = None
    started_at:      datetime
    last_set_at:     datetime
    sets_completed:  int
    completed:       bool
    view_verified:   bool
    pain_before:     Optional[int] = None
    pain_after:      Optional[int] = None
    rpe:             Optional[int] = None
    prescription_id: Optional[str] = None
    sets:            List[SetOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class SessionHistory(BaseModel):
    count:    int
    sessions: List[SessionOut]


# ---------------------------------------------------------------------------
# Patient-reported outcome measures
# ---------------------------------------------------------------------------
#
# The scoring, the item wording and every threshold live in outcome_measures.py.
# These are only the shapes they travel in.


class OutcomeInstrumentName(str, Enum):
    koos_jr = "koos_jr"


class OutcomeOption(BaseModel):
    value: int
    label: str


class OutcomeItem(BaseModel):
    code:    str = Field(..., description="The item's identifier in the full KOOS, e.g. 'P5'")
    section: str = Field(..., description="Pain, Stiffness or Function")
    lead_in: str = Field(..., description="The question stem, which several items share")
    prompt:  str = Field(..., description="The activity being rated")


class OutcomeInstrument(BaseModel):
    """
    The questionnaire itself, served so the form has one source of truth.

    A frontend that hard-codes its own copy of the items is a frontend that
    drifts, and a KOOS-JR whose wording has been edited is not a KOOS-JR — it is
    a bespoke survey whose scores only look comparable to everyone else's.
    """

    instrument:       OutcomeInstrumentName
    name:             str
    full_name:        str
    citation:         str
    recall_period:    str
    min_score:        int
    max_score:        int
    higher_is_better: bool
    score_meaning:    str
    mcid:             float = Field(..., description="Change a patient would notice, in points")
    mdc:              float = Field(..., description="Smallest change the instrument can reliably detect")
    min_interval_days: int
    due_after_days:    int
    response_options: List[OutcomeOption]
    items:            List[OutcomeItem]
    caveat: Optional[str] = Field(
        None,
        description=(
            "Set when the instrument is a poor fit for this patient's operation — "
            "KOOS-JR asks nothing about sport or pivoting, so it misses most of what "
            "matters after an ACL reconstruction."
        ),
    )


class OutcomeScoreCreate(BaseModel):
    """
    One completed questionnaire.

    All seven answers or none. KOOS-JR has no published rule for a missing item,
    and averaging the six that were answered produces a number indistinguishable
    from a real score that is not one.
    """

    instrument: OutcomeInstrumentName = OutcomeInstrumentName.koos_jr
    knee_side:  KneeSide = KneeSide.right
    responses:  List[int] = Field(
        ...,
        min_length=7, max_length=7,
        description="The seven answers in item order, each 0 (none) to 4 (extreme).",
    )


class OutcomeChange(BaseModel):
    """
    The move from the previous score to this one — the part that carries meaning.

    A patient at 58 who was at 41 a month ago and a patient at 58 who was at 72
    are in opposite situations, and only one of them needs a phone call.
    """

    delta:      Optional[float] = None
    direction:  str = Field(..., description="'first' | 'improved' | 'declined' | 'unchanged'")
    meaningful: bool = Field(
        ..., description="True when the change is at least the MCID — large enough to be felt"
    )
    summary:    str


class OutcomeScoreOut(BaseModel):
    id:             str
    instrument:     OutcomeInstrumentName
    knee_side:      str
    recorded_at:    datetime
    weeks_post_op:  Optional[int] = None
    raw_sum:        int = Field(..., description="Sum of the seven answers, 0-28, where higher is worse")
    interval_score: float = Field(
        ..., description="The Rasch-calibrated 0-100 score, where higher is better"
    )
    band:           str
    band_text:      str
    responses:      List[int] = Field(default_factory=list)
    # Against the previous score for the SAME knee. Absent on a bare list where
    # no comparison was computed.
    change:         Optional[OutcomeChange] = None


class OutcomeSchedule(BaseModel):
    """Whether it is worth asking again, and when it next will be."""

    due:         bool
    can_record:  bool = Field(
        ..., description="False inside the minimum interval, when another answer would only add noise"
    )
    next_due_at: datetime
    days_since:  Optional[int] = None
    reason:      str


class OutcomePoint(BaseModel):
    date:           date
    interval_score: float
    raw_sum:        int
    weeks_post_op:  Optional[int] = None


class OutcomeSeries(BaseModel):
    """
    One knee's scores over time.

    Per side for the same reason range of motion is per exercise: KOOS-JR asks
    about "your knee", singular, and someone with two bad knees has two different
    answers. Averaging them hides exactly the difference worth seeing.
    """

    instrument:    OutcomeInstrumentName
    knee_side:     str
    count:         int
    baseline:      float = Field(..., description="The first score recorded for this knee")
    latest:        float
    best:          float
    latest_at:     datetime
    band:          str
    band_text:     str
    change_from_previous: OutcomeChange
    change_from_baseline: OutcomeChange
    points:        List[OutcomePoint]


class OutcomeHistory(BaseModel):
    instrument: OutcomeInstrumentName
    schedule:   OutcomeSchedule
    count:      int
    scores:     List[OutcomeScoreOut]


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

class ProgressSummary(BaseModel):
    sessions:            int
    sets:                int
    reps:                int
    active_days:         int
    current_streak_days: int = Field(..., description="Consecutive days ending today or yesterday")
    longest_streak_days: int
    # These describe the primary exercise only — the one with the most measured
    # days — because a "best flexion" pooled across exercises compares a squat
    # against a straight-leg hold and means nothing.
    primary_exercise:    Optional[str]   = None
    best_flexion_deg:    Optional[float] = Field(
        None, description="Furthest verified flexion for the primary exercise. Null if nothing measurable yet."
    )
    latest_flexion_deg:  Optional[float] = None
    latest_pain_after:   Optional[int] = None
    # Mean of (after − before) across sessions that reported both. Positive
    # means sessions are typically leaving the patient in more pain than they
    # started with, which is the direction worth noticing.
    mean_pain_change:    Optional[float] = None
    sessions_with_pain:  int = 0
    unverified_sessions: int = Field(
        0,
        description=(
            "Sessions performed with the camera-view gate overridden. Counted for "
            "adherence but left out of every angle, because a knee measured "
            "square-on to the camera reads far lower than it really is."
        ),
    )
    # The patient's own verdict, which is the one figure here that nothing else
    # on the page can substitute for. Taken from the knee with the most recent
    # questionnaire; `outcome_measures` below carries each side in full.
    latest_outcome_score: Optional[float] = Field(
        None, description="Most recent KOOS-JR interval score, 0-100, where higher is better"
    )
    outcome_band:         Optional[str]   = None
    outcome_recorded_at:  Optional[datetime] = None


class RomPoint(BaseModel):
    """One day's furthest verified flexion, against the ceiling in force."""

    date:             date
    peak_flexion_deg: float
    angle_limit:      int
    sessions:         int


class RomSeries(BaseModel):
    """
    Range of motion for ONE exercise over time.

    Kept per exercise on purpose. Peak flexion only means something relative to
    what the exercise asked for: Quad Sets are held with the knee locked straight
    at about 4°, Mini Squats bend past 50°. Charting the best of both together
    makes a day of quad sets look like a collapse in range of motion, when the
    patient did exactly what was prescribed.
    """

    exercise_name: str
    days_measured: int
    latest_deg:    float
    best_deg:      float
    angle_limit:   int = Field(..., description="The ceiling in force at the most recent session")
    points:        List[RomPoint]


class PainPoint(BaseModel):
    """
    One day's pain, before and after exercising.

    The pair is the point. Rehab that hurts a little during and settles
    afterwards is working; the same exercise leaving someone worse every time is
    not — and neither shows up in range of motion.
    """

    date:        date
    pain_before: Optional[float] = None
    pain_after:  Optional[float] = None
    sessions:    int


class AdherenceDay(BaseModel):
    date:               date
    sessions:           int
    sets:               int
    completed_sessions: int = Field(..., description="Sessions where every prescribed set was finished")


class ExerciseBreakdown(BaseModel):
    exercise_name:    str
    sessions:         int
    sets:             int
    reps:             int
    peak_flexion_deg: Optional[float] = Field(None, description="Verified sessions only")
    breach_count:     int
    breach_seconds:   float
    mean_visibility:  Optional[float] = None


class ProgressResponse(BaseModel):
    range_days:      int
    tz_offset_minutes: int = Field(
        ..., description="Minutes east of UTC used to bucket days, as sent by the browser"
    )
    generated_at:    datetime
    summary:         ProgressSummary
    # Ordered by how much data each has, so the first is the one worth charting.
    rom_by_exercise: List[RomSeries]
    pain_trend:      List[PainPoint]
    adherence:       List[AdherenceDay]
    by_exercise:     List[ExerciseBreakdown]
    # One entry per knee assessed. Deliberately NOT clipped to `range_days`: a
    # questionnaire answered monthly has three or four points in ninety days, and
    # the baseline it is all measured against is usually older than the window.
    # A trend that dropped its own starting point would be worse than no trend.
    outcome_measures: List[OutcomeSeries] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Share links
# ---------------------------------------------------------------------------

class ShareLinkCreate(BaseModel):
    label: Optional[str] = Field(
        None, max_length=120,
        description="The patient's own note about who it went to, e.g. 'Dr Okonkwo, 14 Aug'.",
    )
    # Two weeks covers an appointment and the follow-up. Long enough to be
    # useful, short enough that a forgotten link stops working on its own.
    days: int = Field(14, ge=1, le=90, description="How long the link stays open")


class ShareLinkOut(BaseModel):
    """A link as the patient sees it. Never carries the token."""

    id:               str
    label:            Optional[str] = None
    created_at:       datetime
    expires_at:       datetime
    revoked_at:       Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    access_count:     int
    is_active:        bool

    model_config = ConfigDict(from_attributes=True)


class ShareLinkCreated(ShareLinkOut):
    """
    The one and only time the token is returned.

    Only its hash is stored, so this response is unrecoverable — matching how a
    password reset or an API key works, and for the same reason.
    """

    token: str
    path:  str = Field(..., description="Where to open it, e.g. /share.html?t=…")


class SharedProgress(BaseModel):
    """
    What a clinician sees when they open a share link.

    Deliberately narrower than the patient's own view: no email, no X-ray, no
    stored prescriptions. A display name only if the patient set one — a
    clinician needs to know whose knee this is, and nothing beyond that is
    required to read the charts.
    """

    patient_name: Optional[str] = None
    shared_at:    datetime
    expires_at:   datetime
    label:        Optional[str] = None
    progress:     ProgressResponse


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

class FlagSeverity(str, Enum):
    info    = "info"
    warning = "warning"
    urgent  = "urgent"


class FlagOut(BaseModel):
    """
    One reason a human should look.

    A triage signal, not a finding: it moves a patient up a list, and nothing
    here decides anything on its own. `evidence` carries the numbers it was
    raised on so a clinician can disagree with it at a glance.
    """

    code:     str
    severity: FlagSeverity
    summary:  str
    evidence: dict = Field(default_factory=dict)


class PatientFlag(BaseModel):
    """The same finding, worded for the person it is about."""

    code:     str
    severity: FlagSeverity
    message:  str


class PatientFlags(BaseModel):
    flags: List[PatientFlag] = Field(default_factory=list)
    worst: Optional[FlagSeverity] = None


class PatientFlagsForClinician(BaseModel):
    patient_id: str
    flags:      List[FlagOut] = Field(default_factory=list)
    worst:      Optional[FlagSeverity] = None


# ---------------------------------------------------------------------------
# Clinicians and care links
# ---------------------------------------------------------------------------

class ClinicianRegister(BaseModel):
    email:        EmailStr
    password:     str           = Field(..., min_length=8, max_length=1024)
    display_name: Optional[str] = Field(None, max_length=120)
    # Free text: registration bodies and title conventions differ by country,
    # and a dropdown of guesses would be wrong more often than useful.
    registration: Optional[str] = Field(None, max_length=64, description="e.g. HCPC PH123456")


class ClinicianOut(BaseModel):
    id:           str
    email:        EmailStr
    display_name: Optional[str] = None
    registration: Optional[str] = None
    created_at:   datetime

    model_config = ConfigDict(from_attributes=True)


class ClinicianToken(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int
    clinician:    ClinicianOut


class InviteCreate(BaseModel):
    patient_label: Optional[str] = Field(
        None, max_length=120,
        description="The clinician's own note for who this code is for, before it is redeemed.",
    )
    # Short, because the code is short. An invite that hangs around for months
    # is a credential nobody is watching.
    days: int = Field(7, ge=1, le=30)


class InviteOut(BaseModel):
    id:            str
    patient_label: Optional[str] = None
    invite_hint:   str = Field(..., description="First characters of the code, to tell pending invites apart")
    created_at:    datetime
    expires_at:    datetime
    is_pending:    bool

    model_config = ConfigDict(from_attributes=True)


class InviteCreated(InviteOut):
    """The one time the code is returned. Only its hash is stored."""

    code: str


class RedeemInvite(BaseModel):
    # Dashes and case are cosmetic; the server normalises before hashing.
    code: str = Field(..., min_length=4, max_length=32)


class CareLinkOut(BaseModel):
    """A link as either side sees it."""

    id:               str
    clinician_name:   Optional[str] = None
    clinician_email:  Optional[EmailStr] = None
    patient_name:     Optional[str] = None
    accepted_at:      Optional[datetime] = None
    revoked_at:       Optional[datetime] = None
    revoked_by:       Optional[str] = None
    is_active:        bool


class CaseloadEntry(BaseModel):
    """
    One patient on a clinician's list, with just enough to triage by.

    Deliberately not the whole record: this is the screen a clinician scans, and
    what belongs on it is what tells them who needs looking at. The detail is one
    click away.
    """

    link_id:         str
    patient_id:      str
    patient_name:    Optional[str] = None
    patient_label:   Optional[str] = None
    surgery_type:    Optional[SurgeryType] = None
    weeks_post_op:   Optional[int] = None
    linked_at:       Optional[datetime] = None

    last_session_at:     Optional[datetime] = None
    sessions_last_7_days: int = 0
    latest_flexion_deg:  Optional[float] = None
    latest_pain_after:   Optional[int] = None
    breaches_last_7_days: int = 0

    # The patient's own verdict on the knee, and which way it has moved. On the
    # caseload rather than one click in, because a score that has fallen since
    # last month is the single best reason to open a record — and it is invisible
    # in adherence, which often looks fine right up until someone gives up.
    latest_outcome_score: Optional[float] = None
    outcome_recorded_at:  Optional[datetime] = None
    outcome_change:       Optional[float] = Field(
        None, description="Points moved since the previous questionnaire; negative is worse"
    )

    # Why this patient might need looking at, worst first. Carried on the
    # caseload so a clinician can sort by it rather than opening thirty records
    # to find the two that changed.
    flags:        List[FlagOut] = Field(default_factory=list)
    worst_flag:   Optional[FlagSeverity] = None


class Caseload(BaseModel):
    count:    int
    patients: List[CaseloadEntry]


# ---------------------------------------------------------------------------
# Clinical review
# ---------------------------------------------------------------------------

class ReviewAction(str, Enum):
    keep   = "keep"
    remove = "remove"
    adjust = "adjust"
    add    = "add"


class PrescriptionStatus(str, Enum):
    draft               = "draft"
    clinician_approved  = "clinician_approved"
    clinician_modified  = "clinician_modified"


class ExerciseDecision(BaseModel):
    """What a clinician decided about one exercise."""

    name:   str          = Field(..., min_length=1, max_length=120)
    action: ReviewAction

    # Only meaningful for adjust/add. Omitted means "leave as drafted" for an
    # adjustment, or "use the catalogue's own limit" for an addition.
    angle_limit:  Optional[int] = Field(None, ge=0, le=180)
    target_reps:  Optional[int] = Field(None, ge=1, le=200)
    target_sets:  Optional[int] = Field(None, ge=1, le=20)
    hold_seconds: Optional[int] = Field(None, ge=1, le=600)

    # Required by the API whenever the change loosens a restriction. Tightening
    # needs no justification; loosening is the direction that can hurt.
    reason: Optional[str] = Field(None, max_length=1000)


class ReviewRequest(BaseModel):
    decisions: List[ExerciseDecision] = Field(default_factory=list)
    note:      Optional[str] = Field(None, max_length=2000)

    # Raises or lowers the whole safe flexion ceiling the X-ray produced. A
    # surgeon who replaced the joint knows things the radiograph does not.
    ceiling:        Optional[int] = Field(None, ge=0, le=180)
    ceiling_reason: Optional[str] = Field(None, max_length=1000)


class AuditEntry(BaseModel):
    exercise_name:  Optional[str] = None
    field:          str
    previous_value: Optional[str] = None
    new_value:      Optional[str] = None
    actor:          str = Field(..., description="'model', or the clinician's id")
    clinician_name: Optional[str] = None
    reason:         Optional[str] = None
    created_at:     datetime

    model_config = ConfigDict(from_attributes=True)


class PrescriptionDetail(BaseModel):
    """
    One prescription as a clinician reviews it.

    Both versions travel together: `draft` is what the model produced and is
    never rewritten, `effective` is what the patient is actually following. Until
    a review happens they are the same document.
    """

    id:             str
    patient_id:     str
    patient_name:   Optional[str] = None
    created_at:     datetime
    status:         PrescriptionStatus
    reviewed_at:    Optional[datetime] = None
    reviewed_by:    Optional[str] = None
    review_note:    Optional[str] = None

    kl_grade:       int
    max_angle:      int = Field(..., description="The ceiling in force, after any clinician override")
    model_max_angle: int = Field(..., description="What the X-ray alone produced")
    model_version:  str
    demo_mode:      bool
    confidence_band: Optional[str] = None
    calibrated:      bool = False

    draft:     dict
    effective: dict
    audit:     List[AuditEntry] = Field(default_factory=list)


class PrescriptionSummaryForClinician(BaseModel):
    id:           str
    created_at:   datetime
    status:       PrescriptionStatus
    kl_grade:     int
    max_angle:    int
    surgery_type: str
    weeks_post_op: Optional[int] = None
    rehab_phase:  str
    reviewed_at:  Optional[datetime] = None


class CatalogueExercise(BaseModel):
    """An exercise a clinician can add, with the fields the tracker needs."""

    name:          str
    description:   str
    target_reps:   int
    target_sets:   int
    angle_limit:   int = Field(..., description="The protocol's own limit, before any ceiling is applied")
    hold_seconds:  Optional[int] = None
    tracked_joint: TrackedJoint = TrackedJoint.knee
    hold_target:   Optional[HoldTarget] = None
    start_angle:   Optional[int] = None
    instructions:  List[str]
    cautions:      Optional[str] = None
