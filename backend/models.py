"""
models.py — the persistent record.

Two tables for now:

  Patient        an account. Email + Argon2id hash, nothing clinical.
  Prescription   the result of one X-ray analysis, kept so that a course of
                 rehab can be looked at over weeks instead of one page load.
  ExerciseSession / ExerciseSet
                 what the webcam tracker measured while the exercise was
                 performed — the readings it used to discard at the end of a set.
  OutcomeScore   one completed KOOS-JR questionnaire: what the patient says
                 about the knee, on a scale a registry would recognise.
  ShareLink      a revocable, expiring, read-only window onto one patient's
                 progress, openable by a clinician without an account.
  Clinician      a physiotherapist or surgeon, with a caseload rather than a knee.
  CareLink       one clinician looking after one patient — and, before it is
                 redeemed, the invite that will create it.

The uploaded X-ray itself is deliberately NOT stored. Keeping the derived grade
is what the progress view and the clinician review need; keeping the image turns
this into a system holding diagnostic imaging, which is a much larger promise
about storage, retention and access than the app is ready to make.

"""

import uuid
from datetime import date as dt_date
from datetime import datetime, timezone

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """Timezone-aware UTC. `datetime.utcnow` is naive and deprecated in 3.12+."""
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)

    # Stored lower-cased and stripped (see auth.normalise_email) so that
    # Ada@example.com and ada@example.com cannot become two accounts. The unique
    # index enforces it at the database level rather than trusting callers.
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)

    # Argon2id output, which carries its own salt and parameters. Length is
    # generous so a future parameter change does not need a schema change.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ── The operation ────────────────────────────────────────────────────────
    # Stored once so weeks-post-op can be derived instead of retyped. That field
    # selects the entire rehab protocol, and asking a patient to recompute it
    # from memory on every visit is how someone ends up prescribed a phase they
    # are not in — silently, because a plausible wrong number looks exactly like
    # a plausible right one.
    #
    # A date, not a datetime: nobody knows or needs the hour, and a timezone on
    # it would only invite the kind of off-by-one that moves someone a week.
    surgery_date: Mapped[dt_date | None] = mapped_column(Date, nullable=True)

    # One current operation per patient. A full surgical history would be its
    # own table; this is the thing the protocol selector actually reads.
    surgery_type: Mapped[str | None] = mapped_column(String(16), nullable=True)


    # ── Session and credential control ───────────────────────────────────────
    # Every token carries the version this counter stood at when it was issued,
    # and a token whose version no longer matches is refused. Bumping it is what
    # makes "sign out everywhere" and "changing your password ends other
    # sessions" possible at all: a JWT is otherwise valid until it expires, and
    # there was no way to take one back.
    #
    # A counter rather than a cutoff timestamp, because `iat` is only accurate
    # to the second: a token issued and revoked inside the same second would
    # have survived a timestamp comparison. A counter has no clock in it, so
    # signing out always takes effect immediately.
    #
    # And a counter rather than a table of revoked tokens: one column, nothing
    # to sweep up later, and it revokes at the only granularity that is
    # actually useful after a password change — all of them.
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # A pending password reset. Only the hash is kept, for the same reason the
    # password is: this column is a credential while it lives, and a readable
    # one in a stolen backup would be an account takeover.
    reset_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reset_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    care_links: Mapped[list["CareLink"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    share_links: Mapped[list["ShareLink"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="ShareLink.created_at.desc()",
    )

    prescriptions: Mapped[list["Prescription"]] = relationship(
        back_populates="patient",
        # A deleted account takes its clinical records with it; leaving orphans
        # behind would be both wrong and a data-protection problem.
        cascade="all, delete-orphan",
        order_by="Prescription.created_at.desc()",
    )

    # Oldest first: these are read as a trend, and the first one is the baseline
    # every later score is compared against.
    outcome_scores: Mapped[list["OutcomeScore"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="OutcomeScore.recorded_at",
    )

    @property
    def weeks_post_op(self) -> int | None:
        """
        Whole weeks since the operation, derived on read.

        Never stored: a week count written to the database is wrong by the next
        Monday. A future date returns None rather than a negative number — the
        API refuses to store one, but a clock skew or a hand-edited row should
        degrade to "unknown", not to a phase before surgery.
        """
        if self.surgery_date is None:
            return None
        from clinical_logic import weeks_since_surgery

        try:
            return weeks_since_surgery(self.surgery_date)
        except ValueError:
            return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Patient {self.email}>"


class Prescription(Base):
    """
    One analysis: what the model read off the X-ray, and the plan built from it.

    The full API response is kept verbatim in `payload` so a historical record
    still renders correctly after the exercise database or the wording changes.
    The columns beside it are the fields worth querying and charting — they are a
    denormalised copy of what is already inside the payload, which is a trade
    made knowingly: without them, plotting a range-of-motion trend means parsing
    every stored document.
    """

    __tablename__ = "prescriptions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )

    kl_grade: Mapped[int] = mapped_column(Integer, nullable=False)
    health_score: Mapped[int] = mapped_column(Integer, nullable=False)
    max_angle: Mapped[int] = mapped_column(Integer, nullable=False)

    knee_side: Mapped[str] = mapped_column(String(8), nullable=False)
    surgery_type: Mapped[str] = mapped_column(String(16), nullable=False)
    weeks_post_op: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rehab_phase: Mapped[str] = mapped_column(String(32), nullable=False)

    # Which model produced this, and whether its confidence was calibrated. A
    # grade read by an uncalibrated checkpoint should not be compared against one
    # that was, and without recording it here that distinction is lost the moment
    # the weights are replaced.
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    demo_mode: Mapped[bool] = mapped_column(default=False, nullable=False)

    # The model's own output. Never rewritten, whatever a clinician later
    # decides: the point of an audit trail is that the original is still there.
    payload: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Clinical review ──────────────────────────────────────────────────────
    # This is what turns the model from a prescriber into a drafter. Until a
    # clinician looks at it, a prescription is a draft — a machine's reading of
    # an X-ray, and a movement ceiling derived from it, with nobody accountable
    # for either.
    #
    #   draft               nobody has reviewed it
    #   clinician_approved  reviewed and left as it stood
    #   clinician_modified  reviewed and changed; approved_payload is what the
    #                       patient should follow
    status: Mapped[str] = mapped_column(String(24), default="draft", nullable=False)

    reviewed_by_clinician_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("clinicians.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The clinician's version, when they changed something. Null means follow
    # `payload` — either because nobody has reviewed it, or because they read it
    # and agreed.
    approved_payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="prescriptions")
    reviewed_by: Mapped["Clinician | None"] = relationship()
    audit: Mapped[list["PrescriptionAudit"]] = relationship(
        back_populates="prescription",
        cascade="all, delete-orphan",
        order_by="PrescriptionAudit.created_at",
    )

    @property
    def effective_payload(self) -> str:
        """What the patient should actually follow."""
        return self.approved_payload or self.payload

    @property
    def is_reviewed(self) -> bool:
        return self.status != "draft"

    __table_args__ = (
        # Every read of this table is "the newest few for one patient".
        Index("ix_prescriptions_patient_created", "patient_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Prescription {self.id} KL{self.kl_grade} {self.created_at:%Y-%m-%d}>"


class ExerciseSession(Base):
    """
    One sitting with the tracker: a single exercise, worked through set by set.

    Created the first time a set is reported and updated as later sets arrive,
    rather than opened at launch and closed at the end. A patient who shuts the
    laptop after two of three sets is the normal case, not an error case, and
    that shape means their two sets are already durable with nothing to
    reconcile afterwards.

    The prescription's numbers are copied onto the row instead of being read
    through `prescription_id` at display time. A clinician can raise or lower the
    ceiling later (and will, once override exists), and a session has to keep
    showing the limit that was actually in force while it was performed.
    """

    __tablename__ = "exercise_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )

    # Supplied by the browser so that sets from one sitting group together
    # without the client having to wait for a server id before the first set.
    # Unique per patient, not globally: it is a client-chosen value and two
    # patients colliding must not merge their sessions.
    client_session_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # Nullable: the tracker can be opened directly, without an analysis behind
    # it, and a prescription may be deleted while its sessions remain.
    prescription_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("prescriptions.id", ondelete="SET NULL"), nullable=True
    )

    exercise_name: Mapped[str] = mapped_column(String(120), nullable=False)
    tracked_joint: Mapped[str] = mapped_column(String(8), nullable=False, default="knee")
    knee_side: Mapped[str] = mapped_column(String(8), nullable=False, default="right")

    angle_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    target_reps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_sets: Mapped[int] = mapped_column(Integer, nullable=False)
    hold_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_set_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    sets_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed: Mapped[bool] = mapped_column(default=False, nullable=False)

    # False when the patient overrode the camera-view gate. Angles from an
    # unverified viewpoint read low, so these sessions must never be mixed into
    # a range-of-motion trend as though they were measured properly.
    view_verified: Mapped[bool] = mapped_column(default=True, nullable=False)

    # ── What the patient felt ────────────────────────────────────────────────
    # Pain on the Numeric Pain Rating Scale, 0 (none) to 10 (worst imaginable),
    # taken before and after the session. The pair matters more than either
    # number: rehab that hurts a little during and settles afterwards is working,
    # and the same exercise leaving someone worse every time is not, and neither
    # is visible from range of motion alone.
    #
    # Nullable throughout — a rating is asked for, never required. Blocking the
    # camera behind a mandatory form is how people stop opening the app.
    pain_before: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pain_after: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Perceived exertion, Borg CR10 (0 = nothing at all, 10 = maximal). Held on
    # the session rather than per set: asking after each set needs a rest screen
    # between sets, which the tracker does not have yet. One honest
    # session-level rating beats three copies of the same guess.
    rpe: Mapped[int | None] = mapped_column(Integer, nullable=True)

    sets: Mapped[list["ExerciseSet"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ExerciseSet.set_index",
    )

    __table_args__ = (
        # Makes the upsert-by-client-id lookup a single indexed hit, and stops a
        # retried request opening a second session for the same sitting.
        UniqueConstraint("patient_id", "client_session_id", name="uq_session_patient_client"),
        Index("ix_sessions_patient_started", "patient_id", "started_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ExerciseSession {self.exercise_name} {self.sets_completed}/{self.target_sets}>"


class ExerciseSet(Base):
    """
    One completed set, and what the tracker measured while it was performed.

    This is the row that did not exist before: tracker.html computed every one of
    these numbers at thirty frames a second and threw them away when the set
    ended.
    """

    __tablename__ = "exercise_sets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("exercise_sessions.id", ondelete="CASCADE"), nullable=False
    )

    set_index: Mapped[int] = mapped_column(Integer, nullable=False)   # 1-based
    reps_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # The headline number for a progress chart: how far the knee actually bent.
    peak_flexion_deg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Time spent past the safe ceiling, and how many separate times it happened.
    # One long breach and six brief ones mean different things clinically.
    breach_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    breach_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Isometric holds only: how long the position was actually held against the
    # target. Null for rep-counted exercises.
    hold_seconds_achieved: Mapped[float | None] = mapped_column(Float, nullable=True)

    # How well the camera could see the leg (mean of the minimum landmark
    # visibility per frame), and how long tracking was suspended because it
    # could not. Both qualify how much the numbers above are worth.
    mean_visibility: Mapped[float | None] = mapped_column(Float, nullable=True)
    suspended_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    session: Mapped[ExerciseSession] = relationship(back_populates="sets")

    __table_args__ = (
        # A dropped connection makes the browser retry; without this a set could
        # be counted twice.
        UniqueConstraint("session_id", "set_index", name="uq_set_session_index"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ExerciseSet {self.set_index} reps={self.reps_completed} peak={self.peak_flexion_deg:.0f}°>"


class OutcomeScore(Base):
    """
    One completed patient-reported outcome questionnaire.

    Everything else in this file is something the app measured. This is the one
    row that records what the patient says, which is the only place the question
    they actually care about — is my knee getting better — is ever answered.

    Both the answers and the score are stored. That is not redundant:

      * `responses` is the evidence. A scoring bug, a corrected lookup table or a
        second instrument added later can all be recomputed from it, and none of
        that is possible from a total.
      * `interval_score` is what gets charted and compared, and it is written
        once. Deriving it on read would mean a chart silently redrawing itself
        the day the scoring code changes — including the historical points, which
        were answered against the old one.

    `weeks_post_op` is a snapshot for the same reason a prescription's is: it is
    derived from surgery_date, and a patient who corrects that date a year later
    must not retroactively move every questionnaire they have ever answered to a
    different point in their recovery.
    """

    __tablename__ = "outcome_scores"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )

    # 'koos_jr' today. A column rather than an assumption, because the score
    # ranges and the direction of "better" differ between instruments, and a
    # chart that mixed two of them would be meaningless.
    instrument: Mapped[str] = mapped_column(String(24), nullable=False, default="koos_jr")

    # KOOS-JR asks about "your knee", singular. Someone with two bad knees has
    # two different answers, and pooling them would average away the difference
    # that matters — the same reason range of motion is kept per exercise.
    knee_side: Mapped[str] = mapped_column(String(8), nullable=False, default="right")

    # The seven raw answers, 0-4 each, as a JSON array in item order. Item order
    # is fixed by outcome_measures.KOOS_JR_ITEMS and is what makes this
    # array interpretable at all.
    responses: Mapped[str] = mapped_column(Text, nullable=False)

    raw_sum: Mapped[int] = mapped_column(Integer, nullable=False)
    # The Rasch-calibrated 0-100 score, where higher is better. Float because the
    # published lookup table is not integral.
    interval_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Where in the recovery this was answered, frozen at the time.
    weeks_post_op: Mapped[int | None] = mapped_column(Integer, nullable=True)

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="outcome_scores")

    __table_args__ = (
        # Every read of this table is "this patient's scores, oldest to newest".
        Index("ix_outcome_scores_patient_recorded", "patient_id", "recorded_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OutcomeScore {self.instrument} {self.interval_score:.0f}/100 {self.knee_side}>"


class ShareLink(Base):
    """
    A read-only window onto one patient's progress, openable without an account.

    Requiring a clinician to register is where this feature dies: they will click
    a link and they will not sign up for an app their patient uses. So the link
    itself is the credential.

    Which makes revocation the thing that matters most. A signed token carrying
    an expiry would be simpler and completely stateless — and impossible to take
    back before it expires. Sharing health data you cannot un-share is not a
    feature worth having, so the tokens live here instead.

    Only a hash of the token is stored. A leaked database then yields no working
    links, exactly as it yields no working passwords.
    """

    __tablename__ = "share_links"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )

    # SHA-256 of the token. A fast hash, deliberately: these are 256 bits of
    # randomness, so there is no dictionary to slow an attacker down through,
    # and the lookup has to be a single indexed hit on every page view.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # The patient's own note about who it went to — "Dr Okonkwo, 14 Aug".
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Shown back to the patient. Someone who shares a link deserves to know
    # whether it was ever opened, and how often.
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="share_links")

    __table_args__ = (
        Index("ix_share_links_patient", "patient_id", "created_at"),
    )

    @property
    def is_active(self) -> bool:
        if self.revoked_at is not None:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:          # SQLite hands back naive datetimes
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > utcnow()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ShareLink {self.id} active={self.is_active}>"


class Clinician(Base):
    """
    A physiotherapist or surgeon.

    A separate table rather than a role column on Patient, because the two are
    genuinely different things: a clinician has a caseload, not a knee. Giving
    them a row in `patients` would mean a surgery date, a prescription history
    and an exercise log that can never be anything but null, and a foreign key
    from Prescription that could point at someone who never had one.

    The password handling is shared with Patient through auth.py — that logic
    has one implementation, and this table does not fork it.
    """

    __tablename__ = "clinicians"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # Free text on purpose. Registration bodies and title conventions differ by
    # country, and a dropdown of guesses would be wrong more often than useful.
    registration: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


    # ── Session and credential control ───────────────────────────────────────
    # Every token carries the version this counter stood at when it was issued,
    # and a token whose version no longer matches is refused. Bumping it is what
    # makes "sign out everywhere" and "changing your password ends other
    # sessions" possible at all: a JWT is otherwise valid until it expires, and
    # there was no way to take one back.
    #
    # A counter rather than a cutoff timestamp, because `iat` is only accurate
    # to the second: a token issued and revoked inside the same second would
    # have survived a timestamp comparison. A counter has no clock in it, so
    # signing out always takes effect immediately.
    #
    # And a counter rather than a table of revoked tokens: one column, nothing
    # to sweep up later, and it revokes at the only granularity that is
    # actually useful after a password change — all of them.
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # A pending password reset. Only the hash is kept, for the same reason the
    # password is: this column is a credential while it lives, and a readable
    # one in a stolen backup would be an account takeover.
    reset_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reset_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    care_links: Mapped[list["CareLink"]] = relationship(
        back_populates="clinician",
        cascade="all, delete-orphan",
        order_by="CareLink.created_at.desc()",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Clinician {self.email}>"


class CareLink(Base):
    """
    One clinician looking after one patient — and the invite that created it.

    Invite and link are the same row at two points in its life, rather than two
    tables. A code that was never redeemed is simply a link with no patient on
    it yet, which is exactly what it is.

    The direction matters: the clinician issues a code, the patient redeems it.
    Redeeming is the patient's act, and that act is the consent. A clinician
    cannot attach themselves to someone's record without the patient doing
    something.

    Either side can end it. A patient must be able to stop sharing; a clinician
    must be able to discharge. `revoked_by` records which, because "my
    physiotherapist discharged me" and "I withdrew access" are different events
    and a caseload that cannot tell them apart is misleading.
    """

    __tablename__ = "care_links"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    clinician_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("clinicians.id", ondelete="CASCADE"), nullable=False
    )
    # Null until the code is redeemed.
    patient_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("patients.id", ondelete="CASCADE"), nullable=True
    )

    # Hashed, like a share token. Short enough to read down a phone line, so it
    # carries far less entropy than a share link — which is why it expires
    # quickly and stops working the moment it is used.
    invite_code_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # Shown back to the clinician so they can tell two pending invites apart
    # without storing the code itself.
    invite_hint: Mapped[str] = mapped_column(String(16), nullable=False)
    patient_label: Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by: Mapped[str | None] = mapped_column(String(16), nullable=True)   # 'patient' | 'clinician'

    clinician: Mapped[Clinician] = relationship(back_populates="care_links")
    patient: Mapped[Patient | None] = relationship(back_populates="care_links")

    __table_args__ = (
        # The caseload query, and the patient's "who can see this" list.
        Index("ix_care_links_clinician", "clinician_id", "accepted_at"),
        Index("ix_care_links_patient", "patient_id"),
    )

    @property
    def is_active(self) -> bool:
        """Redeemed, and not since ended by either side."""
        return self.patient_id is not None and self.revoked_at is None

    @property
    def is_pending(self) -> bool:
        """Issued, unredeemed, and still in date."""
        if self.patient_id is not None or self.revoked_at is not None:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:          # SQLite hands back naive datetimes
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > utcnow()


class PrescriptionAudit(Base):
    """
    Every decision that set or changed a movement ceiling, and who made it.

    The safe angle is the one number in this system that can hurt someone. It is
    set by a model reading an X-ray, and it can be overridden by a clinician who
    knows things the X-ray does not — they operated on the knee. Both are
    legitimate. Neither should be silent.

    So every value the patient is asked to keep their knee under has a row here
    saying where it came from, what it replaced, and — when a human raised it
    past what the model allowed — why. A clinician's judgement wins. It does not
    get to be invisible.
    """

    __tablename__ = "prescription_audit"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    prescription_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("prescriptions.id", ondelete="CASCADE"), nullable=False
    )

    # Null for the prescription-wide ceiling; otherwise the exercise it applies to.
    exercise_name: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # 'ceiling' | 'angle_limit' | 'removed' | 'added' | 'reps' | 'sets' | 'hold'
    field: Mapped[str] = mapped_column(String(24), nullable=False)
    previous_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    new_value: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 'model' when the checkpoint set it, otherwise the clinician's id.
    actor: Mapped[str] = mapped_column(String(32), nullable=False)
    clinician_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("clinicians.id", ondelete="SET NULL"), nullable=True
    )

    # Required by the API whenever a change loosens a restriction. A tightening
    # needs no justification; a loosening is the one that can hurt.
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    prescription: Mapped[Prescription] = relationship(back_populates="audit")

    __table_args__ = (
        Index("ix_audit_prescription", "prescription_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PrescriptionAudit {self.field} {self.previous_value}->{self.new_value} by {self.actor}>"
