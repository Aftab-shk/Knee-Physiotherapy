"""
triage.py — the signals worth interrupting someone about.

These are **triage heuristics, not clinical criteria**. Every threshold below is
a judgement about when a human should look, not a diagnosis. They exist to move
a patient up a clinician's list, and to tell a patient to stop and make contact —
nothing here decides anything on its own.

That distinction shapes the design:

  * Nothing fires from a single reading. One sore session after a hard week is
    ordinary; three in a row is a pattern.
  * Every flag carries the numbers it was raised on, so the clinician can
    disagree with it in one glance rather than going digging.
  * Angles come only from verified camera views. An unverified one reads far too
    low, and a flag saying "range of motion is collapsing" on the strength of a
    badly-placed webcam would be worse than no flag at all — it would train
    clinicians to ignore the ones that matter.

Kept torch-free and free of any database import so the rules can be tested
directly, on plain data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Each one is a place where a human should look, chosen to be quiet enough that
# a flag still means something. An alerting system nobody trusts is worse than
# none, because it is the one that gets muted.

RECENT_DAYS = 7          # "lately"
BASELINE_DAYS = 28       # what "lately" is compared against

# Pain, on the 0-10 Numeric Pain Rating Scale.
PAIN_SEVERE = 8          # a single reading this high is worth a call
PAIN_RISE_PER_SESSION = 2.0   # mean (after - before) across recent sessions
PAIN_RISE_MIN_SESSIONS = 3    # …over at least this many, so one bad day is not a pattern

# Range of motion, in degrees of knee flexion.
ROM_DROP_DEG = 10        # below the earlier best; roughly twice the measurement noise
ROM_MIN_SESSIONS = 3     # baseline sessions needed before a drop means anything

# Exceeding the safe ceiling.
BREACH_COUNT = 5         # separate events in the recent window
BREACH_SECONDS = 30.0    # …or this long past the limit in total

# Stopping.
LAPSE_DAYS = 7               # nothing recorded for this long
LAPSE_PRIOR_SESSIONS = 3     # …after having been this active beforehand

# A draft nobody has looked at.
UNREVIEWED_DAYS = 3


URGENT = "urgent"
WARNING = "warning"
INFO = "info"

_SEVERITY_ORDER = {INFO: 0, WARNING: 1, URGENT: 2}


@dataclass
class Flag:
    code:     str
    severity: str
    # Written for the clinician scanning a caseload.
    summary:  str
    # Written for the patient, in the second person. None where the finding is
    # about the clinician's own workflow rather than the patient's body.
    patient_message: Optional[str] = None
    evidence: dict = field(default_factory=dict)


def _utc(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat those as the UTC they were stored as."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _peak(session) -> Optional[float]:
    """Furthest flexion in a session, or None if nothing usable was measured."""
    if not session.view_verified:
        return None
    return max((s.peak_flexion_deg for s in session.sets), default=None)


def evaluate(sessions, prescriptions=(), now: Optional[datetime] = None) -> list[Flag]:
    """
    Assess one patient. `sessions` and `prescriptions` are ORM rows or anything
    with the same attributes; nothing here touches a database.

    Returned worst-first, so a caseload can sort on the head of the list.
    """
    now = now or datetime.now(timezone.utc)
    recent_from = now - timedelta(days=RECENT_DAYS)
    baseline_from = now - timedelta(days=BASELINE_DAYS)

    ordered = sorted(sessions, key=lambda s: _utc(s.started_at))
    recent = [s for s in ordered if _utc(s.started_at) >= recent_from]
    baseline = [s for s in ordered if baseline_from <= _utc(s.started_at) < recent_from]

    flags: list[Flag] = []

    # ── Severe pain ──────────────────────────────────────────────────────────
    worst = [s for s in recent if s.pain_after is not None and s.pain_after >= PAIN_SEVERE]
    if worst:
        peak_pain = max(s.pain_after for s in worst)
        flags.append(Flag(
            code="pain_severe", severity=URGENT,
            summary=f"Reported pain {peak_pain}/10 after exercising in the last {RECENT_DAYS} days.",
            patient_message=(
                f"You rated your pain {peak_pain} out of 10 after exercising. That is high enough "
                "to stop and speak to your physiotherapist before your next session."
            ),
            evidence={"max_pain_after": peak_pain, "sessions": len(worst)},
        ))

    # ── Pain climbing across sessions ────────────────────────────────────────
    # The before/after pair is the point: rehab that hurts during and settles
    # afterwards is working. Consistently ending worse than it started is not.
    paired = [s for s in ordered if s.pain_before is not None and s.pain_after is not None]
    considered = paired[-PAIN_RISE_MIN_SESSIONS:]
    if len(considered) >= PAIN_RISE_MIN_SESSIONS:
        change = sum(s.pain_after - s.pain_before for s in considered) / len(considered)
        if change >= PAIN_RISE_PER_SESSION:
            flags.append(Flag(
                code="pain_rising", severity=WARNING,
                summary=(
                    f"Sessions are ending {change:.1f} points more painful than they start, "
                    f"across the last {len(considered)}."
                ),
                patient_message=(
                    "Your recent sessions have been leaving your knee more painful than when you "
                    "started. Mention this to your physiotherapist — the programme may need easing."
                ),
                evidence={"mean_change": round(change, 1), "sessions": len(considered)},
            ))

    # ── Range of motion going backwards ──────────────────────────────────────
    baseline_peaks = [p for p in (_peak(s) for s in baseline) if p is not None]
    recent_peaks = [p for p in (_peak(s) for s in recent) if p is not None]
    if len(baseline_peaks) >= ROM_MIN_SESSIONS and recent_peaks:
        was, now_best = max(baseline_peaks), max(recent_peaks)
        if was - now_best >= ROM_DROP_DEG:
            flags.append(Flag(
                code="rom_regression", severity=URGENT,
                summary=f"Knee flexion down from {was:.0f}° to {now_best:.0f}° in the last {RECENT_DAYS} days.",
                patient_message=(
                    f"Your knee has been bending less far than it was — about {was - now_best:.0f}° less. "
                    "That is worth telling your physiotherapist about."
                ),
                evidence={"baseline_deg": round(was, 1), "recent_deg": round(now_best, 1),
                          "drop_deg": round(was - now_best, 1)},
            ))

    # ── Going past the safe ceiling, repeatedly ──────────────────────────────
    breach_count = sum(st.breach_count for s in recent for st in s.sets)
    breach_seconds = sum(st.breach_seconds for s in recent for st in s.sets)
    if breach_count >= BREACH_COUNT or breach_seconds >= BREACH_SECONDS:
        flags.append(Flag(
            code="repeated_breaches", severity=WARNING,
            summary=(
                f"Went past the safe angle {breach_count} times "
                f"({breach_seconds:.0f}s total) in the last {RECENT_DAYS} days."
            ),
            patient_message=(
                "You have been bending past your safe limit fairly often. If the limit feels wrong, "
                "ask your physiotherapist to review it rather than working through the alarm."
            ),
            evidence={"breach_count": breach_count, "breach_seconds": round(breach_seconds, 1)},
        ))

    # ── Stopped ──────────────────────────────────────────────────────────────
    # Only for someone who was going. A patient who never started is not a
    # deterioration, and flagging them would bury the ones who are slipping.
    if not recent and len(baseline) >= LAPSE_PRIOR_SESSIONS:
        last = _utc(ordered[-1].started_at)
        days = (now - last).days
        flags.append(Flag(
            code="stopped_exercising", severity=WARNING,
            summary=f"No sessions for {days} days, after {len(baseline)} in the weeks before.",
            patient_message=(
                "You have not recorded a session in a while. If something is stopping you — pain, "
                "time, or the exercises not feeling right — your physiotherapist can help."
            ),
            evidence={"days_since_last": days, "prior_sessions": len(baseline)},
        ))

    # ── A draft nobody has looked at ─────────────────────────────────────────
    # About the clinician's workload, not the patient's knee, so it carries no
    # patient message: telling someone their physiotherapist has not read their
    # notes yet is alarming and not theirs to act on.
    stale = [
        p for p in prescriptions
        if p.status == "draft" and (now - _utc(p.created_at)).days >= UNREVIEWED_DAYS
    ]
    if stale:
        oldest = max((now - _utc(p.created_at)).days for p in stale)
        flags.append(Flag(
            code="unreviewed_prescription", severity=INFO,
            summary=(
                f"{len(stale)} prescription{'s' if len(stale) != 1 else ''} awaiting review, "
                f"oldest {oldest} days."
            ),
            evidence={"count": len(stale), "oldest_days": oldest},
        ))

    flags.sort(key=lambda f: -_SEVERITY_ORDER[f.severity])
    return flags


def worst_severity(flags) -> Optional[str]:
    """The highest severity present, or None."""
    if not flags:
        return None
    return max((f.severity for f in flags), key=lambda s: _SEVERITY_ORDER[s])
