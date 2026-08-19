from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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