from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


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


class Exercise(BaseModel):
    name:         str
    description:  str
    target_reps:  int
    target_sets:  int
    angle_limit:  int           = Field(..., description="Effective angle limit in degrees (capped by X-ray severity)")
    hold_seconds: Optional[int] = Field(None, description="Hold duration in seconds for isometric exercises")
    instructions: List[str]
    cautions:     Optional[str] = None
    angle_capped: bool          = Field(False, description="True if X-ray severity reduced the standard protocol angle")


class AnalyseXrayResponse(BaseModel):
    kl_grade:          int            = Field(..., ge=0, le=4,     description="KL Grade 0–4")
    health_score:      int            = Field(..., ge=0, le=100,   description="Joint health score 0–100")
    max_angle:         int            = Field(...,                  description="Safe flexion ceiling in degrees")
    confidence:        float          = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")
    knee_side:         str
    surgery_type:      str
    weeks_post_op:     Optional[int]  = None
    rehab_phase:       str
    rehab_phase_label: str
    rehab_phase_goal:  str
    exercise_list:     List[Exercise]
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
    disclaimer:        str


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    demo_mode:     bool