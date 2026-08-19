"""
KL grade → clinical values.

Single source of truth, deliberately free of any torch import so both the
serving path (model/inference.py, which needs torch) and the clinical path
(clinical_logic.py, which must not pay for it) can read the same numbers.

These tables previously existed as two independent copies with a comment
explaining that duplication avoided importing torch for two dicts. The reason
was sound; the remedy was not. They set how far a recovering knee is allowed to
bend, and two copies of a safety table drift silently.

Kellgren–Lawrence grading:
  0  Normal
  1  Doubtful     — possible osteophytes, no joint-space narrowing
  2  Minimal      — definite osteophytes, possible narrowing
  3  Moderate     — multiple osteophytes, definite narrowing, some sclerosis
  4  Severe       — large osteophytes, marked narrowing, bone-on-bone changes
"""

from types import MappingProxyType

# Joint health score shown to the patient (0–100).
KL_HEALTH_SCORE = MappingProxyType({0: 95, 1: 80, 2: 60, 3: 35, 4: 15})

# Safe flexion ceiling in degrees. Every exercise's angle limit is capped by
# this, and exercises whose starting position exceeds it are withheld entirely
# (see clinical_logic._cap_exercises).
KL_MAX_ANGLE = MappingProxyType({0: 120, 1: 120, 2: 90, 3: 60, 4: 45})

KL_DESCRIPTIONS = MappingProxyType({
    0: "Normal — no radiographic features of osteoarthritis",
    1: "Doubtful — possible osteophytes, no joint-space narrowing",
    2: "Minimal — definite osteophytes, possible narrowing",
    3: "Moderate — multiple osteophytes, definite narrowing, some sclerosis",
    4: "Severe — large osteophytes, significant narrowing, bone-on-bone changes",
})

VALID_GRADES = frozenset(KL_MAX_ANGLE)

# Mappings are read-only so a caller cannot mutate a safety table in place and
# have it silently affect every other consumer in the process.
assert VALID_GRADES == set(KL_HEALTH_SCORE) == set(KL_DESCRIPTIONS), (
    "KL tables must cover exactly the same grades"
)
