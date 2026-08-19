"""
Exercise protocol database.

Structure:
  PROTOCOLS[surgery_type]["phases"] = list of phase dicts  (for surgical cases)
  PROTOCOLS["none"]["oa_levels"]    = list of OA-severity dicts (for non-surgical)

Each exercise dict:
  name, description, target_reps, target_sets,
  protocol_angle_limit (degrees), hold_seconds (optional),
  tracked_joint, hold_target, start_angle, hold_angle,
  instructions (list), cautions (optional str)

Tracker fields
--------------
tracked_joint  Which joint drives rep/hold counting in the webcam tracker:
               "knee" for every exercise except Ankle Pumps, which is an ankle
               movement. Note the *safety* check always watches the knee against
               angle_limit regardless of this field — the ceiling is a knee
               flexion ceiling.

hold_target    Only meaningful when hold_seconds is set. Says where the hold
               position sits relative to angle_limit:
                 "straight" — hold at or below the limit (isometrics with the
                              knee locked, extension end-range, single-leg
                              balance). Quad Sets holds at ~0°, limit 5°.
                 "flexed"   — hold near the limit (flexion end-range, e.g.
                              Seated Knee Bend holds at ~90°, limit 90°).
               None for rep-counted exercises.

Without hold_target the tracker cannot tell a straight-leg hold from a bent-leg
hold, and defaulted to "bent past 15°" — which made every isometric with a limit
under 15° impossible to complete without triggering the safety alarm.

start_angle    The unavoidable passive knee flexion the exercise begins from,
               for exercises performed seated or on equipment. Sitting in a
               chair puts the knee at ~90° before the patient does anything;
               a stationary bike bottoms out at ~90°; a leg press starts at
               ~90°. None when the exercise starts from extension.

               This is a SCREENING field, not an exemption. If start_angle
               exceeds the patient's X-ray-derived ceiling, they cannot get
               into the starting position safely, so the exercise is excluded
               from their prescription rather than prescribed-and-alarmed-at.
               See clinical_logic._cap_exercises.

hold_angle     The angle to reach and hold, when it differs from angle_limit.
               Only Seated Knee Extension needs it: performed seated, the knee
               sits at 90° (its true ceiling) while the goal is to EXTEND to
               45° and hold there. Without this the two numbers collapse and
               the hold either never registers or never ends.
               None → the tracker derives the hold position from angle_limit.

The distinction that matters: angle_limit is a safety ceiling (never exceed),
hold_angle is a performance target (try to reach). Seated Knee Extension was
originally filed with 45 as its angle_limit, which read as "never bend past 45°"
for an exercise you perform sitting at 90°.

Sources:
  ACL  — MassGeneral Hospital ACL Reconstruction Protocol
  TKR  — Brigham & Women's / OrthoInfo AAOS TKR Protocol
  Meniscus — OHSU Meniscus Repair Protocol
  Arthroscopy — Standard post-arthroscopy physiotherapy guidelines
  OA   — American College of Rheumatology OA Exercise Guidelines
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Shared exercise building blocks (referenced by multiple protocols)
# ---------------------------------------------------------------------------

_ANKLE_PUMPS = {
    "name": "Ankle Pumps",
    "description": "Rhythmic ankle flexion/extension to promote circulation and prevent DVT.",
    "target_reps": 20,
    "target_sets": 3,
    "protocol_angle_limit": 10,
    "hold_seconds": None,
    "tracked_joint": "ankle",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Lie flat or sit with legs extended.",
        "Flex your foot toward you, then point it away — one full cycle is one rep.",
        "Move slowly and deliberately. Perform every 1–2 hours while awake.",
    ],
    "cautions": None,
}

_QUAD_SETS = {
    "name": "Quad Sets (Isometric)",
    "description": "Isometric quadriceps activation with the knee fully straight.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 5,
    "hold_seconds": 10,
    "tracked_joint": "knee",
    "hold_target": "straight",
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Lie flat with knee as straight as possible.",
        "Tighten the thigh muscle firmly — you should see the kneecap draw upward.",
        "Hold 10 s, then fully relax. Do not hold your breath.",
    ],
    "cautions": None,
}

_SLR = {
    "name": "Straight Leg Raise",
    "description": "Quad and hip-flexor strengthening with zero knee movement.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 5,
    "hold_seconds": 2,
    "tracked_joint": "knee",
    "hold_target": "straight",
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Lie flat; tighten the quad to lock the knee fully straight.",
        "Raise the leg to the height of the opposite (bent) knee.",
        "Hold 2 s at the top, then lower slowly — do not let it drop.",
    ],
    "cautions": "Do not perform if you cannot hold the knee fully straight.",
}

_HEEL_SLIDES_90 = {
    "name": "Heel Slides",
    "description": "Active knee flexion by sliding the heel toward the buttocks.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 90,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Lie on your back with legs straight.",
        "Slowly slide the heel toward your buttocks as far as comfortable.",
        "Hold briefly at end-range, then slide back to start.",
    ],
    "cautions": "Do not force past comfortable range.",
}

_PATELLAR_MOB = {
    "name": "Patellar Mobilisation",
    "description": "Manual gliding of the kneecap to prevent scar-tissue adhesion.",
    "target_reps": 10,
    "target_sets": 2,
    "protocol_angle_limit": 5,
    "hold_seconds": 5,
    "tracked_joint": "knee",
    "hold_target": "straight",
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Sit with knee fully straight and relaxed.",
        "Use fingertips to gently push the kneecap up, down, and side-to-side.",
        "Hold each direction 5 s. Perform gently — never force.",
    ],
    "cautions": "Stop if sharp pain occurs beneath the kneecap.",
}

_MINI_SQUAT_60 = {
    "name": "Mini Squat",
    "description": "Partial squat to 45–60° for safe quad loading.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 60,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Stand feet shoulder-width apart; hold a support if needed.",
        "Bend knees slowly to 45–60° keeping knees behind toes.",
        "Hold 2 s at the bottom, then press through heels to stand.",
    ],
    "cautions": "Do not squat deeper than 60° in this phase.",
}

_STATIONARY_BIKE_LOW = {
    "name": "Stationary Bike (Low Resistance)",
    "description": "Low-load ROM and cardiovascular exercise — preferred for early rehab.",
    "target_reps": 1,
    "target_sets": 1,
    "protocol_angle_limit": 100,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": 90,
    "hold_angle": None,
    "instructions": [
        "Set seat so knee bends to ~90° at the lowest pedal position.",
        "Start with half-revolutions until full rotation is comfortable.",
        "Progress to 15–20 min at low resistance.",
    ],
    "cautions": None,
}

_STEP_UP_LOW = {
    "name": "Step Up (4–6 inch step)",
    "description": "Functional stair training starting with a low step.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 60,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Stand at the base of a low step (4–6 in / 10–15 cm) with a handrail.",
        "Step up with the surgical leg; bring the other up to meet it.",
        "Step down with the surgical leg last.",
    ],
    "cautions": "Keep handrail within reach at all times.",
}

_HAMSTRING_CURL = {
    "name": "Prone Hamstring Curl",
    "description": "Isolated hamstring strengthening lying face-down.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 90,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Lie face-down with legs straight.",
        "Slowly bend the surgical knee toward the buttocks — stop at 90° or discomfort.",
        "Lower slowly under control.",
    ],
    "cautions": None,
}

_FULL_SQUAT = {
    "name": "Full Squat (Progressed)",
    "description": "Progressive depth squat toward full comfortable range.",
    "target_reps": 12,
    "target_sets": 3,
    "protocol_angle_limit": 120,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Feet shoulder-width apart, toes slightly out.",
        "Lower slowly, adding depth each week as tolerated.",
        "Keep chest tall, drive through heels to return.",
    ],
    "cautions": "Increase depth gradually — do not chase depth through pain.",
}

_STATIONARY_BIKE_MOD = {
    "name": "Stationary Bike (Moderate Resistance)",
    "description": "Cardiovascular conditioning and progressive knee strengthening.",
    "target_reps": 1,
    "target_sets": 1,
    "protocol_angle_limit": 110,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": 90,
    "hold_angle": None,
    "instructions": [
        "Cycle 20–30 min at moderate resistance.",
        "Maintain a comfortable cadence of 60–80 RPM.",
    ],
    "cautions": None,
}

_FORWARD_LUNGE = {
    "name": "Forward Lunge",
    "description": "Single-leg eccentric quad and glute strengthening.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 90,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Step forward with the surgical leg; lower the back knee toward the floor.",
        "Keep the front shin vertical — knee must not pass the toes.",
        "Push back through the front heel to return.",
    ],
    "cautions": "Use a support or wall during early attempts.",
}

_SINGLE_LEG_BALANCE = {
    "name": "Single-Leg Balance",
    "description": "Proprioception and neuromuscular control on the surgical leg.",
    "target_reps": 5,
    "target_sets": 3,
    "protocol_angle_limit": 20,
    "hold_seconds": 30,
    "tracked_joint": "knee",
    "hold_target": "straight",
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Stand on the surgical leg only, near a wall for safety.",
        "Hold 30 s without excessive trunk sway.",
        "Progress to eyes closed once stable on eyes-open.",
    ],
    "cautions": "Always have a wall or chair within touching distance.",
}

_LEG_PRESS = {
    "name": "Leg Press",
    "description": "Machine-based quad and glute strengthening through full ROM.",
    "target_reps": 12,
    "target_sets": 4,
    "protocol_angle_limit": 120,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": 90,
    "hold_angle": None,
    "instructions": [
        "Set machine so the knee starts at ~90°.",
        "Press through both heels to full extension.",
        "Return slowly — do not let the weight stack drop.",
    ],
    "cautions": None,
}

_RUNNING_PROGRAM = {
    "name": "Walk-to-Run Programme",
    "description": "Graduated return to running on a flat surface.",
    "target_reps": 1,
    "target_sets": 1,
    "protocol_angle_limit": 120,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Begin with 1 min walk / 1 min jog intervals for 20 min.",
        "Increase jogging intervals each week if no swelling follows.",
        "Run on flat surfaces only in the first 2 weeks.",
    ],
    "cautions": "Do not start running without explicit physiotherapist clearance.",
}

_SEATED_BEND = {
    "name": "Seated Knee Bend",
    "description": "Gravity-assisted flexion in a chair to increase ROM.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 90,
    "hold_seconds": 5,
    "tracked_joint": "knee",
    "hold_target": "flexed",
    "start_angle": 90,
    "hold_angle": None,
    "instructions": [
        "Sit in a chair and slide the foot back under it using gravity.",
        "Use the other foot to gently assist if needed.",
        "Hold 5 s at end-range, then slide forward to release.",
    ],
    "cautions": "Stop at sharp pain — never force the bend.",
}

_STAIR_DESCENT = {
    "name": "Reciprocal Stair Descent",
    "description": "Step-over-step descent — a key TKR functional milestone.",
    "target_reps": 10,
    "target_sets": 2,
    "protocol_angle_limit": 100,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": 80,
    "hold_angle": None,
    "instructions": [
        "Hold the handrail. Descend step-over-step (not step-to-step).",
        "Control the surgical knee as you lower the opposite foot to the next step.",
        "This requires 80–100° of controlled, weight-bearing flexion.",
    ],
    "cautions": "Attempt only when confident on flat surfaces. Always use a rail.",
}

_STANDING_KNEE_BEND = {
    "name": "Standing Knee Bend",
    "description": "Standing active flexion to build upright ROM.",
    "target_reps": 10,
    "target_sets": 3,
    "protocol_angle_limit": 90,
    "hold_seconds": 5,
    "tracked_joint": "knee",
    "hold_target": "flexed",
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Stand facing a wall with fingertips touching for balance.",
        "Bend the surgical knee, lifting the foot behind you.",
        "Hold at the top for 5 s, then lower slowly.",
    ],
    "cautions": None,
}

_WALKING_PROGRAM = {
    "name": "Walking Programme",
    "description": "Progressive daily walking for cardiovascular and functional recovery.",
    "target_reps": 1,
    "target_sets": 1,
    "protocol_angle_limit": 120,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Walk 20–40 min daily on flat, even surfaces.",
        "Aim for a normal heel-to-toe gait pattern.",
        "Gradually introduce gentle inclines and uneven terrain from week 16+.",
    ],
    "cautions": None,
}

_SEATED_EXT_GRAVITY = {
    "name": "Seated Knee Extension (Gravity Only)",
    "description": "Gentle active extension with no added resistance.",
    "target_reps": 10,
    "target_sets": 3,
    # Performed seated, so the knee passively sits at ~90 deg: that is the real
    # flexion ceiling. Extending to 45 deg is the goal, tracked via hold_angle.
    "protocol_angle_limit": 90,
    "hold_seconds": 5,
    "tracked_joint": "knee",
    "hold_target": "straight",
    "start_angle": 90,
    "hold_angle": 45,
    "instructions": [
        "Sit in a chair.",
        "Slowly extend the knee as far as comfortable — use gravity only.",
        "Hold 5 s at end-range, then lower slowly.",
    ],
    "cautions": "Do not add ankle weights in this phase.",
}

_POOL_WALKING = {
    "name": "Pool Walking (if available)",
    "description": "Buoyancy-assisted walking — ideal low-load option for severe OA.",
    "target_reps": 1,
    "target_sets": 1,
    "protocol_angle_limit": 60,
    "hold_seconds": None,
    "tracked_joint": "knee",
    "hold_target": None,
    "start_angle": None,
    "hold_angle": None,
    "instructions": [
        "Walk in chest-deep water for 15–20 min at a comfortable pace.",
        "Water reduces effective body weight by ~75%, protecting the joint.",
    ],
    "cautions": "Requires pool access. Not suitable immediately post-surgery (wound risk).",
}


# ---------------------------------------------------------------------------
# Protocol database
# ---------------------------------------------------------------------------

PROTOCOLS = {

    # ── ACL Reconstruction ─────────────────────────────────────────────────
    "acl": {
        "phases": [
            {
                "key":   "phase_1",
                "weeks": (0, 2),
                "label": "Phase I — Immediate Post-Op (Weeks 0–2)",
                "goal":  "Control swelling, restore full extension, activate quadriceps.",
                "exercises": [
                    _ANKLE_PUMPS,
                    _QUAD_SETS,
                    _HEEL_SLIDES_90,
                    _SLR,
                    _PATELLAR_MOB,
                ],
            },
            {
                "key":   "phase_2",
                "weeks": (3, 6),
                "label": "Phase II — Early Strengthening (Weeks 3–6)",
                "goal":  "Restore 0–90° ROM, begin weight-bearing strengthening.",
                "exercises": [
                    _MINI_SQUAT_60,
                    _STATIONARY_BIKE_LOW,
                    _STEP_UP_LOW,
                    _HAMSTRING_CURL,
                ],
            },
            {
                "key":   "phase_3",
                "weeks": (7, 12),
                "label": "Phase III — Progressive Loading (Weeks 7–12)",
                "goal":  "Full ROM recovery, progressive strength, proprioception.",
                "exercises": [
                    _FULL_SQUAT,
                    _STATIONARY_BIKE_MOD,
                    _FORWARD_LUNGE,
                    _SINGLE_LEG_BALANCE,
                ],
            },
            {
                "key":   "phase_4",
                "weeks": (13, 9999),
                "label": "Phase IV — Return to Activity (Weeks 13+)",
                "goal":  "Sport-specific conditioning, power, full return to activity.",
                "exercises": [
                    _RUNNING_PROGRAM,
                    _LEG_PRESS,
                    _FORWARD_LUNGE,
                ],
            },
        ]
    },

    # ── Total Knee Replacement ─────────────────────────────────────────────
    "tkr": {
        "phases": [
            {
                "key":   "phase_1",
                "weeks": (0, 2),
                "label": "Phase I — Early Post-Op (Weeks 0–2)",
                "goal":  "Regain full extension, DVT prevention, quad re-activation.",
                "exercises": [
                    _ANKLE_PUMPS,
                    _QUAD_SETS,
                    _HEEL_SLIDES_90,
                    _SLR,
                    _SEATED_BEND,
                ],
            },
            {
                "key":   "phase_2",
                "weeks": (3, 6),
                "label": "Phase II — Progressive Mobility (Weeks 3–6)",
                "goal":  "Achieve 0–110° ROM, reduce gait-aid dependence.",
                "exercises": [
                    _STATIONARY_BIKE_LOW,
                    _MINI_SQUAT_60,
                    _STEP_UP_LOW,
                    _STANDING_KNEE_BEND,
                ],
            },
            {
                "key":   "phase_3",
                "weeks": (7, 12),
                "label": "Phase III — Strengthening (Weeks 7–12)",
                "goal":  "Full ROM, progressive strength, independent mobility.",
                "exercises": [
                    _FULL_SQUAT,
                    _STATIONARY_BIKE_MOD,
                    _STAIR_DESCENT,
                ],
            },
            {
                "key":   "phase_4",
                "weeks": (13, 9999),
                "label": "Phase IV — Return to Full Activity (Weeks 13+)",
                "goal":  "Full independence in daily and recreational activities.",
                "exercises": [
                    _LEG_PRESS,
                    _WALKING_PROGRAM,
                    _FULL_SQUAT,
                ],
            },
        ]
    },

    # ── Meniscus Repair ────────────────────────────────────────────────────
    "meniscus": {
        "phases": [
            {
                "key":   "phase_1",
                "weeks": (0, 4),
                "label": "Phase I — Protection (Weeks 0–4)",
                "goal":  "Protect repair, control inflammation, maintain quad strength.",
                "exercises": [
                    _ANKLE_PUMPS,
                    _QUAD_SETS,
                    _SLR,
                    {   # Strict 90° limit variant
                        **_HEEL_SLIDES_90,
                        "cautions": "STOP at 90° — do not exceed this. The repair is vulnerable to deeper flexion.",
                    },
                ],
            },
            {
                "key":   "phase_2",
                "weeks": (5, 8),
                "label": "Phase II — Progressive Weight Bearing (Weeks 5–8)",
                "goal":  "Full weight bearing, restore ROM beyond 90°, begin strengthening.",
                "exercises": [
                    _MINI_SQUAT_60,
                    _STATIONARY_BIKE_LOW,
                    _STEP_UP_LOW,
                ],
            },
            {
                "key":   "phase_3",
                "weeks": (9, 9999),
                "label": "Phase III — Strengthening & Return to Activity (Weeks 9+)",
                "goal":  "Full ROM, full strength, return to sport or work.",
                "exercises": [
                    _FULL_SQUAT,
                    _FORWARD_LUNGE,
                    _STATIONARY_BIKE_MOD,
                ],
            },
        ]
    },

    # ── Knee Arthroscopy ───────────────────────────────────────────────────
    "arthroscopy": {
        "phases": [
            {
                "key":   "phase_1",
                "weeks": (0, 1),
                "label": "Phase I — Acute Recovery (Week 0–1)",
                "goal":  "Reduce swelling, protect portal sites, restore quad activation.",
                "exercises": [
                    {**_ANKLE_PUMPS, "cautions": "Perform every hour to prevent DVT."},
                    _QUAD_SETS,
                ],
            },
            {
                "key":   "phase_2",
                "weeks": (2, 4),
                "label": "Phase II — Early Mobility (Weeks 2–4)",
                "goal":  "Restore full ROM, begin weight-bearing exercises.",
                "exercises": [
                    _HEEL_SLIDES_90,
                    _STATIONARY_BIKE_LOW,
                    _MINI_SQUAT_60,
                ],
            },
            {
                "key":   "phase_3",
                "weeks": (5, 9999),
                "label": "Phase III — Return to Activity (Weeks 5+)",
                "goal":  "Full ROM, full strength, return to sport or work.",
                "exercises": [
                    _FULL_SQUAT,
                    _FORWARD_LUNGE,
                    _RUNNING_PROGRAM,
                ],
            },
        ]
    },
}

# ── OA / No Surgery ────────────────────────────────────────────────────────
OA_LEVELS = [
    {
        "key":   "oa_mild",
        "kl_max": 1,
        "label": "OA Management — Mild (KL Grade 0–1)",
        "goal":  "Maintain strength and joint health, prevent deconditioning.",
        "exercises": [
            _QUAD_SETS,
            _SLR,
            _STATIONARY_BIKE_MOD,
            _FULL_SQUAT,
            _WALKING_PROGRAM,
        ],
    },
    {
        "key":   "oa_moderate",
        "kl_max": 2,
        "label": "OA Management — Moderate (KL Grade 2)",
        "goal":  "Pain-safe strengthening within moderate ROM limits.",
        "exercises": [
            _QUAD_SETS,
            _HEEL_SLIDES_90,
            _MINI_SQUAT_60,
            _STATIONARY_BIKE_LOW,
        ],
    },
    {
        "key":   "oa_severe",
        "kl_max": 4,
        "label": "OA Management — Severe (KL Grade 3–4)",
        "goal":  "Gentle ROM maintenance, pain reduction, avoid joint deterioration.",
        "exercises": [
            _ANKLE_PUMPS,
            _QUAD_SETS,
            _SLR,
            _SEATED_EXT_GRAVITY,
            _POOL_WALKING,
        ],
    },
]


# ---------------------------------------------------------------------------
# Phase resolver
# ---------------------------------------------------------------------------

def get_phase(
    surgery_type: str,
    weeks_post_op: Optional[int],
    kl_grade: int,
) -> dict:
    """
    Returns the appropriate phase dict with keys:
      key, label, goal, exercises (list of raw exercise dicts)

    For surgery_type == "none" or weeks_post_op is None with non-surgical intent,
    falls back to OA level based on kl_grade.
    """

    if surgery_type == "none" or weeks_post_op is None:
        # Non-surgical: pick least-severe OA level that fits the KL grade.
        # OA_LEVELS is ordered mild → moderate → severe, so the first match
        # gives the most appropriate (least restrictive appropriate) protocol.
        for level in OA_LEVELS:
            if kl_grade <= level["kl_max"]:
                return level
        return OA_LEVELS[-1]   # fallback: severe

    protocol = PROTOCOLS.get(surgery_type)
    if protocol is None:
        # Unknown surgery type: fall back to the most restrictive OA protocol.
        #
        # This was written as `for level in reversed(OA_LEVELS): if kl_grade <=
        # level["kl_max"]: return level`, which reads like a grade-dependent
        # search but is not — reversed() puts oa_severe first and its kl_max is
        # 4, so every grade matched on the first iteration. Worse, if nothing
        # had matched, control fell through to protocol["phases"] and raised
        # TypeError on None.
        #
        # Returning the most restrictive protocol for an input we do not
        # recognise is the right call; it is just stated directly now.
        # Unreachable through the API, where the SurgeryType enum rejects
        # unknown values first, but get_phase is also called from tests and
        # scripts.
        return OA_LEVELS[-1]

    for phase in protocol["phases"]:
        lo, hi = phase["weeks"]
        if lo <= weeks_post_op <= hi:
            return phase

    # weeks_post_op exceeds all defined phases — return last phase
    return protocol["phases"][-1]
