"""
summary_pdf.py — the one page a patient brings to an appointment.

The share link is the online version of this: a clinician opens a URL and sees
live charts. This is the version for the consultation where nobody opens a
laptop — printed, or held up on a phone with no signal.

Three things follow from that, and they shape everything below.

**It is one page, and the page is enforced.** A summary that runs to four pages
is not a summary, and a physiotherapist has about thirty seconds with it. So the
layout hands out vertical space from a cursor that can refuse, and anything that
does not fit is dropped *and named* in the notes at the foot. Silently spilling
the last two exercises off the bottom of a clinical document is the one failure
mode worth engineering against.

**It cannot ask a question.** On screen, an unverified camera angle can be a
tooltip and a flag can link to an explanation. On paper, every caveat has to be
printed or it does not exist. Hence the notes block, which is not decoration: it
carries the measurement limits, the count of excluded sessions, and whether a
clinician has actually approved the plan.

**It renders only what it is handed.** `render()` takes a `SummaryData` and has
no database session, no ORM row and no patient object — so it cannot reach for
an email address while rendering a share link that is supposed to show only a
display name. The caller decides what is disclosable; this file cannot exceed it.

Fonts
-----
The base-14 PDF fonts reach Latin-1 and no further, and the faces bundled with
reportlab turn out to cover no more than that either — no Greek, no Cyrillic,
certainly no Devanagari or CJK. A name outside Latin-1 therefore cannot be drawn
without a font the deployment provides:

    SUMMARY_PDF_FONT=/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf

The Dockerfile now installs Noto Sans and sets exactly that, so Latin Extended,
Greek, Cyrillic and Vietnamese names print on a deployed image with nobody
configuring anything. Run outside that image and the variable is still the way
in; leave it unset and you are back to Helvetica and Latin-1.

What Noto Sans does not reach is Devanagari, Arabic, Hebrew, Thai and CJK.
reportlab binds one file per face and has no fallback chain, so a name in one of
those scripts means pointing SUMMARY_PDF_FONT at the face that carries it — the
Noto package ships one per script, NotoSansDevanagari-Regular.ttf beside the
rest. Any character the active font cannot draw still becomes "?", with a line
printed on the sheet saying so and naming the variable. Quietly mangling
somebody's name on their own medical summary is not an option; saying plainly
that it happened is.

A bold face is picked up automatically when one sits beside the regular. See
_bold_candidates for why that is not simply a matter of appending "-Bold".

Kept free of FastAPI, SQLAlchemy and torch so the layout can be tested on plain
data.
"""

from __future__ import annotations

import io
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Sequence

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas

logger = logging.getLogger("physio-backend.summary_pdf")

# ---------------------------------------------------------------------------
# What the sheet is allowed to say
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RomRow:
    """One exercise's range of motion over the period."""

    exercise:      str
    best_deg:      float
    latest_deg:    float
    angle_limit:   int
    days_measured: int
    # (day, peak degrees, ceiling that applied) — charted for the first row only.
    points: Sequence[tuple[date, float, int]] = ()


@dataclass(frozen=True)
class FlagRow:
    severity: str      # urgent | warning | info
    summary:  str


@dataclass(frozen=True)
class OutcomeRow:
    instrument:  str          # display name, e.g. "KOOS-JR"
    knee_side:   str
    latest:      float
    baseline:    float
    band:        str
    change_note: str          # the instrument's own words about the change
    recorded_on: date


@dataclass(frozen=True)
class SummaryData:
    generated_at: datetime
    range_days:   int

    # None where the caller is not permitted to disclose one — a share link with
    # no display name set, for instance. Never fall back to an email here.
    patient_name: Optional[str] = None

    surgery_type:  Optional[str] = None
    surgery_date:  Optional[date] = None
    weeks_post_op: Optional[int] = None
    knee_side:     Optional[str] = None

    # The plan currently in force, and who stands behind it. A boolean rather
    # than a status string: which of several review states a prescription is in
    # is the API's vocabulary, and the only thing this page has to say is whether
    # a person has put their name to the limits printed above.
    ceiling_deg:   Optional[int] = None
    has_plan:      bool = False
    plan_reviewed: bool = False
    reviewed_by:   Optional[str] = None
    reviewed_at:   Optional[datetime] = None
    demo_mode:     bool = False

    sessions:            int = 0
    active_days:         int = 0
    current_streak_days: int = 0
    longest_streak_days: int = 0
    unverified_sessions: int = 0

    latest_pain_after: Optional[float] = None
    mean_pain_change:  Optional[float] = None

    rom:      Sequence[RomRow]  = field(default_factory=tuple)
    flags:    Sequence[FlagRow] = field(default_factory=tuple)
    outcomes: Sequence[OutcomeRow] = field(default_factory=tuple)

    # Printed under the scores when the instrument was not validated for this
    # operation. Supplied by the caller so the rule deciding it lives with the
    # instrument, not with the page layout.
    outcome_caveat: Optional[str] = None


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4
MARGIN   = 42
CONTENT  = PAGE_W - 2 * MARGIN
COL_GAP  = 20
LEFT_W   = CONTENT * 0.615          # chart and the exercise table
RIGHT_X  = MARGIN + LEFT_W + COL_GAP
RIGHT_W  = CONTENT - LEFT_W - COL_GAP

INK      = HexColor("#101A21")
INK_2    = HexColor("#35474F")
MUTED    = HexColor("#64777F")
LINE     = HexColor("#D9E2E6")
LINE_SOFT= HexColor("#EAF0F2")
ACCENT   = HexColor("#0B6C8C")
WARN     = HexColor("#B45309")
URGENT   = HexColor("#B3261E")
CHART_BG = HexColor("#F6F9FA")

SEVERITY_COLOUR = {"urgent": URGENT, "warning": WARN, "info": MUTED}
SEVERITY_MARK   = {"urgent": "!", "warning": "!", "info": "•"}

# Replacement for anything the active font cannot draw. Chosen to be obviously
# missing rather than plausibly correct.
MISSING_CHAR = "?"

FONT_ENV = "SUMMARY_PDF_FONT"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

_registered: dict[str, tuple[str, str]] = {}


def _bold_candidates(path: str) -> list[str]:
    """
    Where a bold face might sit beside `path`.

    A family that names its weight in the filename replaces that token rather
    than appending to it: NotoSans-Regular.ttf sits beside NotoSans-Bold.ttf and
    never beside NotoSans-Regular-Bold.ttf. Appending alone — which is all this
    used to do — finds arialbd.ttf and misses every Noto face, so the font a
    deployment ships would load and then draw the masthead, every stat and the
    patient's own name at regular weight, with nothing anywhere to say the sheet
    had quietly lost its typography.
    """
    stem = path[:-4] if path[-4:].lower() == ".ttf" else path
    candidates = [
        f"{stem[: -len(token)]}-Bold.ttf"
        for token in ("-Regular", "-regular")
        if stem.endswith(token)
    ]
    candidates += [f"{stem}{suffix}.ttf" for suffix in ("-Bold", "-bold", "Bold", "bd")]
    return [c for c in candidates if c != path]


def _register_fonts() -> tuple[str, str]:
    """
    Return (regular, bold) font names, honouring SUMMARY_PDF_FONT.

    A deployment serving names outside Latin-1 points that at a TTF. Anything
    unreadable falls back to Helvetica with a warning rather than failing the
    download — a summary with a mangled name still beats no summary at all, and
    the sheet says which happened.
    """
    path = os.getenv(FONT_ENV, "").strip()
    if not path:
        return "Helvetica", "Helvetica-Bold"

    if path in _registered:
        return _registered[path]

    try:
        pdfmetrics.registerFont(TTFont("SummaryFont", path))
        # One face used for both weights unless a bold sits beside it. Synthetic
        # emboldening is not worth a second guess at a filename.
        bold = "SummaryFont"
        for candidate in _bold_candidates(path):
            if os.path.exists(candidate):
                pdfmetrics.registerFont(TTFont("SummaryFont-Bold", candidate))
                bold = "SummaryFont-Bold"
                break
        _registered[path] = ("SummaryFont", bold)
    except Exception:
        logger.warning("%s=%s could not be loaded; falling back to Helvetica", FONT_ENV, path)
        _registered[path] = ("Helvetica", "Helvetica-Bold")

    return _registered[path]


def _can_draw(font: str, ch: str) -> bool:
    if ch in "\n\r\t":
        return False
    try:
        face = pdfmetrics.getFont(font).face
    except Exception:
        return True
    glyphs = getattr(face, "charToGlyph", None)
    if glyphs is not None:                       # an embedded TrueType face
        return glyphs.get(ord(ch)) is not None
    try:                                         # a base-14 face, WinAnsi
        ch.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


class _Text:
    """
    Everything drawn goes through here, so nothing reaches the page in a
    character the font has no glyph for — which reportlab would otherwise render
    as a blank, indistinguishable from a name that genuinely has a space in it.
    """

    def __init__(self, font: str):
        self.font = font
        self.dropped = False

    def __call__(self, value: object) -> str:
        text = "" if value is None else str(value)
        out = []
        for ch in text:
            if _can_draw(self.font, ch):
                out.append(ch)
            else:
                out.append(MISSING_CHAR)
                self.dropped = True
        return "".join(out)


# ---------------------------------------------------------------------------
# Vertical space, handed out by something that can say no
# ---------------------------------------------------------------------------


class _Cursor:
    """
    One page. `take(h)` returns the top of a block of that height, or None when
    the page is full — callers are expected to check and record the omission
    rather than draw past the footer.
    """

    def __init__(self, top: float, bottom: float):
        self.y = top
        self.bottom = bottom

    def room(self, height: float) -> bool:
        return self.y - height >= self.bottom

    def take(self, height: float) -> Optional[float]:
        if not self.room(height):
            return None
        self.y -= height
        return self.y


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def _fit(text: str, font: str, size: float, width: float) -> str:
    """Truncate to fit, with an ellipsis, so a long name never runs into a column."""
    if pdfmetrics.stringWidth(text, font, size) <= width:
        return text
    ellipsis = "…"
    while text and pdfmetrics.stringWidth(text + ellipsis, font, size) > width:
        text = text[:-1]
    return text + ellipsis


def _wrap(text: str, font: str, size: float, width: float, max_lines: int) -> list[str]:
    lines: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(trial, font, size) <= width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if lines and len(lines) == max_lines:
        lines[-1] = _fit(lines[-1], font, size, width)
    return lines


def _pretty_date(value: Optional[date]) -> str:
    if value is None:
        return "—"
    return f"{value.day} {value:%B %Y}"


def _degrees(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0f}°"


# ---------------------------------------------------------------------------
# The chart
# ---------------------------------------------------------------------------


def _draw_chart(c, tx, fonts, x: float, y: float, w: float, h: float, row: RomRow) -> None:
    """
    Peak flexion against the ceiling that applied on the day.

    The ceiling is drawn as its own stepped line rather than one flat rule: it
    moves when a clinician adjusts it, and a summary that flattened those steps
    would show a patient breaking a limit that had in fact been raised.
    """
    regular, _bold = fonts
    points = list(row.points)

    c.setFillColor(CHART_BG)
    c.setStrokeColor(LINE_SOFT)
    c.rect(x, y, w, h, stroke=1, fill=1)

    if not points:
        c.setFont(regular, 8)
        c.setFillColor(MUTED)
        c.drawCentredString(x + w / 2, y + h / 2 - 3, tx("No verified measurements in this period."))
        return

    pad_l, pad_r, pad_t, pad_b = 26, 8, 10, 14
    plot_x, plot_w = x + pad_l, w - pad_l - pad_r
    plot_y, plot_h = y + pad_b, h - pad_t - pad_b

    # Bounds are rounded outwards to a multiple of five, because they are drawn
    # as labels: "60°" reads as a scale, "58°" reads as a measurement that means
    # something.
    values = [p[1] for p in points] + [float(p[2]) for p in points]
    lo, hi = min(values), max(values)
    if hi - lo < 10:                       # a flat series still needs a visible band
        mid = (hi + lo) / 2
        lo, hi = mid - 5, mid + 5
    lo = max(0.0, 5 * math.floor((lo - 3) / 5))
    hi = 5 * math.ceil((hi + 3) / 5)

    ordinals = [p[0].toordinal() for p in points]
    span = max(ordinals) - min(ordinals)

    def px(i: int) -> float:
        if span == 0:
            return plot_x + plot_w / 2
        return plot_x + plot_w * (ordinals[i] - min(ordinals)) / span

    def py(v: float) -> float:
        return plot_y + plot_h * (v - lo) / (hi - lo)

    # Axis labels: the two values that bound the plot, and the two dates.
    c.setFont(regular, 7)
    c.setFillColor(MUTED)
    c.drawRightString(plot_x - 4, py(hi) - 2, f"{hi:.0f}°")
    c.drawRightString(plot_x - 4, py(lo) - 2, f"{lo:.0f}°")
    c.drawString(plot_x, y + 4, tx(f"{points[0][0]:%d %b}"))
    if span:
        c.drawRightString(plot_x + plot_w, y + 4, tx(f"{points[-1][0]:%d %b}"))

    # The ceiling, stepped.
    c.setStrokeColor(WARN)
    c.setLineWidth(0.9)
    c.setDash([2.5, 2], 0)
    path = c.beginPath()
    path.moveTo(px(0), py(points[0][2]))
    for i in range(1, len(points)):
        path.lineTo(px(i), py(points[i - 1][2]))
        path.lineTo(px(i), py(points[i][2]))
    path.lineTo(plot_x + plot_w, py(points[-1][2]))
    c.drawPath(path)
    c.setDash()

    # Measured peaks.
    c.setStrokeColor(ACCENT)
    c.setLineWidth(1.4)
    if len(points) > 1:
        line = c.beginPath()
        line.moveTo(px(0), py(points[0][1]))
        for i in range(1, len(points)):
            line.lineTo(px(i), py(points[i][1]))
        c.drawPath(line)
    c.setFillColor(ACCENT)
    for i, p in enumerate(points):
        c.circle(px(i), py(p[1]), 1.6, stroke=0, fill=1)

    c.setFont(regular, 6.5)
    c.setFillColor(WARN)
    c.drawString(plot_x + 2, y + h - pad_t + 1, tx("dashed: safe limit"))


# ---------------------------------------------------------------------------
# The notes at the foot
# ---------------------------------------------------------------------------
#
# Not decoration, and not a disclaimer either. On screen an unverified camera
# angle can be a tooltip and a score can carry a link explaining what it does not
# measure. On paper there is no hover and no second click, so anything a reader
# would need in order not to over-read a number has to be printed under it.

NOTE_LEADING = 8.6
NOTE_LINES_MAX = 3
NOTE_SLACK = 2 * NOTE_LEADING     # room for notes added during layout
FOOTER_MAX = 132                  # notes can inform the page, not consume it

MEASUREMENT_NOTE = (
    "Angles are measured from a webcam, not a goniometer; treat small changes as noise. "
    "This sheet summarises what the patient recorded and is not a diagnosis."
)


def _standing_notes(data: SummaryData) -> list[str]:
    """Everything known before the page is laid out."""
    notes: list[str] = []

    if data.has_plan and not data.plan_reviewed:
        notes.append(
            "This plan has not been reviewed by a clinician — the limits above are the "
            "model's own draft."
        )
    elif data.plan_reviewed:
        who  = f" by {data.reviewed_by}" if data.reviewed_by else ""
        when = f" on {_pretty_date(data.reviewed_at.date())}" if data.reviewed_at else ""
        notes.append(f"Plan reviewed{who}{when}.")

    if data.demo_mode:
        notes.append(
            "The grade behind these limits came from demonstration mode, not a trained model."
        )

    if data.outcomes and data.outcome_caveat:
        notes.append(data.outcome_caveat)

    if data.unverified_sessions:
        n = data.unverified_sessions
        notes.append(
            f"{n} session{'s' if n != 1 else ''} had an unverified camera angle. "
            "They count towards adherence but are excluded from every angle above, because a "
            "knee angle measured off the sagittal plane reads low."
        )

    notes.append(MEASUREMENT_NOTE)
    return notes


def _note_lines(notes: Sequence[str], font: str) -> list[str]:
    lines: list[str] = []
    for note in notes:
        lines.extend(_wrap(note, font, 7, CONTENT - 10, NOTE_LINES_MAX))
    return lines


def _footer_height(notes: Sequence[str], font: str) -> float:
    height = 14 + NOTE_LEADING * len(_note_lines(notes, font)) + NOTE_SLACK
    return min(height, FOOTER_MAX)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render(data: SummaryData) -> bytes:
    """Lay the sheet out and return the PDF bytes. Never raises on thin data."""
    regular, bold = _register_fonts()
    tx = _Text(regular)
    fonts = (regular, bold)

    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)

    name = data.patient_name or "Patient"
    c.setTitle(tx(f"Physiotherapy summary — {name}"))
    c.setAuthor(tx("Knee Physiotherapy"))
    c.setSubject(tx(f"Progress over the last {data.range_days} days"))
    c.setCreator("knee-physiotherapy")

    # Assembled before anything is drawn, because the footer's height is what
    # the body is allowed to fill and it depends on how much there is to say.
    # Two more notes can be appended during layout, when something is dropped
    # for want of room; NOTE_SLACK leaves them a line each.
    notes: list[str] = _standing_notes(data)

    # ── Masthead ─────────────────────────────────────────────────────────────
    y = PAGE_H - MARGIN

    c.setFont(bold, 8)
    c.setFillColor(ACCENT)
    c.drawString(MARGIN, y - 8, tx("PHYSIOTHERAPY PROGRESS SUMMARY"))

    y -= 30
    c.setFont(bold, 19)
    c.setFillColor(INK)
    c.drawString(MARGIN, y, tx(_fit(name, bold, 19, CONTENT * 0.7)))

    c.setFont(regular, 8.5)
    c.setFillColor(MUTED)
    c.drawRightString(
        PAGE_W - MARGIN, y + 2,
        tx(f"Generated {_pretty_date(data.generated_at.date())}"),
    )
    c.drawRightString(
        PAGE_W - MARGIN, y - 10,
        tx(f"Covering the last {data.range_days} days"),
    )

    y -= 16
    c.setStrokeColor(INK)
    c.setLineWidth(1.1)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)

    # ── Context strip ────────────────────────────────────────────────────────
    y -= 14
    cells = [
        ("Surgery",       (data.surgery_type or "—").upper()),
        ("Date",          _pretty_date(data.surgery_date)),
        ("Weeks post-op", "—" if data.weeks_post_op is None else str(data.weeks_post_op)),
        ("Knee",          (data.knee_side or "—").title()),
        ("Safe limit",    _degrees(data.ceiling_deg)),
    ]
    cell_w = CONTENT / len(cells)
    for i, (label, value) in enumerate(cells):
        cx = MARGIN + i * cell_w
        c.setFont(regular, 7)
        c.setFillColor(MUTED)
        c.drawString(cx, y - 7, tx(label.upper()))
        c.setFont(bold, 11)
        c.setFillColor(INK)
        c.drawString(cx, y - 21, tx(_fit(value, bold, 11, cell_w - 6)))

    y -= 30
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)

    # The footer is written last, but its height is settled now so that nothing
    # above can grow into it.
    footer_reserve = _footer_height(notes, regular)
    cursor = _Cursor(top=y - 14, bottom=MARGIN + footer_reserve)

    # ── Flags ────────────────────────────────────────────────────────────────
    if data.flags:
        shown, dropped = [], 0
        for flag in data.flags:
            lines = _wrap(tx(flag.summary), regular, 8.5, CONTENT - 30, 2)
            need = 11 * len(lines) + 3
            if cursor.room(need + 16):
                shown.append((flag, lines, need))
            else:
                dropped += 1

        if shown:
            block_h = sum(s[2] for s in shown) + 18
            top = cursor.take(block_h)
            c.setFillColor(HexColor("#FCF6F5"))
            c.setStrokeColor(HexColor("#E9D5D2"))
            c.setLineWidth(0.6)
            c.rect(MARGIN, top, CONTENT, block_h, stroke=1, fill=1)

            ty = top + block_h - 12
            c.setFont(bold, 8)
            c.setFillColor(URGENT)
            c.drawString(MARGIN + 10, ty, tx("WORTH RAISING AT THIS APPOINTMENT"))
            ty -= 12
            for flag, lines, _need in shown:
                colour = SEVERITY_COLOUR.get(flag.severity, MUTED)
                c.setFont(bold, 8.5)
                c.setFillColor(colour)
                c.drawString(MARGIN + 10, ty, SEVERITY_MARK.get(flag.severity, "•"))
                c.setFont(regular, 8.5)
                c.setFillColor(INK_2)
                for line in lines:
                    c.drawString(MARGIN + 20, ty, line)
                    ty -= 11
                ty -= 3
            cursor.y -= 10

        if dropped:
            notes.append(
                f"{dropped} further flag{'s' if dropped != 1 else ''} did not fit on this page."
            )

    # ── Left column: chart, then the exercise table ──────────────────────────
    left_top = cursor.y
    ly = left_top

    rows = list(data.rom)
    if rows:
        c.setFont(bold, 9)
        c.setFillColor(INK)
        c.drawString(MARGIN, ly - 9, tx(f"Knee flexion — {_fit(rows[0].exercise, bold, 9, LEFT_W - 90)}"))
        ly -= 16

        chart_h = 96
        if cursor.room((left_top - ly) + chart_h):
            _draw_chart(c, tx, fonts, MARGIN, ly - chart_h, LEFT_W, chart_h, rows[0])
            ly -= chart_h + 16

        # Table
        c.setFont(regular, 7)
        c.setFillColor(MUTED)
        col_best, col_latest, col_limit = MARGIN + LEFT_W - 132, MARGIN + LEFT_W - 74, MARGIN + LEFT_W - 8
        c.drawString(MARGIN, ly - 7, tx("EXERCISE"))
        c.drawRightString(col_best, ly - 7, tx("BEST"))
        c.drawRightString(col_latest, ly - 7, tx("LATEST"))
        c.drawRightString(col_limit, ly - 7, tx("LIMIT"))
        ly -= 12
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(MARGIN, ly, MARGIN + LEFT_W, ly)
        ly -= 12

        drawn = 0
        for row in rows:
            if not cursor.room((left_top - ly) + 14):
                break
            c.setFont(regular, 8.5)
            c.setFillColor(INK_2)
            c.drawString(MARGIN, ly - 7, tx(_fit(row.exercise, regular, 8.5, LEFT_W - 150)))
            c.setFillColor(INK)
            c.drawRightString(col_best, ly - 7, _degrees(row.best_deg))
            c.drawRightString(col_latest, ly - 7, _degrees(row.latest_deg))
            c.setFillColor(MUTED)
            c.drawRightString(col_limit, ly - 7, _degrees(row.angle_limit))
            ly -= 14
            drawn += 1

        if drawn < len(rows):
            missing = len(rows) - drawn
            notes.append(
                f"{missing} less-practised exercise{'s are' if missing != 1 else ' is'} "
                "not listed here; the app has the full breakdown."
            )
    else:
        c.setFont(regular, 8.5)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, ly - 9, tx("No verified range-of-motion measurements in this period."))
        ly -= 20

    # ── Right column: the numbers that need no chart ─────────────────────────
    ry = left_top

    def stat_block(title: str, entries: list[tuple[str, str]], foot: Optional[str] = None) -> None:
        nonlocal ry
        c.setFont(bold, 8)
        c.setFillColor(ACCENT)
        c.drawString(RIGHT_X, ry - 8, tx(title.upper()))
        ry -= 18
        for label, value in entries:
            c.setFont(bold, 12)
            c.setFillColor(INK)
            c.drawString(RIGHT_X, ry - 9, tx(_fit(value, bold, 12, RIGHT_W * 0.52)))
            c.setFont(regular, 7.5)
            c.setFillColor(MUTED)
            c.drawRightString(RIGHT_X + RIGHT_W, ry - 9, tx(_fit(label, regular, 7.5, RIGHT_W * 0.45)))
            ry -= 15
        if foot:
            c.setFont(regular, 7)
            c.setFillColor(MUTED)
            for line in _wrap(tx(foot), regular, 7, RIGHT_W, 3):
                c.drawString(RIGHT_X, ry - 6, line)
                ry -= 9
        ry -= 10
        c.setStrokeColor(LINE_SOFT)
        c.setLineWidth(0.5)
        c.line(RIGHT_X, ry, RIGHT_X + RIGHT_W, ry)
        ry -= 12

    stat_block("Adherence", [
        (f"session{'s' if data.sessions != 1 else ''}", str(data.sessions)),
        ("days active", str(data.active_days)),
        ("day streak", str(data.current_streak_days)),
    ], foot=f"Best streak {data.longest_streak_days} days." if data.longest_streak_days else None)

    if data.latest_pain_after is not None or data.mean_pain_change is not None:
        entries = []
        if data.latest_pain_after is not None:
            entries.append(("latest, after", f"{data.latest_pain_after:.0f}/10"))
        if data.mean_pain_change is not None:
            change = data.mean_pain_change
            arrow = "+" if change > 0 else ""
            entries.append(("mean per session", f"{arrow}{change:.1f}"))
        stat_block("Pain (0–10)", entries,
                   foot="A session that ends more painful than it started is the number to watch.")

    for outcome in data.outcomes[:2]:
        stat_block(
            f"{outcome.instrument} · {outcome.knee_side}",
            [("out of 100", f"{outcome.latest:.0f}"),
             ("at baseline", f"{outcome.baseline:.0f}")],
            foot=f"{outcome.band}. {outcome.change_note}",
        )

    cursor.y = min(ly, ry)

    # ── Notes ────────────────────────────────────────────────────────────────
    # Only knowable once every string has been through `tx`.
    if tx.dropped:
        notes.insert(-1, (
            "Some characters could not be drawn with the available font and appear as \"?\". "
            f"Set {FONT_ENV} to a font covering them."
        ))

    fy = MARGIN + footer_reserve - 12
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, fy + 8, PAGE_W - MARGIN, fy + 8)

    c.setFont(regular, 7)
    c.setFillColor(MUTED)
    for line in _note_lines([tx(n) for n in notes], regular):
        if fy < MARGIN:
            # The reserve is capped, so a page with a great deal to say can still
            # run out. Losing the last standing note is the least-bad outcome
            # available and beats overprinting the margin.
            break
        c.drawString(MARGIN, fy, line)
        fy -= NOTE_LEADING

    c.showPage()
    c.save()
    return buf.getvalue()
