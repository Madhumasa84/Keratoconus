"""
report_styles.py — ReportLab style definitions for KERASCAN PDF reports.

Colorblind-safe palette: navy blue and amber only (no red/green).
"""
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.units import cm
from reportlab.platypus import TableStyle

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
NAVY        = colors.HexColor('#003366')
AMBER       = colors.HexColor('#FF8C00')
LIGHT_BLUE  = colors.HexColor('#E8F0FE')
LIGHT_AMBER = colors.HexColor('#FFF3E0')
WHITE       = colors.white
DARK_GRAY   = colors.HexColor('#333333')
MID_GRAY    = colors.HexColor('#666666')
LIGHT_GRAY  = colors.HexColor('#F5F5F5')
BORDER_GRAY = colors.HexColor('#CCCCCC')


def get_styles() -> dict:
    """Return a dict of named ParagraphStyle objects."""
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title",
        parent=base["Title"],
        fontSize=18,
        textColor=NAVY,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontSize=11,
        textColor=MID_GRAY,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    styles["section_header"] = ParagraphStyle(
        "section_header",
        parent=base["Heading2"],
        fontSize=12,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=4,
        borderPad=2,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=9,
        textColor=DARK_GRAY,
        spaceAfter=3,
    )
    styles["small"] = ParagraphStyle(
        "small",
        parent=base["Normal"],
        fontSize=7.5,
        textColor=MID_GRAY,
    )
    styles["disclaimer"] = ParagraphStyle(
        "disclaimer",
        parent=base["Normal"],
        fontSize=8,
        textColor=DARK_GRAY,
        backColor=LIGHT_AMBER,
        borderPad=6,
        spaceAfter=6,
    )
    styles["result_negative"] = ParagraphStyle(
        "result_negative",
        parent=base["Normal"],
        fontSize=11,
        textColor=NAVY,
        backColor=LIGHT_BLUE,
        alignment=TA_CENTER,
        borderPad=4,
    )
    styles["result_referral"] = ParagraphStyle(
        "result_referral",
        parent=base["Normal"],
        fontSize=11,
        textColor=colors.HexColor('#7A5200'),
        backColor=LIGHT_AMBER,
        alignment=TA_CENTER,
        borderPad=4,
    )
    styles["footer"] = ParagraphStyle(
        "footer",
        parent=base["Normal"],
        fontSize=7,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
    )
    styles["table_header"] = ParagraphStyle(
        "table_header",
        parent=base["Normal"],
        fontSize=9,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
    styles["cell"] = ParagraphStyle(
        "cell",
        parent=base["Normal"],
        fontSize=8.5,
        textColor=DARK_GRAY,
        alignment=TA_LEFT,
    )
    return styles


def get_header_table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def get_measurements_table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])


def get_result_table_style(is_referral: bool) -> TableStyle:
    bg = LIGHT_AMBER if is_referral else LIGHT_BLUE
    border = AMBER if is_referral else NAVY
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, border),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])


def get_audit_table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK_GRAY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("FONTSIZE",      (0, 1), (-1, -1), 7.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER_GRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ])
