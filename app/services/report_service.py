"""
report_service.py — PDF, JSON, and Excel report generation for KERASCAN.

All output is produced locally; no network access required.
ReportLab built-in fonts only.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DISCLAIMER = (
    "AI-assisted keratoconus screening result. This is not a confirmed diagnosis. "
    "Positive, discordant, ungradable or clinically concerning findings require "
    "repeat assessment, corneal tomography and qualified clinical review."
)

HUMAN_REASON = {
    "IMG_SUSPICIOUS":       "Suspicious Placido pattern",
    "IMG_UNGRADABLE":       "Image ungradable — recapture required",
    "K_HIGH":               "Elevated keratometry (K2)",
    "PACHY_LOW":            "Low corneal thickness (pachymetry)",
    "CYL_HIGH":             "High cylinder (astigmatism)",
    "TWO_DOMAIN_ABNORMAL":  "Two or more quantitative domains abnormal",
    "CLINICAL_SIGN":        "Clinical sign present (e.g. Vogt striae, Fleischer ring)",
    "INTER_EYE_ASYMMETRY":  "Inter-eye keratometry asymmetry",
    "REPEAT_REQUIRED":      "Repeat measurement required",
    "MEASUREMENT_MISSING":  "Required measurement(s) missing",
}


def _safe(val: Any, default: str = "—") -> str:
    if val is None or val == "":
        return default
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


class ReportService:
    """Generates PDF, JSON, and Excel exports for a KERASCAN screening."""

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def generate_json(self, screening_data: dict, output_path: str) -> str:
        """Generate complete JSON export. Returns path."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        export = {
            "export_generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": DISCLAIMER,
            "screening": screening_data,
        }
        output.write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
        log.info("generate_json: wrote %s", output)
        return str(output.resolve())

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------

    def generate_excel(self, screening_data: dict, output_path: str) -> str:
        """Generate Excel with 3 sheets: Summary, Measurements, Audit."""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        except ImportError:
            raise RuntimeError("openpyxl is required for Excel export. Install it with: pip install openpyxl")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        wb = openpyxl.Workbook()
        navy = "003366"
        amber = "FF8C00"
        light_blue = "E8F0FE"
        light_amber = "FFF3E0"
        white = "FFFFFF"

        def header_style(cell, bg=navy, fg=white):
            cell.font = Font(bold=True, color=fg, size=10)
            cell.fill = PatternFill("solid", fgColor=bg)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        def set_col_widths(ws, widths):
            for col, w in widths.items():
                ws.column_dimensions[col].width = w

        thin = Side(style="thin", color="CCCCCC")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        # ── Sheet 1: Summary ────────────────────────────────────────
        ws1 = wb.active
        ws1.title = "Summary"
        ws1.append(["KERASCAN Screening Report"])
        ws1["A1"].font = Font(bold=True, size=14, color=navy)
        ws1.append(["Protocol version:", _safe(screening_data.get("protocol_version"))])
        ws1.append(["Generated:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")])
        ws1.append([])

        # Metadata
        meta_headers = ["Field", "Value"]
        ws1.append(meta_headers)
        for cell in ws1[ws1.max_row]:
            header_style(cell)
        meta_rows = [
            ("Screening ID", screening_data.get("screening_id", "")),
            ("Age", screening_data.get("age", "")),
            ("Sex", screening_data.get("sex", "")),
            ("Site / School", screening_data.get("site", "")),
            ("Screening Date", screening_data.get("screening_date", "")),
            ("Operator ID", screening_data.get("operator_id", "")),
            ("Device ID", screening_data.get("device_id", "")),
            ("Consent Recorded", "Yes" if screening_data.get("consent_recorded") else "No"),
            ("Overall Result", screening_data.get("overall_result", "")),
            ("Referral Priority", screening_data.get("referral_priority", "")),
        ]
        for row in meta_rows:
            ws1.append(list(row))
            ws1.cell(ws1.max_row, 1).font = Font(bold=True)
        ws1.append([])
        ws1.append(["DISCLAIMER:", DISCLAIMER])
        ws1.cell(ws1.max_row, 1).font = Font(bold=True, color=amber)
        ws1.cell(ws1.max_row, 2).alignment = Alignment(wrap_text=True)
        ws1.row_dimensions[ws1.max_row].height = 60
        set_col_widths(ws1, {"A": 22, "B": 60})

        # ── Sheet 2: Measurements ────────────────────────────────────
        ws2 = wb.create_sheet("Measurements")
        meas_headers = [
            "Eye", "Reading #", "K1 (D)", "K1 Axis (°)",
            "K2/Steep K (D)", "K2 Axis (°)", "Kmax (D)", "Mean K (D)",
            "Pachymetry (µm)", "Pachy Type", "Sphere (D)", "Cylinder (D)",
            "Cyl Axis (°)", "VA (logMAR)", "Refraction Type", "Quality",
        ]
        ws2.append(meas_headers)
        for cell in ws2[1]:
            header_style(cell)

        for eye in screening_data.get("eyes", []):
            for m in eye.get("measurements", []):
                ws2.append([
                    eye.get("laterality", ""),
                    _safe(m.get("reading_number")),
                    _safe(m.get("k1_d")),
                    _safe(m.get("k1_axis")),
                    _safe(m.get("k2_d")),
                    _safe(m.get("k2_axis")),
                    _safe(m.get("kmax_d")),
                    _safe(m.get("mean_k_d")),
                    _safe(m.get("pachymetry_um")),
                    _safe(m.get("pachymetry_type")),
                    _safe(m.get("sphere_d")),
                    _safe(m.get("cylinder_d")),
                    _safe(m.get("cylinder_axis")),
                    _safe(m.get("va_logmar")),
                    _safe(m.get("refraction_type")),
                    _safe(m.get("measurement_quality")),
                ])
        set_col_widths(ws2, {chr(65+i): 14 for i in range(len(meas_headers))})
        ws2.column_dimensions["A"].width = 6

        # ── Sheet 3: Audit ───────────────────────────────────────────
        ws3 = wb.create_sheet("Audit Trail")
        audit_headers = ["Table", "Record ID", "Action", "Performed By", "Timestamp", "Old Value", "New Value"]
        ws3.append(audit_headers)
        for cell in ws3[1]:
            header_style(cell, bg="333333")

        for eye in screening_data.get("eyes", []):
            for d in eye.get("decisions", []):
                ws3.append([
                    "decisions", d.get("id", ""), "INSERT",
                    "system", d.get("created_at", ""), "", json.dumps(d, default=str)
                ])
                if d.get("is_overridden"):
                    ws3.append([
                        "decisions", d.get("id", ""), "UPDATE",
                        d.get("override_by", ""), d.get("override_at", ""),
                        d.get("override_original", ""),
                        f"{d.get('override_new', '')} — {d.get('override_reason', '')}",
                    ])

        set_col_widths(ws3, {"A": 14, "B": 38, "C": 10, "D": 20, "E": 22, "F": 20, "G": 50})
        ws3.column_dimensions["G"].width = 50

        wb.save(str(output))
        log.info("generate_excel: wrote %s", output)
        return str(output.resolve())

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def generate_pdf(self, screening_data: dict, output_path: str) -> str:
        """Generate a professional 3-page PDF report. Returns path."""
        try:
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, Image as RLImage, HRFlowable,
            )
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.units import cm
        except ImportError:
            raise RuntimeError("reportlab is required. Install with: pip install reportlab")

        from app.templates.report_styles import (
            get_styles, get_header_table_style, get_measurements_table_style,
            get_result_table_style, get_audit_table_style,
            NAVY, AMBER, LIGHT_BLUE, LIGHT_AMBER, WHITE, DARK_GRAY, MID_GRAY, LIGHT_GRAY,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        S = get_styles()
        proto_ver = _safe(screening_data.get("protocol_version"), "unknown")
        gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        def make_header_footer(canvas, doc):
            canvas.saveState()
            w, h = A4
            # Header bar
            canvas.setFillColor(NAVY)
            canvas.rect(0, h - 1.2*cm, w, 1.2*cm, fill=True, stroke=False)
            canvas.setFillColor(WHITE)
            canvas.setFont("Helvetica-Bold", 10)
            canvas.drawString(1*cm, h - 0.85*cm, "KERASCAN Keratoconus Screening Report")
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(w - 1*cm, h - 0.85*cm, f"Protocol v{proto_ver}  |  {gen_time}")
            # Footer
            canvas.setFillColor(MID_GRAY)
            canvas.setFont("Helvetica", 7)
            canvas.drawString(1*cm, 0.6*cm, DISCLAIMER[:120] + "...")
            canvas.drawRightString(w - 1*cm, 0.6*cm, f"Page {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(
            str(output),
            pagesize=A4,
            topMargin=1.8*cm,
            bottomMargin=1.4*cm,
            leftMargin=1.5*cm,
            rightMargin=1.5*cm,
        )

        story = []

        # ================================================================
        # PAGE 1 — Clinical Summary
        # ================================================================
        story.append(Paragraph("KERASCAN Keratoconus Screening Report", S["title"]))
        story.append(Paragraph(f"Protocol Version: {proto_ver}  |  Generated: {gen_time}", S["subtitle"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8))

        # Metadata table
        story.append(Paragraph("Screening Information", S["section_header"]))
        meta = [
            ["Field", "Value"],
            ["Screening ID", _safe(screening_data.get("screening_id"))],
            ["Age", _safe(screening_data.get("age"))],
            ["Sex", _safe(screening_data.get("sex"))],
            ["Site / School", _safe(screening_data.get("site"))],
            ["Screening Date", _safe(screening_data.get("screening_date"))],
            ["Operator ID", _safe(screening_data.get("operator_id"))],
            ["Device ID", _safe(screening_data.get("device_id"))],
            ["Consent Recorded", "Yes" if screening_data.get("consent_recorded") else "No"],
        ]
        t = Table(meta, colWidths=[5*cm, 10*cm])
        t.setStyle(get_header_table_style())
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Measurements table
        story.append(Paragraph("Measurements", S["section_header"]))
        meas_data = [["Measurement", "OD (Right Eye)", "OS (Left Eye)"]]
        eyes = {e.get("laterality"): e for e in screening_data.get("eyes", [])}
        od_meas = next((m for m in eyes.get("OD", {}).get("measurements", []) if m.get("reading_number") in (None, 1)), {})
        os_meas = next((m for m in eyes.get("OS", {}).get("measurements", []) if m.get("reading_number") in (None, 1)), {})

        meas_rows = [
            ("K1 (flat K, D)", "k1_d"),
            ("K1 Axis (°)", "k1_axis"),
            ("K2 / Steep K (D)", "k2_d"),
            ("K2 Axis (°)", "k2_axis"),
            ("Kmax (D)", "kmax_d"),
            ("Mean K (D)", "mean_k_d"),
            ("Pachymetry (µm)", "pachymetry_um"),
            ("Pachymetry Type", "pachymetry_type"),
            ("Sphere (D)", "sphere_d"),
            ("Cylinder (D)", "cylinder_d"),
            ("Cylinder Axis (°)", "cylinder_axis"),
            ("VA (logMAR)", "va_logmar"),
            ("Refraction Type", "refraction_type"),
            ("Quality", "measurement_quality"),
        ]
        for label, key in meas_rows:
            meas_data.append([label, _safe(od_meas.get(key)), _safe(os_meas.get(key))])

        mt = Table(meas_data, colWidths=[5.5*cm, 4.5*cm, 4.5*cm])
        mt.setStyle(get_measurements_table_style())
        story.append(mt)
        story.append(Spacer(1, 0.4*cm))

        # Per-eye results
        story.append(Paragraph("Per-Eye Screening Results", S["section_header"]))
        for lat in ("OD", "OS"):
            eye = eyes.get(lat, {})
            decisions = eye.get("decisions", [])
            final_dec = decisions[-1].get("final_result", "—") if decisions else "—"
            codes = eye.get("reason_codes") or []
            code_str = ", ".join(HUMAN_REASON.get(c, c) for c in codes) if codes else "None"
            is_ref = final_dec not in ("SCREEN_NEGATIVE", "")
            eye_data = [
                [f"{lat} — {'Right Eye' if lat == 'OD' else 'Left Eye'}", ""],
                ["Engine result:", _safe(eye.get("eye_result"))],
                ["Decision:", final_dec],
                ["Reason codes:", code_str],
            ]
            et = Table(eye_data, colWidths=[4.5*cm, 10*cm])
            et.setStyle(get_result_table_style(is_ref))
            story.append(et)
            story.append(Spacer(1, 0.2*cm))

        # Child-level decision
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Overall Child-Level Decision", S["section_header"]))
        overall = _safe(screening_data.get("overall_result"))
        priority = _safe(screening_data.get("referral_priority"))
        is_child_ref = overall not in ("SCREEN_NEGATIVE", "—")
        child_data = [
            ["OVERALL RESULT", "REFERRAL PRIORITY"],
            [overall, priority],
        ]
        ct = Table(child_data, colWidths=[7.5*cm, 7.5*cm])
        ct.setStyle(get_result_table_style(is_child_ref))
        story.append(ct)
        story.append(Spacer(1, 0.4*cm))

        # Disclaimer box
        story.append(Paragraph("⚠ " + DISCLAIMER, S["disclaimer"]))

        story.append(PageBreak())

        # ================================================================
        # PAGE 2 — Images
        # ================================================================
        story.append(Paragraph("Image Analysis", S["section_header"]))
        story.append(Spacer(1, 0.3*cm))

        img_cells = []
        for lat in ("OD", "OS"):
            eye = eyes.get(lat, {})
            img_path = eye.get("image_path")
            cell_label = f"{lat} — Original Image"
            if img_path and Path(img_path).exists():
                try:
                    img = RLImage(img_path, width=7*cm, height=7*cm)
                    img_cells.append([img, Paragraph(cell_label, S["small"])])
                except Exception:
                    img_cells.append([Paragraph(f"{cell_label}\n[Image load error]", S["small"]), ""])
            else:
                img_cells.append([Paragraph(f"{cell_label}\n[Image not available]", S["small"]), ""])

        # Render as 2-column table
        if img_cells:
            row = [img_cells[0][0], img_cells[1][0] if len(img_cells) > 1 else ""]
            label_row = [img_cells[0][1], img_cells[1][1] if len(img_cells) > 1 else ""]
            img_table = Table([row, label_row], colWidths=[7.5*cm, 7.5*cm])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(img_table)
        story.append(Spacer(1, 0.4*cm))

        # Quality findings
        story.append(Paragraph("Quality Findings", S["section_header"]))
        q_data = [["Eye", "Gradable", "Quality Score", "Flags"]]
        for lat in ("OD", "OS"):
            eye = eyes.get(lat, {})
            flags = eye.get("quality_flags") or []
            q_data.append([
                lat,
                "Yes" if eye.get("quality_gradable") else "No",
                _safe(eye.get("quality_score")),
                ", ".join(flags) if flags else "None",
            ])
        qt = Table(q_data, colWidths=[2*cm, 2.5*cm, 3*cm, 7.5*cm])
        qt.setStyle(get_header_table_style())
        story.append(qt)
        story.append(Spacer(1, 0.3*cm))

        # ROI info
        story.append(Paragraph("ROI Detection", S["section_header"]))
        roi_data = [["Eye", "Method", "Confidence", "Radius (px)"]]
        for lat in ("OD", "OS"):
            eye = eyes.get(lat, {})
            roi_data.append([
                lat,
                _safe(eye.get("roi_method")),
                _safe(eye.get("roi_confidence")),
                _safe(eye.get("roi_radius")),
            ])
        rt = Table(roi_data, colWidths=[2*cm, 5*cm, 4*cm, 4*cm])
        rt.setStyle(get_header_table_style())
        story.append(rt)

        story.append(PageBreak())

        # ================================================================
        # PAGE 3 — Detailed Data & Audit
        # ================================================================
        story.append(Paragraph("Detailed Measurements & Repeat Readings", S["section_header"]))
        repeat_data = [["Eye", "Reading #", "K2 (D)", "Pachy (µm)", "Cyl (D)", "Quality"]]
        for lat in ("OD", "OS"):
            eye = eyes.get(lat, {})
            for m in eye.get("measurements", []):
                repeat_data.append([
                    lat,
                    _safe(m.get("reading_number")),
                    _safe(m.get("k2_d")),
                    _safe(m.get("pachymetry_um")),
                    _safe(m.get("cylinder_d")),
                    _safe(m.get("measurement_quality")),
                ])
        rdt = Table(repeat_data, colWidths=[2*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 2.5*cm])
        rdt.setStyle(get_measurements_table_style())
        story.append(rdt)
        story.append(Spacer(1, 0.4*cm))

        # Versions table
        story.append(Paragraph("Versions & Provenance", S["section_header"]))
        od_eye = eyes.get("OD", {})
        ia_list = od_eye.get("image_analyses", [{}])
        ia = ia_list[0] if ia_list else {}
        ver_data = [
            ["Item", "Value"],
            ["Protocol version", _safe(screening_data.get("protocol_version"))],
            ["Pipeline version (OD)", _safe(od_eye.get("pipeline_version"))],
            ["Model hash (OD)", _safe(ia.get("model_hash"))],
            ["Classification skipped (OD)", str(ia.get("classification_skipped", "—"))],
            ["Prototype score (OD)", _safe(ia.get("prototype_score"))],
        ]
        vt = Table(ver_data, colWidths=[5.5*cm, 9.5*cm])
        vt.setStyle(get_header_table_style())
        story.append(vt)
        story.append(Spacer(1, 0.4*cm))

        # Audit trail
        story.append(Paragraph("Decision Audit Trail", S["section_header"]))
        audit_rows = [["Level", "Result", "Overridden", "Override By", "Reason"]]
        all_decisions = screening_data.get("decisions", [])
        for eye_d in screening_data.get("eyes", []):
            all_decisions += eye_d.get("decisions", [])
        for d in all_decisions:
            audit_rows.append([
                _safe(d.get("decision_level")),
                _safe(d.get("final_result")),
                "Yes" if d.get("is_overridden") else "No",
                _safe(d.get("override_by")),
                _safe(d.get("override_reason")),
            ])
        at = Table(audit_rows, colWidths=[2*cm, 4*cm, 2.5*cm, 3.5*cm, 3*cm])
        at.setStyle(get_audit_table_style())
        story.append(at)
        story.append(Spacer(1, 0.4*cm))

        # Pentacam follow-up
        story.append(Paragraph("Pentacam / Corneal Tomography Follow-Up", S["section_header"]))
        pf_list = screening_data.get("pentacam_followups", [])
        if pf_list:
            pf = pf_list[0]
            pf_data = [
                ["Field", "Value"],
                ["Exam Date", _safe(pf.get("exam_date"))],
                ["Kmax OD", _safe(pf.get("kmax_od"))],
                ["Kmax OS", _safe(pf.get("kmax_os"))],
                ["Belin-Ambrósio D OD", _safe(pf.get("belin_ambrosio_d_od"))],
                ["Belin-Ambrósio D OS", _safe(pf.get("belin_ambrosio_d_os"))],
                ["Performed By", _safe(pf.get("performed_by"))],
                ["Notes", _safe(pf.get("notes"))],
            ]
        else:
            pf_data = [
                ["Field", "Value"],
                ["Exam Date", "[Not yet recorded]"],
                ["Kmax OD", ""],
                ["Kmax OS", ""],
                ["Belin-Ambrósio D OD", ""],
                ["Belin-Ambrósio D OS", ""],
                ["Performed By", ""],
                ["Notes", ""],
            ]
        pft = Table(pf_data, colWidths=[5.5*cm, 9.5*cm])
        pft.setStyle(get_header_table_style())
        story.append(pft)
        story.append(Spacer(1, 0.6*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=NAVY))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(DISCLAIMER, S["disclaimer"]))

        doc.build(story, onFirstPage=make_header_footer, onLaterPages=make_header_footer)
        log.info("generate_pdf: wrote %s", output)
        return str(output.resolve())

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def generate_all_exports(self, screening_data: dict, output_dir: str) -> dict[str, str]:
        """Generate PDF, JSON, and Excel. Returns dict of format -> path."""
        base = Path(output_dir)
        sid = screening_data.get("screening_id", "report")
        results = {}
        errors = {}

        for fmt, method, ext in [
            ("pdf",   self.generate_pdf,   "pdf"),
            ("json",  self.generate_json,  "json"),
            ("excel", self.generate_excel, "xlsx"),
        ]:
            path = str(base / f"{sid}.{ext}")
            try:
                results[fmt] = method(screening_data, path)
            except Exception as exc:
                log.error("generate_all_exports: %s failed: %s", fmt, exc)
                errors[fmt] = str(exc)

        if errors:
            results["errors"] = errors
        return results
