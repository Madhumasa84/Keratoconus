"""Local de-identified operational export; never includes source paths or direct IDs."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

def _token(value: str | None) -> str:
    return hashlib.sha256((value or "").encode()).hexdigest()[:16]

def export_deidentified(screenings: list[dict], output_path: str | Path) -> str:
    rows=[]
    for row in screenings:
        rows.append({"encounter_token":_token(row.get("screening_id")),"age":row.get("age"),"sex":row.get("sex"),
                     "screening_date":row.get("screening_date"),"overall_result":row.get("overall_result"),
                     "referral_priority":row.get("referral_priority"),"protocol_version":row.get("protocol_version")})
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({"deidentified":True,"records":rows},indent=2),encoding="utf-8")
    return str(output.resolve())
