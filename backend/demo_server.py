import sys
import json
import threading
import io
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT  = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FRONTEND_DIR  = PROJECT_ROOT / "frontend"

app = FastAPI(title="AI Resume Scanner", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUN_STATE = {
    "status":       "idle",
    "current":      0,
    "total":        0,
    "current_file": None,
    "error":        None,
}


class RunRequest(BaseModel):
    jd_text:         str
    scoring_weights: Optional[dict] = None


class QuestionsBody(BaseModel):
    questions: Optional[dict] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_latest_batch() -> dict:
    files = sorted(
        PROCESSED_DIR.glob("batch_ranking_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No ranking results found yet. Submit a job first."
        )
    return json.loads(files[0].read_text(encoding="utf-8"))


def _get_candidate(index: int, pool: str = "ranking") -> tuple[dict, dict]:
    data       = _load_latest_batch()
    candidates = data.get(pool, [])
    if index < 0 or index >= len(candidates):
        raise HTTPException(
            status_code=404,
            detail=f"Candidate index {index} not found in '{pool}'."
        )
    return candidates[index], data.get("jd_criteria", {})


def _make_pdf_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    INK   = colors.HexColor("#14253d")
    GOLD  = colors.HexColor("#b8842c")
    GREEN = colors.HexColor("#156a52")
    RED   = colors.HexColor("#a6501c")
    GRAY  = colors.HexColor("#74705f")
    LIGHT = colors.HexColor("#f6f4ed")

    styles = getSampleStyleSheet()
    extra  = {
        "title":  ParagraphStyle("title",  fontSize=18, textColor=INK,
                                 fontName="Helvetica-Bold", spaceAfter=4),
        "sub":    ParagraphStyle("sub",    fontSize=11, textColor=GRAY,
                                 fontName="Helvetica",      spaceAfter=2),
        "label":  ParagraphStyle("label",  fontSize=8,  textColor=GOLD,
                                 fontName="Helvetica-Bold", spaceBefore=14,
                                 spaceAfter=4, tracking=1),
        "body":   ParagraphStyle("body",   fontSize=10, textColor=INK,
                                 fontName="Helvetica",      leading=15,
                                 spaceAfter=4),
        "italic": ParagraphStyle("italic", fontSize=10, textColor=GRAY,
                                 fontName="Helvetica-Oblique", leading=15),
        "warn":   ParagraphStyle("warn",   fontSize=9,  textColor=RED,
                                 fontName="Helvetica",      leading=13),
        "footer": ParagraphStyle("footer", fontSize=8,  textColor=GRAY,
                                 fontName="Helvetica",      spaceBefore=6),
        "qhead":  ParagraphStyle("qhead",  fontSize=9,  textColor=GOLD,
                                 fontName="Helvetica-Bold", spaceBefore=10,
                                 spaceAfter=4, tracking=1),
        "qitem":  ParagraphStyle("qitem",  fontSize=10, textColor=INK,
                                 fontName="Helvetica",      leading=15,
                                 spaceAfter=3, leftIndent=12),
    }
    colors_map = dict(INK=INK, GOLD=GOLD, GREEN=GREEN,
                      RED=RED, GRAY=GRAY, LIGHT=LIGHT)
    return styles, extra, colors_map


def _build_score_table(candidate: dict, colors_map: dict):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    LINE  = colors.HexColor("#ded6c0")
    INK   = colors_map["INK"]
    GRAY  = colors_map["GRAY"]
    LIGHT = colors_map["LIGHT"]

    data = [
        ["Fit score", "Rank", "Skills", "Experience", "Semantic"],
        [
            str(candidate.get("fit_score", "—")),
            f"#{candidate.get('rank_position', '—')}",
            str(candidate.get("skill_score", "—")),
            str(candidate.get("experience_score", "—")),
            str(candidate.get("semantic_score", "—")),
        ],
    ]
    t = Table(data, colWidths=[3.2*cm, 2*cm, 2.5*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), LIGHT),
        ("TEXTCOLOR",     (0, 0), (-1, 0), GRAY),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("FONTNAME",      (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 1), (-1, 1), 13),
        ("TEXTCOLOR",     (0, 1), (0,  1), INK),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("BOX",           (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _build_candidate_story(candidate: dict, jd_criteria: dict,
                            extra: dict, colors_map: dict,
                            questions: dict | None = None) -> list:
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    from reportlab.lib import colors

    LINE = colors.HexColor("#ded6c0")
    INK  = colors_map["INK"]

    name          = candidate.get("name") or "Unknown"
    file_name     = candidate.get("file") or ""
    institution   = candidate.get("institution") or "Not extracted"
    edu_notes     = candidate.get("education_notes") or "—"
    reasoning     = candidate.get("reasoning") or "—"
    gate_note     = candidate.get("education_gate_note") or ""
    skill_matches = candidate.get("skill_matches") or []
    skill_gaps    = candidate.get("skill_gaps") or []

    role_type  = (jd_criteria.get("role_type") or "Role").upper()
    seniority  = jd_criteria.get("seniority") or ""
    location   = jd_criteria.get("location") or ""
    req_skills = (jd_criteria.get("skills") or {}).get("required") or []

    story = []
    story.append(Paragraph(name, extra["title"]))
    story.append(Paragraph(file_name, extra["sub"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.8, color=INK))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Job", extra["label"]))
    story.append(Paragraph(
        f"{role_type} · {seniority}{' · ' + location if location else ''}",
        extra["body"]
    ))
    if req_skills:
        story.append(Paragraph(f"Required: {', '.join(req_skills)}", extra["body"]))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.4, color=LINE))

    story.append(Paragraph("Score breakdown", extra["label"]))
    story.append(_build_score_table(candidate, colors_map))

    if gate_note:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"⚠ {gate_note}", extra["warn"]))

    story.append(Paragraph("Matched skills", extra["label"]))
    story.append(Paragraph(
        ", ".join(skill_matches) if skill_matches else "None verified",
        extra["body"] if skill_matches else extra["italic"]
    ))

    story.append(Paragraph("Missing skills", extra["label"]))
    story.append(Paragraph(
        ", ".join(skill_gaps) if skill_gaps else "None",
        extra["body"] if skill_gaps else extra["italic"]
    ))

    story.append(Paragraph("Education", extra["label"]))
    story.append(Paragraph(f"Institution: {institution}", extra["body"]))
    story.append(Paragraph(edu_notes, extra["italic"]))

    story.append(Paragraph("AI reasoning", extra["label"]))
    story.append(HRFlowable(width="100%", thickness=0.4, color=LINE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(reasoning, extra["italic"]))

    if questions and questions.get("sections"):
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.4, color=LINE))
        story.append(Paragraph("Interview question bank", extra["label"]))
        section_labels = {
            "technical_depth":         "Technical depth",
            "skill_gap_probing":       "Skill gap probing",
            "experience_leadership":   "Experience & leadership",
            "culture_problem_solving": "Culture & problem solving",
        }
        for key, label in section_labels.items():
            qs = questions["sections"].get(key) or []
            if not qs:
                continue
            story.append(Paragraph(label.upper(), extra["qhead"]))
            for idx, q in enumerate(qs, 1):
                story.append(Paragraph(f"{idx}. {q}", extra["qitem"]))

    return story


def _build_candidate_pdf(candidate: dict, jd_criteria: dict,
                          questions: dict | None = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Spacer, HRFlowable, Paragraph
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)

    _, extra, colors_map = _make_pdf_styles()
    story = _build_candidate_story(candidate, jd_criteria, extra, colors_map, questions)

    LINE = colors.HexColor("#ded6c0")
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.4, color=LINE))
    story.append(Paragraph(
        f"Generated by AI Resume Scanner · "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')} · TechnoRise 2.0",
        extra["footer"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _build_full_report_pdf(ranking: list, jd_criteria: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Spacer, HRFlowable,
        Paragraph, PageBreak, Table, TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)

    _, extra, colors_map = _make_pdf_styles()
    INK  = colors_map["INK"]
    GRAY = colors_map["GRAY"]
    GOLD = colors_map["GOLD"]
    LINE = colors.HexColor("#ded6c0")

    role_type  = (jd_criteria.get("role_type") or "Role").upper()
    seniority  = jd_criteria.get("seniority") or ""
    location   = jd_criteria.get("location") or ""
    req_skills = (jd_criteria.get("skills") or {}).get("required") or []

    cover_title = ParagraphStyle("ctitle", fontSize=22, textColor=INK,
                                 fontName="Helvetica-Bold", spaceAfter=8)
    cover_sub   = ParagraphStyle("csub",   fontSize=13, textColor=GRAY,
                                 fontName="Helvetica",      spaceAfter=4)
    cover_label = ParagraphStyle("clabel", fontSize=9,  textColor=GOLD,
                                 fontName="Helvetica-Bold", spaceAfter=3, tracking=1)

    story = []

    # Cover page
    story.append(Spacer(1, 40))
    story.append(Paragraph("Candidate Ranking Report", cover_title))
    story.append(Paragraph(
        f"{role_type} · {seniority}{' · ' + location if location else ''}",
        cover_sub
    ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=INK))
    story.append(Spacer(1, 12))

    story.append(Paragraph("REQUIRED SKILLS", cover_label))
    story.append(Paragraph(", ".join(req_skills) if req_skills else "—", extra["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("TOTAL CANDIDATES RANKED", cover_label))
    story.append(Paragraph(str(len(ranking)), extra["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("GENERATED", cover_label))
    story.append(Paragraph(datetime.now().strftime("%d %b %Y, %H:%M"), extra["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("GENERATED BY", cover_label))
    story.append(Paragraph("AI Resume Scanner · TechnoRise 2.0 · Tech Mahindra", extra["body"]))

    # Summary table
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.4, color=LINE))
    story.append(Paragraph("RANKING SUMMARY", cover_label))

    summary_data = [["Rank", "Candidate", "File", "Fit Score", "Skills", "Experience"]]
    for c in ranking:
        summary_data.append([
            f"#{c.get('rank_position', '—')}",
            (c.get("name") or "Unknown")[:22],
            (c.get("file") or "")[:20],
            str(c.get("fit_score", "—")),
            str(c.get("skill_score", "—")),
            str(c.get("experience_score", "—")),
        ])

    summary_table = Table(
        summary_data,
        colWidths=[1.5*cm, 4.5*cm, 4*cm, 2.5*cm, 2*cm, 2.5*cm]
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors_map["LIGHT"]),
        ("TEXTCOLOR",      (0, 0), (-1, 0), GRAY),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0), 8),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#faf9f4")]),
        ("BOX",            (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID",      (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)

    # One page per candidate
    for candidate in ranking:
        story.append(PageBreak())
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"CANDIDATE #{candidate.get('rank_position', '—')} OF {len(ranking)}",
            ParagraphStyle("rank_badge", fontSize=8, textColor=GRAY,
                           fontName="Helvetica-Bold", spaceAfter=6, tracking=1)
        ))
        story.extend(
            _build_candidate_story(candidate, jd_criteria, extra, colors_map, questions=None)
        )
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.4, color=LINE))
        story.append(Paragraph(
            f"AI Resume Scanner · TechnoRise 2.0 · "
            f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
            extra["footer"]
        ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _execute_run(jd_text: str, scoring_weights: dict | None):
    from pipeline.matching.batch_runner import run_full_batch

    RUN_STATE["status"]       = "running"
    RUN_STATE["current"]      = 0
    RUN_STATE["total"]        = 0
    RUN_STATE["current_file"] = None
    RUN_STATE["error"]        = None

    def progress(i, total, filename):
        RUN_STATE["current"]      = i
        RUN_STATE["total"]        = total
        RUN_STATE["current_file"] = filename

    try:
        run_full_batch(jd_text, on_progress=progress, custom_weights=scoring_weights)
        RUN_STATE["status"] = "done"
    except Exception as e:
        RUN_STATE["status"] = "error"
        RUN_STATE["error"]  = str(e)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/api/run")
def start_run(req: RunRequest):
    if RUN_STATE["status"] == "running":
        raise HTTPException(status_code=409, detail="A batch is already running.")
    if not req.jd_text or not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text is required.")
    thread = threading.Thread(
        target=_execute_run, args=(req.jd_text, req.scoring_weights), daemon=True
    )
    thread.start()
    return {"status": "started"}


@app.get("/api/run-status")
def get_run_status():
    return RUN_STATE


@app.get("/api/rankings/latest")
def get_latest_ranking():
    data = _load_latest_batch()
    data["_source_file"] = ""
    return data


@app.api_route("/api/candidate-report/{pool}/{index}", methods=["GET", "POST"])
async def download_candidate_report(
    pool:              str,
    index:             int,
    include_questions: bool         = Query(False),
    body:              QuestionsBody = None,
):
    """
    GET  ?include_questions=true  — generate questions fresh via Qwen
    GET  (no param)               — no questions in PDF
    POST {questions: {...}}       — use pre-generated questions from frontend cache,
                                    no Qwen call made
    """
    if pool not in ("ranking", "excluded"):
        raise HTTPException(status_code=400, detail="Pool must be 'ranking' or 'excluded'.")
    candidate, jd_criteria = _get_candidate(index, pool)

    questions = None

    if body and body.questions:
        # Frontend sent pre-generated questions — use directly, skip Qwen
        questions = body.questions
    elif include_questions:
        # Questions not cached — generate fresh
        from pipeline.matching.interview_questions import generate_interview_questions
        try:
            questions = generate_interview_questions(candidate, jd_criteria)
        except Exception:
            questions = None

    pdf_bytes = _build_candidate_pdf(candidate, jd_criteria, questions)
    name      = (candidate.get("name") or "candidate").replace(" ", "_")
    filename  = f"report_{name}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/full-report")
def download_full_report():
    data    = _load_latest_batch()
    ranking = data.get("ranking", [])
    if not ranking:
        raise HTTPException(status_code=404, detail="No ranked candidates found.")
    jd_criteria = data.get("jd_criteria", {})
    pdf_bytes   = _build_full_report_pdf(ranking, jd_criteria)
    role        = (jd_criteria.get("role_type") or "candidates").replace(" ", "_")
    filename    = f"ranking_report_{role}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/api/interview-questions/{pool}/{index}")
def get_interview_questions(pool: str, index: int):
    if pool not in ("ranking", "excluded"):
        raise HTTPException(status_code=400, detail="Pool must be 'ranking' or 'excluded'.")
    candidate, jd_criteria = _get_candidate(index, pool)
    from pipeline.matching.interview_questions import generate_interview_questions
    try:
        result = generate_interview_questions(candidate, jd_criteria)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "dashboard.html")


@app.get("/intake")
def serve_intake():
    return FileResponse(FRONTEND_DIR / "job_intake.html")


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "job_intake.html")