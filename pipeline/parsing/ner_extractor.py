import re
import logging
from dataclasses import dataclass, field
from backend.schemas.candidate import ContactInfo
from config.pipeline_config import (
    SPACY_MODELS,
    SPACY_TEXT_LIMIT,
    MAX_ORGS_DETECTED,
    NAME_MIN_WORDS,
    NAME_MAX_WORDS,
    NAME_HEURISTIC_LINES,
    NAME_MAX_LINE_LENGTH,
    CONTACT_STRIP_NAME_LINES,
)

logger = logging.getLogger(__name__)
_nlp = None

def _get_nlp():
    global _nlp  
    if _nlp is not None:
        return _nlp

    for model_name in SPACY_MODELS:
        try:
            import spacy
            _nlp = spacy.load(model_name)
            logger.info(f"spaCy model loaded: {model_name}")
            return _nlp
        except OSError:
            continue

    logger.warning(
        "No spaCy model found. Running in regex-only mode. "
        "Install with: python -m spacy download en_core_web_lg"
    )
    return None

@dataclass
class NERResult:
    contact: ContactInfo
    detected_orgs: list[str]  = field(default_factory=list)
    detected_dates: list[str] = field(default_factory=list)
    spacy_used: bool = False
    spacy_used: bool = False


def extract_entities(clean_text: str) -> NERResult:
    email    = _extract_email(clean_text)
    phone    = _extract_phone(clean_text)
    linkedin = _extract_linkedin(clean_text)
    github   = _extract_github(clean_text)
    dates    = _extract_dates(clean_text)

    nlp        = _get_nlp()
    name       = None
    location   = None
    orgs       = []
    spacy_used = False

    if nlp is not None:
        spacy_used            = True
        name, location, orgs  = _extract_spacy_entities(clean_text, nlp)
    if not name:
        name = _extract_name_heuristic(clean_text)

    contact = ContactInfo(
        name     = name,
        email    = email,
        phone    = phone,
        location = location,
        linkedin = linkedin,
        github   = github,
    )

    logger.info(
        f"NER complete — name: {name}, email: {bool(email)}, "
        f"phone: {bool(phone)}, orgs: {len(orgs)}, dates: {len(dates)}, "
        f"spacy: {spacy_used}"
    )

    return NERResult(
        contact        = contact,
        detected_orgs  = orgs,
        detected_dates = dates,
        spacy_used     = spacy_used,
    )

def _extract_email(text: str) -> str | None:
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match   = re.search(pattern, text)
    return match.group(0).lower() if match else None


def _extract_phone(text: str) -> str | None:
    patterns = [
        r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4}",
        r"\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}",
        r"\b\d{10}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def _extract_linkedin(text: str) -> str | None:
    patterns = [
        r"linkedin\.com/in/[\w\-]+",
        r"linkedin\.com/pub/[\w\-/]+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group(0)
            return f"https://www.{url}" if not url.startswith("http") else url
    return None


def _extract_github(text: str) -> str | None:
    pattern = r"github\.com/[\w\-]+"
    match   = re.search(pattern, text, re.IGNORECASE)
    if match:
        url = match.group(0)
        return f"https://www.{url}" if not url.startswith("http") else url
    return None


def _extract_dates(text: str) -> list[str]:
    patterns = [
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\.?\s+\d{4}",
        r"\b\d{4}[-/]\d{2}\b|\b\d{2}[-/]\d{4}\b",
        r"\b(19[9][0-9]|20[0-2][0-9]|2030)\b",
        r"\b(?:Present|Current|Now|Till\s+Date|Till\s+Now)\b",
    ]

    dates_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            continue
        if isinstance(matches[0], str):
            dates_found.extend(matches)
        else:
            dates_found.extend([m for m in matches])
    seen         = set()
    unique_dates = []
    for d in dates_found:
        d_clean = str(d).strip()
        if d_clean and d_clean not in seen:
            seen.add(d_clean)
            unique_dates.append(d_clean)

    return unique_dates

def _extract_spacy_entities(
    text: str,
    nlp,
) -> tuple[str | None, str | None, list[str]]:
    doc = nlp(text[:SPACY_TEXT_LIMIT])

    name     = None
    location = None
    orgs     = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and name is None:
            candidate_name = ent.text.strip()
            if NAME_MIN_WORDS < len(candidate_name.split()) <= NAME_MAX_WORDS:
                name = candidate_name

        elif ent.label_ == "GPE" and location is None:
            location = ent.text.strip()

        elif ent.label_ == "ORG":
            org = ent.text.strip()
            if org and org not in orgs and len(org) > 1:
                orgs.append(org)
    return name, location, orgs[:MAX_ORGS_DETECTED]

def _extract_name_heuristic(text: str) -> str | None:
    lines = [l.strip() for l in text.split("\n") if l.strip()][:NAME_HEURISTIC_LINES]

    for line in lines:
        if any(c in line for c in ["@", "http", "linkedin", "github", "+", "/"]):
            continue
        if line.isupper() and len(line.split()) == 1:
            continue
        if len(line) > NAME_MAX_LINE_LENGTH:
            continue

        if re.search(r"\d", line):
            continue

        words = line.split()
        if NAME_MIN_WORDS <= len(words) <= NAME_MAX_WORDS:
            return line
    return None