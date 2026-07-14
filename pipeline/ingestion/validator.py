import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


from config.pipeline_config import (
    MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS, MAGIC_BYTES, OCR_FALLBACK_THRESHOLD
) 
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\s\-\.]+$")

@dataclass
class ValidationResult:
    is_valid:       bool
    file_path:      Path
    file_name:      str
    extension:      str
    file_size_bytes: int
    file_size_mb:   float
    warnings:       list[str]
    error:          str | None = None  

def validate_file(file_path: str | Path) -> ValidationResult:
    path     = Path(file_path)
    warnings = []

    if not path.exists():
        return _fail(path, f"File does not exist: {path}")

    if not path.is_file():
        return _fail(path, f"Path is not a file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        return _fail(path, "File is empty (0 bytes).")

    file_size_mb = file_size / (1024 * 1024)

    if file_size > MAX_FILE_SIZE_BYTES:
        return _fail(
            path,
            f"File too large: {file_size_mb:.1f} MB. "
            f"Maximum allowed is {MAX_FILE_SIZE_MB} MB."
        )

    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return _fail(
            path,
            f"File type '{extension}' is not allowed. "
            f"Accepted types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    magic_check = _check_magic_bytes(path, extension)
    if magic_check is not None:
        return _fail(path, magic_check)
    filename_warning = _check_filename(path.name)
    if filename_warning:
        warnings.append(filename_warning)

    logger.info(
        f"File validated: {path.name} | "
        f"{file_size_mb:.2f} MB | {extension}"
    )

    return ValidationResult(
        is_valid=True,
        file_path=path,
        file_name=path.name,
        extension=extension,
        file_size_bytes=file_size,
        file_size_mb=round(file_size_mb, 2),
        warnings=warnings,
        error=None,
    )

def _check_magic_bytes(path: Path, extension: str) -> str | None:
    expected_signatures = MAGIC_BYTES.get(extension)

    if expected_signatures is None:
        return None

    try:
        with open(path, "rb") as f:
            header = f.read(8)
    except OSError as e:
        return f"Could not read file header: {e}"

    for signature in expected_signatures:
        if header.startswith(signature):
            return None 

    return (
        f"File content does not match its extension '{extension}'. "
        f"The file may have been renamed or is corrupted. "
        f"File header (hex): {header.hex()}"
    )


def _check_filename(filename: str) -> str | None:
    if ".." in filename or "/" in filename or "\\" in filename:
        return f"Suspicious filename detected (path traversal attempt?): {filename}"

    stem = Path(filename).stem
    if not SAFE_FILENAME_PATTERN.match(stem):
        return (
            f"Filename contains unusual characters: '{filename}'. "
            "Will be sanitized before storage."
        )

    return None


def _fail(path: Path, error: str) -> ValidationResult:
    logger.warning(f"Validation failed for {path.name}: {error}")
    return ValidationResult(
        is_valid=False,
        file_path=path,
        file_name=path.name,
        extension=path.suffix.lower(),
        file_size_bytes=path.stat().st_size if path.exists() else 0,
        file_size_mb=0.0,
        warnings=[],
        error=error,
    )

def validate_batch(file_paths: list[str | Path]) -> dict:

    passed = []
    failed = []

    for fp in file_paths:
        result = validate_file(fp)
        if result.is_valid:
            passed.append(result)
        else:
            failed.append(result)

    logger.info(
        f"Batch validation: {len(passed)} passed, {len(failed)} failed "
        f"out of {len(file_paths)} files."
    )

    return {"passed": passed, "failed": failed}
