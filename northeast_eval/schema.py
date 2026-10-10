from __future__ import annotations
import json
from pathlib import Path
from typing import Any

REQUIRED = {"id", "task", "language", "source"}
VALID_LANGUAGES = {"as", "mni", "brx", "hi"}
VALID_TASKS = {"translation", "qa", "summarization"}

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL line must be an object")
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            if row["task"] not in VALID_TASKS:
                raise ValueError(f"{path}:{line_no}: task must be one of {sorted(VALID_TASKS)}")
            if row["language"] not in VALID_LANGUAGES:
                raise ValueError(f"{path}:{line_no}: language must be one of {sorted(VALID_LANGUAGES)}")
            if not isinstance(row["source"], str):
                raise ValueError(f"{path}:{line_no}: source must be a string")
            if row["task"] == "translation" and not isinstance(row.get("reference"), str):
                # Prediction inputs may omit reference only for generation, scoring inputs may not.
                pass
            rows.append(row)
    return rows

def validate_rows(rows: list[dict[str, Any]], require_reference: bool = False) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        rid = str(row.get("id", ""))
        if rid in seen:
            errors.append(f"row {i+1}: duplicate id {rid!r}")
        seen.add(rid)
        if require_reference and not isinstance(row.get("reference"), str):
            errors.append(f"row {i+1} ({rid}): missing string reference")
        if row.get("task") == "qa" and not (row.get("question") or row.get("source")):
            errors.append(f"row {i+1} ({rid}): QA needs question or source")
    return errors
