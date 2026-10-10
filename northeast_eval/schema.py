import json
from pathlib import Path

REQUIRED = {"id", "task", "language", "source"}
LANGUAGES = {"as", "mni", "brx", "hi"}
TASKS = {"translation", "qa", "summarization"}
def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: invalid JSON") from e
            if not isinstance(row, dict): raise ValueError(f"{path}:{n}: expected JSON object")
            missing = REQUIRED - row.keys()
            if missing: raise ValueError(f"{path}:{n}: missing {sorted(missing)}")
            if row["task"] not in TASKS: raise ValueError(f"{path}:{n}: invalid task")
            if row["language"] not in LANGUAGES: raise ValueError(f"{path}:{n}: invalid language")
            if not isinstance(row["source"], str): raise ValueError(f"{path}:{n}: source must be string")
            rows.append(row)
    return rows
def validate_rows(rows, require_reference=False):
    errors, seen = [], set()
    for i, row in enumerate(rows):
        rid = str(row.get("id", ""))
        if rid in seen: errors.append(f"row {i+1}: duplicate id {rid!r}")
        seen.add(rid)
        if require_reference and not isinstance(row.get("reference"), str):
            errors.append(f"row {i+1} ({rid}): missing string reference")
    return errors
