import json
import pytest
from northeast_eval.schema import read_jsonl, validate_rows

def test_read_toy():
    rows = read_jsonl("data/examples/toy_translation.jsonl")
    assert len(rows) == 1
    assert validate_rows(rows, require_reference=True) == []

def test_missing_fields(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id":"x","task":"translation","source":"hi"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(p)
