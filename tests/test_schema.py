import json

import pytest

from northeast_eval.schema import read_jsonl, validate_rows


def test_schema(tmp_path):
    p=tmp_path/"data.jsonl"
    p.write_text(json.dumps({"id":"1","task":"translation","language":"as","source":"hi","reference":"x"})+"\n",encoding="utf-8")
    rows=read_jsonl(p)
    assert validate_rows(rows,require_reference=True)==[]
def test_missing_fields(tmp_path):
    p=tmp_path/"bad.jsonl"; p.write_text('{"id":"x"}\n',encoding="utf-8")
    with pytest.raises(ValueError): read_jsonl(p)
