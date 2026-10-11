"""End-to-end pipeline test with a fake model and synthetic IndicGenBench-shaped files."""
import json

from northeast_eval import generation
from northeast_eval.data import prepare
from northeast_eval.reporting import (
    make_plots,
    sample_for_annotation,
    score_dir,
    write_tables,
)
from northeast_eval.runner import jobs_for


def _write_bench(root):
    def dump(path, examples):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"canary": "x", "examples": examples}, ensure_ascii=False), encoding="utf-8")
    for split in ("dev", "test"):
        dump(root / "flores_in" / f"flores_en_as_{split}.json",
             [{"source": f"Sentence {i}.", "target": f"বাক্য {i}।"} for i in range(20)])
        dump(root / "flores_in" / f"flores_as_en_{split}.json",
             [{"source": f"বাক্য {i}।", "target": f"Sentence {i}."} for i in range(20)])
    for split in ("train", "test"):
        dump(root / "xorqa_in" / f"xorqa_as_{split}.json",
             [{"context": "Ctx", "question": f"প্ৰশ্ন {i}?", "oracle_question": f"Question {i}?",
               "answers": [{"text": "Ans"}], "translated_answers": [{"text": f"উত্তৰ {i}"}]} for i in range(10)])


class FakeLLM:
    model_id, revision = "fake/echo", None

    def generate(self, prompts, max_new_tokens=128, batch_size=8):
        # Echo the input after the last label, which yields partially correct outputs.
        return [{"text": p.rsplit(":", 2)[-2].split("\n")[0], "input_tokens": 1, "output_tokens": 1}
                for p in prompts]


def test_end_to_end(tmp_path):
    bench, data, preds, res = tmp_path / "bench", tmp_path / "data", tmp_path / "preds", tmp_path / "res"
    _write_bench(bench)
    m = prepare(bench, data, ["as"], ["translation", "qa"], 8, 5, 0, seed=1, n_exemplars=5)
    assert m["files"]["translation_en-as"]["eval"] == 8
    cfg = {"languages": ["as"], "tasks": ["translation", "qa"], "conditions": ["zero_shot", "few_shot"],
           "max_new_tokens": {"translation": 16, "qa": 8}}
    jobs = list(jobs_for(cfg, {"id": "fake"}, data))
    assert len(jobs) == 6
    for eval_file, ex_file, cond, max_new in jobs:
        out = preds / "fake" / f"{eval_file.stem}__{cond}.jsonl"
        assert generation.run_job(FakeLLM(), eval_file, out, cond, ex_file, k=2, max_new_tokens=max_new) > 0
        assert generation.run_job(FakeLLM(), eval_file, out, cond, ex_file, k=2) == 0  # resume
    scores = score_dir(preds, res / "scores.csv", n_resamples=50)
    assert {s["task"] for s in scores} == {"translation", "qa"}
    write_tables(scores, res / "tables")
    make_plots(scores, res / "figures")
    assert (res / "tables" / "tables.md").exists() and (res / "figures" / "translation_chrf.png").exists()
    assert sample_for_annotation(preds, res / "ann.csv", per_language=5) == 5
    assert "fake" not in (res / "ann.csv").read_text(encoding="utf-8")


def test_translate_test_flow(tmp_path, monkeypatch):
    from northeast_eval import translator
    from northeast_eval.runner import load_mt_questions, run_back_translate
    from northeast_eval.schema import read_jsonl

    bench, data, preds = tmp_path / "bench", tmp_path / "data", tmp_path / "preds"
    _write_bench(bench)
    prepare(bench, data, ["as"], ["qa"], 0, 3, 0, seed=1, n_exemplars=2)
    eval_file = data / "eval" / "qa_as.jsonl"
    assert load_mt_questions(data, eval_file) is None
    rows = read_jsonl(eval_file)
    (data / "mt").mkdir()
    (data / "mt" / "qa_as.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "mt_question": "Question?"}) + "\n" for r in rows), encoding="utf-8")
    mt = load_mt_questions(data, eval_file)
    out = preds / "fake" / "qa_as__translate_test.jsonl"
    assert generation.run_job(FakeLLM(), eval_file, out, "translate_test", mt_questions=mt) == 3
    assert all(r["prediction"] is None and "Question?" in r["prompt"] for r in read_jsonl(out))

    class FakeTranslator:
        def __init__(self, backend):
            pass

        def translate(self, texts, src, tgt):
            return [f"{tgt}:{t}" for t in texts]

    monkeypatch.setattr(translator, "Translator", FakeTranslator)
    run_back_translate(preds)
    assert all(r["prediction"].startswith("as:") for r in read_jsonl(out))
    assert score_dir(preds, tmp_path / "s.csv", n_resamples=10)[0]["condition"] == "translate_test"
