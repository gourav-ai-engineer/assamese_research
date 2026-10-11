"""Run the full model × task × language × condition grid from configs/experiment.yaml."""
import json
from pathlib import Path

import yaml

from .generation import LLM, environment_info, run_job


def load_config(experiment_path, models_path):
    cfg = yaml.safe_load(Path(experiment_path).read_text(encoding="utf-8"))
    cfg["models"] = yaml.safe_load(Path(models_path).read_text(encoding="utf-8"))["models"]
    return cfg


def jobs_for(cfg, model, data_dir):
    """Yield (eval_file, exemplar_file, condition, max_new_tokens) for one model."""
    data_dir = Path(data_dir)
    conditions = model.get("conditions") or cfg["conditions"]
    for lang in cfg["languages"]:
        names = []
        if "translation" in cfg["tasks"]:
            names += [(f"translation_en-{lang}", "translation_en-hi"), (f"translation_{lang}-en", "translation_hi-en")]
        if "qa" in cfg["tasks"]:
            names.append((f"qa_{lang}", "qa_hi"))
        if "summarization" in cfg["tasks"]:
            names.append((f"summarization_en-{lang}", "summarization_en-hi"))
        for name, hindi_name in names:
            task = name.split("_")[0]
            for cond in conditions:
                if cond == "translate_test" and task != "qa":
                    continue
                if cond == "few_shot_hindi" and lang == "hi":
                    continue
                ex = {"few_shot": name, "few_shot_hindi": hindi_name}.get(cond)
                yield (data_dir / "eval" / f"{name}.jsonl",
                       data_dir / "exemplars" / f"{ex}.jsonl" if ex else None,
                       cond, cfg["max_new_tokens"][task])


def load_mt_questions(data_dir, eval_file):
    path = Path(data_dir) / "mt" / eval_file.name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r["mt_question"] for r in map(json.loads, f) if r.get("mt_question")}


def run_grid(cfg, data_dir, out_dir, model_filter=None, limit=None):
    out_dir = Path(out_dir)
    for m in cfg["models"]:
        if model_filter and model_filter not in m["id"]:
            continue
        llm = LLM(m["id"], revision=m.get("revision"), is_chat=m.get("chat"), family=m.get("family"))
        slug = m["id"].replace("/", "__")
        meta = {"model": m, "environment": environment_info(), "protocol_version": cfg["protocol_version"]}
        (out_dir / slug).mkdir(parents=True, exist_ok=True)
        (out_dir / slug / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for eval_file, ex_file, cond, max_new in jobs_for(cfg, m, data_dir):
            mt_questions = None
            if cond == "translate_test":
                mt_questions = load_mt_questions(data_dir, eval_file)
                if mt_questions is None:
                    print(f"[{m['id']}] {eval_file.stem} translate_test: skipped (run `mt-questions` first)")
                    continue
            out = out_dir / slug / f"{eval_file.stem}__{cond}.jsonl"
            n = run_job(llm, eval_file, out, cond, ex_file, k=cfg["k_shot"], limit=limit,
                        max_new_tokens=max_new, batch_size=cfg.get("batch_size", 8), mt_questions=mt_questions)
            print(f"[{m['id']}] {eval_file.stem} {cond}: +{n}")
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def run_mt_baseline(cfg, data_dir, out_dir, backend="indictrans2", limit=None):
    """Dedicated MT system on the same FLORES items: a reference line for the LLM scores."""
    from .schema import read_jsonl
    from .translator import Translator
    tr = Translator(backend)
    for lang in cfg["languages"]:
        for name in (f"translation_en-{lang}", f"translation_{lang}-en"):
            rows = read_jsonl(Path(data_dir) / "eval" / f"{name}.jsonl")[:limit]
            try:
                preds = tr.translate([r["source"] for r in rows], rows[0]["source_lang"], rows[0]["target_lang"])
            except ValueError as e:
                print(f"skip {name}: {e}")
                continue
            out = Path(out_dir) / f"mt__{backend}" / f"{name}__zero_shot.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                for r, p in zip(rows, preds):
                    f.write(json.dumps(dict(r, prediction=p, model_id=f"mt/{backend}", condition="zero_shot"),
                                       ensure_ascii=False) + "\n")
            print(f"[mt/{backend}] {name}: {len(rows)}")


def run_mt_questions(cfg, data_dir, backend="indictrans2"):
    """Translate XorQA questions to English once; cached in data/processed/mt/."""
    from .schema import read_jsonl
    from .translator import Translator
    tr = Translator(backend)
    for lang in cfg["languages"]:
        rows = read_jsonl(Path(data_dir) / "eval" / f"qa_{lang}.jsonl")
        en = tr.translate([r["question"] for r in rows], lang, "en")
        out = Path(data_dir) / "mt" / f"qa_{lang}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r, q in zip(rows, en):
                f.write(json.dumps({"id": r["id"], "question": r["question"], "mt_question": q,
                                    "mt_backend": backend}, ensure_ascii=False) + "\n")
        print(f"[mt/{backend}] qa_{lang}: {len(rows)} questions")


def run_back_translate(pred_dir, backend="indictrans2"):
    """Fill `prediction` for translate_test outputs by translating `prediction_en` back."""
    from .translator import Translator
    tr = None
    for path in sorted(Path(pred_dir).rglob("*__translate_test.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        todo = [r for r in rows if r.get("prediction") is None]
        if not todo:
            continue
        tr = tr or Translator(backend)
        back = tr.translate([r["prediction_en"] for r in todo], "en", todo[0]["language"])
        for r, p in zip(todo, back):
            r["prediction"], r["mt_backend"] = p, backend
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[mt/{backend}] {path.parent.name}/{path.name}: {len(todo)} answers")
