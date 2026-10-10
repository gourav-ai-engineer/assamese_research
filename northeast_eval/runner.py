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


def run_grid(cfg, data_dir, out_dir, model_filter=None, limit=None, translator_backend=None):
    out_dir = Path(out_dir)
    translator = None
    for m in cfg["models"]:
        if model_filter and model_filter not in m["id"]:
            continue
        llm = LLM(m["id"], revision=m.get("revision"), is_chat=m.get("chat"), family=m.get("family"))
        slug = m["id"].replace("/", "__")
        meta = {"model": m, "environment": environment_info(), "protocol_version": cfg["protocol_version"]}
        (out_dir / slug).mkdir(parents=True, exist_ok=True)
        (out_dir / slug / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for eval_file, ex_file, cond, max_new in jobs_for(cfg, m, data_dir):
            if cond == "translate_test" and translator is None:
                from .translator import Translator
                translator = Translator(translator_backend or cfg.get("translator", "indictrans2"))
            out = out_dir / slug / f"{eval_file.stem}__{cond}.jsonl"
            n = run_job(llm, eval_file, out, cond, ex_file, k=cfg["k_shot"], limit=limit,
                        max_new_tokens=max_new, batch_size=cfg.get("batch_size", 8), translator=translator)
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
