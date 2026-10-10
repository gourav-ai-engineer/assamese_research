"""Tokenizer fertility (tokens per word) on FLORES target-language sentences."""
import csv
from pathlib import Path

from .metrics import tokenizer_fertility
from .schema import read_jsonl


def texts_for_language(eval_dir, lang):
    rows = read_jsonl(Path(eval_dir) / f"translation_en-{lang}.jsonl")
    return [r["reference"] for r in rows]


def fertility_for(tokenizer, texts):
    tokens = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
    words = sum(len(t.split()) for t in texts)
    chars = sum(len(t) for t in texts)
    return {"fertility": tokenizer_fertility(tokens, words), "chars_per_token": chars / max(tokens, 1),
            "tokens": tokens, "words": words}


def run(model_ids, languages, eval_dir, output_csv):
    from transformers import AutoTokenizer
    out_rows = []
    texts = {lang: texts_for_language(eval_dir, lang) for lang in languages}
    texts["en"] = [r["source"] for r in read_jsonl(Path(eval_dir) / f"translation_en-{languages[0]}.jsonl")]
    for mid in model_ids:
        tok = AutoTokenizer.from_pretrained(mid)
        for lang, t in texts.items():
            out_rows.append({"model_id": mid, "language": lang, **fertility_for(tok, t)})
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    return out_rows
