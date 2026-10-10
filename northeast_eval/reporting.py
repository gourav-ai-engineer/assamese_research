from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from .schema import read_jsonl
from .metrics import chrf_corpus, chrf_sentence, token_f1

def score_file(predictions_path: str, output_path: str) -> dict:
    rows = read_jsonl(predictions_path)
    groups = defaultdict(list)
    per_item = []
    for row in rows:
        pred = row.get("prediction")
        ref = row.get("reference")
        if not isinstance(pred, str) or not isinstance(ref, str):
            continue
        key = (row["task"], row["language"], row.get("direction", "unspecified"),
               row.get("model_id", "unknown"), row.get("condition", "unknown"))
        groups[key].append((ref, pred))
        if row["task"] == "qa":
            score = token_f1(ref, pred)
            metric = "token_f1"
        else:
            score = chrf_sentence(ref, pred)
            metric = "sentence_chrf"
        per_item.append({"id": row["id"], "task": row["task"], "language": row["language"],
                         "metric": metric, "score": score, "model_id": row.get("model_id"),
                         "condition": row.get("condition")})
    aggregates = []
    for key, pairs in groups.items():
        task, lang, direction, model, condition = key
        refs, preds = zip(*pairs)
        if task == "qa":
            score = sum(token_f1(r, p) for r, p in pairs) / len(pairs)
            metric = "mean_token_f1"
        else:
            score = chrf_corpus(list(refs), list(preds))
            metric = "corpus_chrf"
        aggregates.append({"task": task, "language": lang, "direction": direction,
                           "model_id": model, "condition": condition, "metric": metric,
                           "score": score, "n": len(pairs)})
    result = {"warning": "Scores are valid only for the exact input rows and protocol recorded.",
              "aggregates": aggregates, "per_item": per_item}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
