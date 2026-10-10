import json
from pathlib import Path
from collections import defaultdict
from .schema import read_jsonl
from .metrics import chrf_corpus, chrf_sentence, token_f1
def score_file(predictions_path, output_path):
    rows = read_jsonl(predictions_path)
    groups, items = defaultdict(list), []
    for row in rows:
        pred, ref = row.get("prediction"), row.get("reference")
        if not isinstance(pred, str) or not isinstance(ref, str): continue
        key = (row["task"], row["language"], row.get("direction","unspecified"),
               row.get("model_id","unknown"), row.get("condition","unknown"))
        groups[key].append((ref,pred))
        metric = "token_f1" if row["task"] == "qa" else "sentence_chrf"
        score = token_f1(ref,pred) if row["task"] == "qa" else chrf_sentence(ref,pred)
        items.append({"id":row["id"],"task":row["task"],"language":row["language"],"metric":metric,
                      "score":score,"model_id":row.get("model_id"),"condition":row.get("condition")})
    aggregates = []
    for key,pairs in groups.items():
        task,lang,direction,model,condition = key
        refs,preds = zip(*pairs)
        if task == "qa":
            metric,score = "mean_token_f1",sum(token_f1(r,p) for r,p in pairs)/len(pairs)
        else:
            metric,score = "corpus_chrf",chrf_corpus(list(refs),list(preds))
        aggregates.append({"task":task,"language":lang,"direction":direction,"model_id":model,
                           "condition":condition,"metric":metric,"score":score,"n":len(pairs)})
    result={"warning":"Scores apply only to these exact rows and recorded protocol.",
            "aggregates":aggregates,"per_item":items}
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
