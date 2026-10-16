"""How much chrF does a fluent translation in the WRONG language earn?

FLORES-IN sentences are parallel across languages (aligned here by their English source).
We score the human reference of a related language against the target-language reference,
e.g. the Bengali reference as a "prediction" for Assamese. This is an upper bound on what a
model that confuses the two languages, but translates perfectly, would receive from chrF.
Usage: python scripts/wrong_language_baseline.py <indic-gen-bench dir> <out.json>
"""
import json
import sys
from pathlib import Path

from northeast_eval.metrics import chrf_corpus
from northeast_eval.script_norm import project

src, out = Path(sys.argv[1]), sys.argv[2]


def refs(lang):
    ex = json.loads((src / "flores_in" / f"flores_en_{lang}_test.json").read_text(encoding="utf-8"))["examples"]
    return {" ".join(e["source"].split()): e["target"].strip() for e in ex}


PAIRS = {"as": ["bn", "or", "mni", "hi"], "brx": ["hi", "ne", "mr", "mai"],
         "mni": ["as", "bn", "hi"], "hi": ["mai", "bho", "mr", "ne"]}
data = {l: refs(l) for l in set(PAIRS) | {x for v in PAIRS.values() for x in v}}
results = []
for target, confusers in PAIRS.items():
    for wrong in confusers:
        keys = sorted(set(data[target]) & set(data[wrong]))
        r = [data[target][k] for k in keys]
        h = [data[wrong][k] for k in keys]
        row = {"target": target, "wrong_language_output": wrong, "aligned_sentences": len(keys),
               "chrf": round(chrf_corpus(r, h), 2)}
        if target == "mni":
            row["chrf_script_projected"] = round(chrf_corpus([project(x, "mni") for x in r],
                                                             [project(x, "mni") for x in h]), 2)
        results.append(row)
        print(row)
Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
