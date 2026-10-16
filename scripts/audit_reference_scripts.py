"""Audit the script of IndicGenBench reference text for the study languages.
Usage: python scripts/audit_reference_scripts.py <indic-gen-bench dir> <out.json>
"""
import json
import sys
from collections import Counter
from pathlib import Path

from northeast_eval.languages import script_profile

src, out = Path(sys.argv[1]), sys.argv[2]
FIELDS = {"flores_in": ("flores_en_{l}_{s}.json", "target", ("dev", "test")),
          "xorqa_in": ("xorqa_{l}_{s}.json", "translated_answers", ("train", "dev", "test")),
          "crosssum_in": ("crosssum_english-{l}_{s}.json", "summary", ("train", "dev", "test"))}
report = {}
for lang in ["as", "mni", "brx", "hi"]:
    for task, (pattern, field, splits) in FIELDS.items():
        letters, majority = Counter(), Counter()
        n = 0
        for split in splits:
            path = src / task / pattern.format(l=lang, s=split)
            if not path.exists():
                continue
            for ex in json.loads(path.read_text(encoding="utf-8"))["examples"]:
                text = ex[field][0]["text"] if field == "translated_answers" else ex[field]
                prof = script_profile(text)
                if not prof:
                    continue
                n += 1
                majority[max(prof, key=prof.get)] += 1
                for k, v in prof.items():
                    letters[k] += v
        report[f"{lang}/{task}"] = {
            "items": n,
            "majority_script_share": {k: round(v / n, 4) for k, v in majority.most_common()},
            "mean_letter_share": {k: round(v / n, 4) for k, v in letters.most_common()},
        }
Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
for k, v in report.items():
    print(f"{k:22s} n={v['items']:5d}  majority={v['majority_script_share']}")
