"""Controlled check that script-projected chrF (a) gives Meitei- and Bengali-script outputs the
same score and (b) preserves the ranking that raw chrF gives same-script outputs.

Corrupts FLORES-IN Manipuri test references at increasing character-edit rates and compares
raw chrF, projected chrF, and raw chrF of the Meitei-transliterated version.
Usage: python scripts/validate_script_projection.py <indic-gen-bench dir> <out.json>
"""
import json
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np

from northeast_eval.data import flores_rows
from northeast_eval.metrics import chrf_corpus, spearman
from northeast_eval.script_norm import project

src, out = sys.argv[1], sys.argv[2]
refs = [r["reference"] for r in flores_rows(src, "mni", "enxx", "test")][:500]
letters = sorted({c for r in refs for c in r if unicodedata.category(c)[0] in "LM" and "ঀ" <= c <= "৿"})
rng = random.Random(0)


def corrupt(text, rate):
    chars = []
    for c in text:
        if c in letters and rng.random() < rate:
            op = rng.random()
            if op < 0.4:
                continue                      # deletion
            chars.append(rng.choice(letters))  # substitution
            if op > 0.8:
                chars.append(rng.choice(letters))  # insertion
        else:
            chars.append(c)
    return "".join(chars)


proj_refs = [project(r, "mni") for r in refs]
rows = []
for rate in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    hyp_beng = [corrupt(r, rate) for r in refs]
    hyp_mtei = [project(h, "mni") for h in hyp_beng]   # same output, written in Meitei Mayek
    rows.append({
        "edit_rate": rate,
        "raw_chrf_bengali_output": chrf_corpus(refs, hyp_beng),
        "raw_chrf_meitei_output": chrf_corpus(refs, hyp_mtei),
        "projected_chrf_bengali_output": chrf_corpus(proj_refs, [project(h, "mni") for h in hyp_beng]),
        "projected_chrf_meitei_output": chrf_corpus(proj_refs, [project(h, "mni") for h in hyp_mtei]),
    })

raw = [r["raw_chrf_bengali_output"] for r in rows]
proj = [r["projected_chrf_bengali_output"] for r in rows]
summary = {
    "n_sentences": len(refs),
    "spearman_raw_vs_projected": spearman(raw, proj),
    "pearson_raw_vs_projected": float(np.corrcoef(raw, proj)[0, 1]),
    "max_abs_gap_meitei_vs_bengali_projected": max(
        abs(r["projected_chrf_bengali_output"] - r["projected_chrf_meitei_output"]) for r in rows),
    "rows": rows,
}
Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
for r in rows:
    print("  ".join(f"{v:6.2f}" for v in r.values()))
