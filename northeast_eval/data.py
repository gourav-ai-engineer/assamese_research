"""Convert IndicGenBench (and optionally IN22) into frozen evaluation JSONL files.

Layout produced under data/processed/:
  eval/<task>_<direction>.jsonl        sampled test items
  exemplars/<task>_<direction>.jsonl   few-shot pool (dev/train split, never test)
  manifest.json                        provenance, seed, counts
"""
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

INDICGENBENCH_URL = "https://github.com/google-research-datasets/indic-gen-bench.git"


def _load_examples(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)["examples"]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample(rows, n, seed):
    if n is None or n >= len(rows):
        return list(rows)
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def download_indicgenbench(dest):
    dest = Path(dest)
    if (dest / "flores_in").is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", INDICGENBENCH_URL, str(dest)], check=True)
    return dest


def _git_commit(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def flores_rows(src_dir, lang, direction, split):
    """direction is 'enxx' or 'xxen'."""
    a, b = ("en", lang) if direction == "enxx" else (lang, "en")
    rows = []
    for i, ex in enumerate(_load_examples(Path(src_dir) / "flores_in" / f"flores_{a}_{b}_{split}.json")):
        rows.append({
            "id": f"flores-{split}-{a}-{b}-{i:04d}", "task": "translation", "language": lang,
            "direction": f"{a}->{b}", "source_lang": a, "target_lang": b,
            "source": ex["source"].strip(), "reference": ex["target"].strip(),
        })
    return rows


def xorqa_rows(src_dir, lang, split):
    rows = []
    for i, ex in enumerate(_load_examples(Path(src_dir) / "xorqa_in" / f"xorqa_{lang}_{split}.json")):
        answers = [a["text"].strip() for a in ex.get("translated_answers", []) if a.get("text")]
        if not answers:
            continue
        rows.append({
            "id": f"xorqa-{split}-{lang}-{i:04d}", "task": "qa", "language": lang,
            "direction": f"qa-{lang}", "source_lang": lang, "target_lang": lang,
            "source": ex["context"].strip(), "context": ex["context"].strip(),
            "question": ex["question"].strip(), "reference": answers[0], "references": answers,
            "en_question": ex.get("oracle_question", ""),
            "en_answer": (ex.get("answers") or [{"text": ""}])[0]["text"],
        })
    return rows


def crosssum_rows(src_dir, lang, split, max_chars=3000):
    rows = []
    path = Path(src_dir) / "crosssum_in" / f"crosssum_english-{lang}_{split}.json"
    for i, ex in enumerate(_load_examples(path)):
        rows.append({
            "id": f"crosssum-{split}-{lang}-{i:04d}", "task": "summarization", "language": lang,
            "direction": f"en->{lang}", "source_lang": "en", "target_lang": lang,
            "source": ex["text"].strip()[:max_chars], "reference": ex["summary"].strip(),
        })
    return rows


def in22_rows(lang_code="mni_Mtei", n=None, seed=0):
    """Optional: IN22-Gen from Hugging Face (Meitei-script Manipuri). Requires `datasets` and
    accepting the dataset terms on the Hub."""
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/IN22-Gen", f"eng_Latn-{lang_code}", split="gen", trust_remote_code=True)
    src_col, tgt_col = "sentence_eng_Latn", f"sentence_{lang_code}"
    rows = []
    for i, ex in enumerate(ds):
        for a, b, s, t in (("en", lang_code, src_col, tgt_col), (lang_code, "en", tgt_col, src_col)):
            rows.append({
                "id": f"in22-{a}-{b}-{i:04d}", "task": "translation", "language": "mni",
                "direction": f"{a}->{b}", "source_lang": a, "target_lang": b, "script": "Meitei",
                "source": ex[s].strip(), "reference": ex[t].strip(),
            })
    return _sample(rows, n, seed) if n else rows


def prepare(src_dir, out_dir, languages, tasks, n_translation, n_qa, n_summ, seed, n_exemplars=50):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    manifest = {"source": INDICGENBENCH_URL, "commit": _git_commit(src_dir), "seed": seed,
                "created_utc": datetime.now(timezone.utc).isoformat(), "files": {}}

    def emit(name, test_rows, pool_rows, n):
        test = _sample(test_rows, n, seed)
        pool = _sample(pool_rows, n_exemplars, seed + 1)
        _write_jsonl(out_dir / "eval" / f"{name}.jsonl", test)
        _write_jsonl(out_dir / "exemplars" / f"{name}.jsonl", pool)
        manifest["files"][name] = {"eval": len(test), "exemplars": len(pool), "available": len(test_rows)}

    for lang in languages:
        if "translation" in tasks:
            for d in ("enxx", "xxen"):
                a, b = ("en", lang) if d == "enxx" else (lang, "en")
                emit(f"translation_{a}-{b}", flores_rows(src_dir, lang, d, "test"),
                     flores_rows(src_dir, lang, d, "dev"), n_translation)
        if "qa" in tasks:
            emit(f"qa_{lang}", xorqa_rows(src_dir, lang, "test"), xorqa_rows(src_dir, lang, "train"), n_qa)
        if "summarization" in tasks:
            emit(f"summarization_en-{lang}", crosssum_rows(src_dir, lang, "test"),
                 crosssum_rows(src_dir, lang, "train"), n_summ)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
