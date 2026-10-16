"""Scoring, automatic error signals, tables and plots."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .languages import lang_script
from .metrics import (
    chrf_bootstrap,
    chrf_corpus,
    chrf_sentence,
    max_token_f1,
    mean_bootstrap,
    spearman,
    token_f1,
)
from .schema import read_jsonl
from .script_norm import project

ERROR_TAGS = ["wrong_language_or_script", "missing_information", "hallucinated_content",
              "wrong_word_sense", "grammar_or_inflection", "untranslated_copy", "other"]


def target_script(row):
    if row.get("script") and row["target_lang"] != "en":
        return row["script"]
    return lang_script(row["target_lang"])


def auto_flags(row):
    """Cheap, automatic error signals. They complement, never replace, manual annotation."""
    from .languages import script_match
    pred = row.get("prediction") or ""
    flags = {"empty": not pred.strip()}
    flags["wrong_script"] = bool(pred.strip()) and not script_match(pred, target_script(row))
    flags["source_copy"] = row["task"] == "translation" and bool(pred) and chrf_sentence(row["source"], pred) > 80
    ref_len = max(len(row.get("reference", "")), 1)
    flags["overlong"] = len(pred) > 3 * ref_len and row["task"] != "summarization"
    return flags


def _iter_prediction_files(pred_dir):
    return sorted(Path(pred_dir).rglob("*.jsonl"))


def _rate(flags, key):
    return sum(f[key] for f in flags) / len(flags)


HINDI_CLUSTER = {"hi", "awa", "bgc", "bho", "hne", "mai", "mup", "mwr", "gbm", "hoj"}


def lid_group(pred_lang, target):
    """Collapse a language-ID label into the categories reported in the paper."""
    if pred_lang == target:
        return "on_target"
    if target == "mni" and pred_lang == "mni_Mtei":
        return "on_target_meitei_script"
    if pred_lang in HINDI_CLUSTER:
        return "hindi_cluster"
    if pred_lang == "en":
        return "english_or_latin"
    if pred_lang in {"bn", "as", "mni", "mni_Mtei", "brx"}:
        return "other_northeast_or_bengali"
    return "other_indic"


def language_breakdown(lid, preds, target):
    """Share of outputs per lid_group; empty outputs are counted as their own category."""
    from collections import Counter
    texts = [p for p in preds if p.strip()]
    groups = Counter(lid_group(str(l), target) for l in lid.predict(texts)) if texts else Counter()
    groups["empty"] = len(preds) - len(texts)
    return {k: round(v / len(preds), 3) for k, v in groups.most_common() if v}


def score_rows(rows, n_resamples=1000, lid=None):
    groups = defaultdict(list)
    for row in rows:
        if isinstance(row.get("prediction"), str) and isinstance(row.get("reference"), str):
            key = (row.get("model_id", "unknown"), row["task"], row["language"], row.get("direction", ""),
                   row.get("condition", "unknown"))
            groups[key].append(row)
    out = []
    for (model, task, lang, direction, condition), items in sorted(groups.items()):
        refs = [r["reference"] for r in items]
        preds = [r["prediction"] for r in items]
        if task == "qa":
            vals = [max_token_f1(r.get("references") or [r["reference"]], r["prediction"]) for r in items]
            metric, (score, lo, hi) = "token_f1", mean_bootstrap([100 * v for v in vals], n_resamples)
        else:
            metric, (score, lo, hi) = "chrf", chrf_bootstrap(refs, preds, n_resamples)
        flags = [auto_flags(r) for r in items]
        target = items[0]["target_lang"]
        extra = {}
        if task != "qa":
            # Script-projected chrF: identical to chrF except for languages with two scripts (mni).
            extra["chrf_script_projected"] = round(chrf_corpus([project(r, target) for r in refs],
                                                               [project(p, target) for p in preds]), 2)
        if lid is not None and task == "translation":
            breakdown = language_breakdown(lid, preds, target)
            on = breakdown.get("on_target", 0) + breakdown.get("on_target_meitei_script", 0)
            extra["off_target_rate"] = round(1 - on - breakdown.get("empty", 0), 3)
            extra["language_breakdown"] = json.dumps(breakdown)
        out.append({"model_id": model, "task": task, "language": lang, "direction": direction,
                    "condition": condition, "metric": metric, "score": round(score, 2),
                    "ci_low": round(lo, 2), "ci_high": round(hi, 2), "n": len(items),
                    "chrfpp": round(chrf_corpus(refs, preds, plus=True), 2) if task != "qa" else "",
                    "empty_rate": round(_rate(flags, "empty"), 3), "wrong_script_rate": round(_rate(flags, "wrong_script"), 3),
                    "source_copy_rate": round(_rate(flags, "source_copy"), 3), "overlong_rate": round(_rate(flags, "overlong"), 3), **extra})
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))  # rows may carry different extras
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_dir(pred_dir, output_csv, n_resamples=1000, lid=None):
    rows = [r for p in _iter_prediction_files(pred_dir) for r in read_jsonl(p)]
    scores = score_rows(rows, n_resamples, lid)
    write_csv(output_csv, scores)
    return scores


def score_file(predictions_path, output_path):
    """Legacy single-file scorer kept for the `score` CLI command."""
    rows = read_jsonl(predictions_path)
    aggregates = score_rows(rows, n_resamples=200)
    per_item = []
    for row in rows:
        if isinstance(row.get("prediction"), str) and isinstance(row.get("reference"), str):
            qa = row["task"] == "qa"
            per_item.append({"id": row["id"], "metric": "token_f1" if qa else "sentence_chrf",
                             "score": token_f1(row["reference"], row["prediction"]) if qa
                             else chrf_sentence(row["reference"], row["prediction"]),
                             **auto_flags(row)})
    result = {"warning": "Scores apply only to these exact rows and recorded protocol.",
              "aggregates": aggregates, "per_item": per_item}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def short_name(model_id):
    return model_id.split("/")[-1]


def pivot_table(scores, task, condition, fmt="md"):
    """Rows: models; columns: language/direction."""
    sel = [s for s in scores if s["task"] == task and s["condition"] == condition]
    cols = sorted({s["direction"] for s in sel})
    models = sorted({s["model_id"] for s in sel})
    cell = {(s["model_id"], s["direction"]): f"{float(s['score']):.1f}" for s in sel}
    header = ["Model", *cols]
    body = [[short_name(m), *[cell.get((m, c), "–") for c in cols]] for m in models]
    if fmt == "latex":
        lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", " & ".join(header) + r" \\",
                 r"\midrule", *[" & ".join(r) + r" \\" for r in body], r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines).replace("->", r"$\rightarrow$").replace("_", r"\_")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def write_tables(scores, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Results tables (auto-generated)\n"]
    for task in sorted({s["task"] for s in scores}):
        for cond in sorted({s["condition"] for s in scores if s["task"] == task}):
            md += [f"## {task} — {cond}\n", pivot_table(scores, task, cond), ""]
            (out_dir / f"{task}_{cond}.tex").write_text(pivot_table(scores, task, cond, "latex"), encoding="utf-8")
    diag = diagnostics_table(scores)
    if diag:
        md += ["## Diagnostics: script penalty and language confusion (en→xx, zero-shot)\n", diag, ""]
    (out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")


def diagnostics_table(scores, condition="zero_shot"):
    """chrF vs script-projected chrF and off-target rate for translation into each language."""
    sel = [s for s in scores if s["task"] == "translation" and s["condition"] == condition
           and s["direction"].startswith("en->") and s.get("chrf_script_projected", "") != ""]
    if not sel:
        return ""
    lines = ["| Model | Direction | chrF | chrF (script-projected) | Script penalty | Off-target | Breakdown |",
             "|---|---|---|---|---|---|---|"]
    for s in sorted(sel, key=lambda x: (x["direction"], x["model_id"])):
        raw, proj = float(s["score"]), float(s["chrf_script_projected"])
        lines.append(f"| {short_name(s['model_id'])} | {s['direction']} | {raw:.1f} | {proj:.1f} | "
                     f"{proj - raw:+.1f} | {s.get('off_target_rate', '–')} | {s.get('language_breakdown', '')} |")
    return "\n".join(lines)


def make_plots(scores, out_dir, fertility_csv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

    # 1. chrF per direction and model (best condition per cell).
    trans = [s for s in scores if s["task"] == "translation"]
    if trans:
        best = {}
        for s in trans:
            k = (s["model_id"], s["direction"])
            if k not in best or float(s["score"]) > float(best[k]["score"]):
                best[k] = s
        dirs = sorted({k[1] for k in best})
        models = sorted({k[0] for k in best})
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(dirs)), 3.6))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            ys = [float(best[(m, d)]["score"]) if (m, d) in best else 0 for d in dirs]
            ax.bar([x + i * w for x in range(len(dirs))], ys, w, label=short_name(m), color=palette[i % 8])
        ax.set_xticks([x + w * (len(models) - 1) / 2 for x in range(len(dirs))], dirs, rotation=30)
        ax.set_ylabel("chrF (best condition)")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "translation_chrf.pdf")
        fig.savefig(out_dir / "translation_chrf.png", dpi=200)
        plt.close(fig)

    # 2. Fertility vs en->xx chrF.
    if fertility_csv and Path(fertility_csv).exists():
        fert = {(r["model_id"], r["language"]): float(r["fertility"]) for r in read_csv(fertility_csv)}
        pts = [(fert[(s["model_id"], s["language"])], float(s["score"]), s["language"]) for s in trans
               if s["direction"].startswith("en->") and s["condition"] == "zero_shot"
               and (s["model_id"], s["language"]) in fert]
        if pts:
            fig, ax = plt.subplots(figsize=(4.5, 3.4))
            langs = sorted({p[2] for p in pts})
            for i, lg in enumerate(langs):
                xs = [p[0] for p in pts if p[2] == lg]
                ys = [p[1] for p in pts if p[2] == lg]
                ax.scatter(xs, ys, label=lg, color=palette[i % 8])
            rho = spearman([p[0] for p in pts], [p[1] for p in pts])
            ax.set_xlabel("Tokenizer fertility (tokens/word)")
            ax.set_ylabel("chrF, en→xx zero-shot")
            ax.set_title(f"Spearman ρ = {rho:.2f}", fontsize=9)
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(out_dir / "fertility_vs_chrf.pdf")
            fig.savefig(out_dir / "fertility_vs_chrf.png", dpi=200)
            plt.close(fig)

    # 3. Wrong-script rate per model × target language (en->xx, zero-shot).
    ws = [s for s in trans if s["direction"].startswith("en->") and s["condition"] == "zero_shot"]
    if ws:
        models = sorted({s["model_id"] for s in ws})
        langs = sorted({s["direction"] for s in ws})
        grid = [[next((float(s["wrong_script_rate"]) for s in ws if s["model_id"] == m and s["direction"] == d),
                      float("nan")) for d in langs] for m in models]
        fig, ax = plt.subplots(figsize=(1.2 * len(langs) + 2, 0.45 * len(models) + 1.2))
        im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(langs)), langs)
        ax.set_yticks(range(len(models)), [short_name(m) for m in models])
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="wrong-script rate")
        fig.tight_layout()
        fig.savefig(out_dir / "wrong_script.pdf")
        fig.savefig(out_dir / "wrong_script.png", dpi=200)
        plt.close(fig)


def sample_for_annotation(pred_dir, output_csv, per_language=50, seed=20261010, condition="zero_shot"):
    """Blind random sample of en->xx outputs for manual error taxonomy annotation."""
    rows = [r for p in _iter_prediction_files(pred_dir) for r in read_jsonl(p)
            if r.get("condition") == condition and r["task"] == "translation" and r["source_lang"] == "en"]
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    rng = random.Random(seed)
    out = []
    for lang, items in sorted(by_lang.items()):
        for r in rng.sample(items, min(per_language, len(items))):
            out.append({"blind_id": "", "language": lang, "source": r["source"], "reference": r["reference"],
                        "prediction": r["prediction"], "_model_id": r["model_id"], "_item_id": r["id"],
                        **{f"auto_{k}": v for k, v in auto_flags(r).items()},
                        "adequacy_1_5": "", "fluency_1_5": "", **{t: "" for t in ERROR_TAGS}, "notes": ""})
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["blind_id"] = f"S{i:04d}"
    # Annotators get the sheet without model identity; the key is kept separately.
    key = [{"blind_id": r["blind_id"], "model_id": r.pop("_model_id"), "item_id": r.pop("_item_id")} for r in out]
    write_csv(output_csv, out)
    write_csv(Path(output_csv).with_name(Path(output_csv).stem + "_key.csv"), key)
    return len(out)
