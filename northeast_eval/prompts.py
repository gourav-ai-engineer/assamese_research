"""Prompt construction.

The same plain-text prompt is used for chat and base models: chat models receive it as a
single user turn, base models (Sarvam-1) receive it raw and we cut the continuation at the
first newline. Templates follow the style of IndicGenBench's task instructions; record
PROMPT_VERSION with every prediction so changes are traceable.
"""
from .languages import lang_name

PROMPT_VERSION = "2026-10-10-v2"
CONDITIONS = ("zero_shot", "few_shot", "few_shot_hindi", "translate_test")


def _translation_parts(row):
    src, tgt = lang_name(row["source_lang"]), lang_name(row["target_lang"])
    instruction = f"Translate the following text from {src} to {tgt}. Reply with only the {tgt} translation."
    return instruction, lambda r: f"{src}: {r['source']}\n{tgt}:", lambda r: f" {r['reference']}"


def _qa_parts(row, answer_lang=None):
    lang = lang_name(answer_lang or row["language"])
    instruction = (f"Answer the question using the English context. Reply with only a short answer "
                   f"in {lang}, without explanation.")
    fmt = lambda r: f"Context: {r['context']}\nQuestion: {r['question']}\nAnswer:"
    return instruction, fmt, lambda r: f" {r['reference']}"


def _summ_parts(row):
    lang = lang_name(row["target_lang"])
    instruction = f"Summarize the following English article in one or two sentences in {lang}."
    return instruction, lambda r: f"Article: {r['source']}\nSummary in {lang}:", lambda r: f" {r['reference']}"


def build_prompt(row, condition="zero_shot", exemplars=None):
    """Return the prompt text. `exemplars` are rows from the exemplar pool (dev/train split)."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    task = row["task"]
    if task == "translation":
        instruction, fmt_in, _ = _translation_parts(row)
    elif task == "qa":
        instruction, fmt_in, _ = _qa_parts(row)
    elif task == "summarization":
        instruction, fmt_in, _ = _summ_parts(row)
    else:
        raise ValueError(f"Unsupported task: {task}")
    if condition in ("few_shot", "few_shot_hindi") and not exemplars:
        raise ValueError(f"{condition} requires exemplars from a dev/train split")
    if condition in ("zero_shot", "translate_test"):
        exemplars = []
    shots = []
    for ex in exemplars or []:
        # Hindi exemplars keep their own language labels so the demonstration is honest.
        if ex["task"] == "translation":
            _, ex_in, ex_out = _translation_parts(ex)
        elif ex["task"] == "qa":
            _, ex_in, ex_out = _qa_parts(ex)
        else:
            _, ex_in, ex_out = _summ_parts(ex)
        shots.append(ex_in(ex) + ex_out(ex))
    blocks = [instruction, *shots, fmt_in(row)]
    return "\n\n".join(blocks)


def english_qa_row(row, en_question):
    """Row used in translate-test QA: English question, answer expected in English."""
    return dict(row, question=en_question, language="en", reference=row.get("en_answer", ""))


def clean_output(text, task):
    """Keep the first non-empty line and strip echoed labels/quotes."""
    text = text.replace("<think>", "").replace("</think>", "").strip()
    if task != "summarization":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = lines[0] if lines else ""
    for label in ("Answer:", "Translation:", "Summary:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    if ":" in text[:25] and text.split(":", 1)[0].strip() in {
            "Assamese", "Manipuri", "Bodo", "Hindi", "English"}:
        text = text.split(":", 1)[1].strip()
    return text.strip().strip('"“”').strip()
