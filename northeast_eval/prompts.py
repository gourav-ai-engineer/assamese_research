"""Prompt construction.

Chat models get few-shot exemplars as alternating user/assistant turns (build_messages);
a single user turn holding all exemplars made chat models translate an exemplar instead
of the input. Base models (Sarvam-1) get the plain-text prompt (build_prompt) and we cut
the continuation at the first newline. Templates follow the style of IndicGenBench's task instructions; record
PROMPT_VERSION with every prediction so changes are traceable.
"""
from .languages import lang_name

PROMPT_VERSION = "2026-10-11-v3"
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


def _parts(row):
    parts = {"translation": _translation_parts, "qa": _qa_parts, "summarization": _summ_parts}
    if row["task"] not in parts:
        raise ValueError(f"Unsupported task: {row['task']}")
    return parts[row["task"]](row)


def _check(condition, exemplars):
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    if condition in ("few_shot", "few_shot_hindi") and not exemplars:
        raise ValueError(f"{condition} requires exemplars from a dev/train split")
    return [] if condition in ("zero_shot", "translate_test") else exemplars


def build_prompt(row, condition="zero_shot", exemplars=None):
    """Plain-text prompt (base models). `exemplars` come from the dev/train pool."""
    exemplars = _check(condition, exemplars)
    instruction, fmt_in, _ = _parts(row)
    # Hindi exemplars keep their own language labels so the demonstration is honest.
    shots = []
    for ex in exemplars:
        _, ex_in, ex_out = _parts(ex)
        shots.append(ex_in(ex) + ex_out(ex))
    return "\n\n".join([instruction, *shots, fmt_in(row)])


def build_messages(row, condition="zero_shot", exemplars=None):
    """Chat prompt: each exemplar is a user turn with its own instruction plus an assistant answer."""
    exemplars = _check(condition, exemplars)
    messages = []
    for ex in exemplars:
        ex_instruction, ex_in, _ = _parts(ex)
        messages += [{"role": "user", "content": f"{ex_instruction}\n\n{ex_in(ex)}"},
                     {"role": "assistant", "content": ex["reference"]}]
    instruction, fmt_in, _ = _parts(row)
    return [*messages, {"role": "user", "content": f"{instruction}\n\n{fmt_in(row)}"}]


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
