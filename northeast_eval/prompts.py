PROMPT_VERSION = "2026-10-10-v1"
def build_prompt(row, condition="zero_shot"):
    task, lang = row["task"], row.get("language", "as")
    if task == "translation":
        src, tgt = row.get("source_lang", "en"), row.get("target_lang", lang)
        instruction, body = f"Translate from {src} to {tgt}. Preserve meaning; add no explanations.", row["source"]
    elif task == "qa":
        instruction = f"Answer using the context in language code {lang}. Do not invent facts."
        body = f"Context: {row.get('context', row.get('source', ''))}\nQuestion: {row.get('question', '')}"
    elif task == "summarization":
        instruction, body = f"Summarize in language code {lang}. Do not add facts.", row["source"]
    else: raise ValueError(f"Unsupported task: {task}")
    if condition not in {"zero_shot", "one_shot_same_language", "one_shot_hindi"}:
        raise ValueError(f"Unsupported condition: {condition}")
    ex = row.get("example") if condition != "zero_shot" else None
    if condition != "zero_shot" and not ex: raise ValueError(f"{condition} requires a verified row['example']")
    prefix = f"Example input: {ex.get('source','')}\nExample output: {ex.get('target','')}\n\n" if ex else ""
    return f"{prefix}{instruction}\n\nInput:\n{body}\n\nOutput:"
