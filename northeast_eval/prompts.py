"""Prompt templates; version and cite them in every experiment record."""
PROMPT_VERSION = "2026-10-10-v1"

def build_prompt(row: dict, condition: str = "zero_shot") -> str:
    task = row["task"]
    language = row.get("language", "as")
    if task == "translation":
        direction = row.get("direction", "en->" + language)
        src = row.get("source_lang", "en")
        tgt = row.get("target_lang", language)
        instruction = f"Translate the following text from {src} to {tgt}. Preserve meaning and do not add explanations."
        body = row["source"]
    elif task == "qa":
        instruction = f"Answer the question using the supplied context. Answer in language code {language}. If the context does not contain the answer, say so."
        body = f"Context: {row.get('context', row.get('source', ''))}\nQuestion: {row.get('question', '')}"
    else:
        instruction = f"Summarize the following text in language code {language}. Do not add facts."
        body = row["source"]
    if condition not in {"zero_shot", "one_shot_same_language", "one_shot_hindi"}:
        raise ValueError(f"Unsupported condition: {condition}")
    # Examples must be supplied explicitly in row['example']; never silently fabricate them.
    example = row.get("example") if condition != "zero_shot" else None
    prefix = ""
    if example:
        prefix = f"Example input: {example.get('source','')}\nExample output: {example.get('target','')}\n\n"
    elif condition != "zero_shot":
        raise ValueError(f"{condition} requires an explicit, verified row['example']")
    return f"{prefix}{instruction}\n\nInput:\n{body}\n\nOutput:"
