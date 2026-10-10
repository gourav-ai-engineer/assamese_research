from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from .prompts import build_prompt, PROMPT_VERSION
from .schema import read_jsonl

def generate_file(input_path: str, output_path: str, model_id: str, task: str = "translation",
                  condition: str = "zero_shot", limit: int | None = None, max_new_tokens: int = 256,
                  revision: str | None = None) -> int:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError("Install PyTorch and transformers in a compatible environment.") from e
    rows = read_jsonl(input_path)
    rows = [r for r in rows if r["task"] == task]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No rows match requested task.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kwargs = {"revision": revision} if revision else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", **kwargs
    )
    model.eval()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(output_path).open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = build_prompt(row, condition=condition)
            messages = [{"role": "user", "content": prompt}]
            if getattr(tokenizer, "chat_template", None):
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                rendered = prompt
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.eos_token_id)
            continuation = output[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            result = dict(row)
            result.update({
                "prediction": prediction, "model_id": model_id, "model_revision": revision,
                "condition": condition, "prompt_version": PROMPT_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(continuation.shape[0]),
            })
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1
    return count
