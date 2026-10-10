import json
from pathlib import Path
from datetime import datetime, timezone
from .prompts import build_prompt, PROMPT_VERSION
from .schema import read_jsonl
def generate_file(input_path, output_path, model_id, task="translation", condition="zero_shot",
                  limit=None, max_new_tokens=256, revision=None):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError("Install a compatible PyTorch build and transformers.") from e
    rows = [r for r in read_jsonl(input_path) if r["task"] == task]
    if limit is not None: rows = rows[:limit]
    if not rows: raise ValueError("No rows match the selected task.")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kw = {"revision": revision} if revision else {}
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", **kw)
    model.eval()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = build_prompt(row, condition)
            messages = [{"role":"user","content":prompt}]
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if tok.chat_template else prompt
            inputs = tok(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                        pad_token_id=tok.eos_token_id)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            result = dict(row, prediction=tok.decode(new_tokens, skip_special_tokens=True).strip(),
                          model_id=model_id, model_revision=revision, condition=condition,
                          prompt_version=PROMPT_VERSION, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                          input_tokens=int(inputs["input_ids"].shape[1]), output_tokens=int(new_tokens.shape[0]))
            f.write(json.dumps(result, ensure_ascii=False)+"\n")
    return len(rows)
