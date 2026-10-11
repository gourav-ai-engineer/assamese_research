"""Batched greedy generation with small open LLMs (Hugging Face transformers)."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from .prompts import (
    PROMPT_VERSION,
    build_messages,
    build_prompt,
    clean_output,
    english_qa_row,
)
from .schema import read_jsonl


def _imports():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise RuntimeError("Install PyTorch and transformers: pip install -e '.[gpu]'") from e
    return torch, transformers


class LLM:
    """Wraps one model. `is_chat=False` means raw-text prompting (base models)."""

    def __init__(self, model_id, revision=None, is_chat=None, family=None, dtype="auto"):
        torch, tf = _imports()
        self.torch, self.model_id, self.revision, self.family = torch, model_id, revision, family
        kw = {"revision": revision} if revision else {}
        self.tok = tf.AutoTokenizer.from_pretrained(model_id, **kw)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = tf.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto", **kw)
        except ValueError:  # e.g. multimodal Gemma 3 4B on older transformers
            self.model = tf.AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto", **kw)
        self.model.eval()
        self.is_chat = bool(self.tok.chat_template) if is_chat is None else is_chat

    def render(self, prompt):
        if not self.is_chat:
            return prompt
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.family in ("qwen3", "gemma4"):
            kwargs["enable_thinking"] = False  # both chat templates support a non-thinking switch
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self.tok.apply_chat_template(messages, **kwargs)

    def generate(self, prompts, max_new_tokens=128, batch_size=8, max_input_tokens=3072):
        rendered = [self.render(p) for p in prompts]
        order = sorted(range(len(rendered)), key=lambda i: len(rendered[i]), reverse=True)
        outputs = [None] * len(rendered)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([rendered[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=max_input_tokens, add_special_tokens=not self.is_chat).to(self.model.device)
            with self.torch.inference_mode():
                out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id)
            for j, i in enumerate(idx):
                new = out[j][enc["input_ids"].shape[1]:]
                outputs[i] = {"text": self.tok.decode(new, skip_special_tokens=True),
                              "input_tokens": int(enc["attention_mask"][j].sum()),
                              "output_tokens": int((new != self.tok.pad_token_id).sum())}
        return outputs


def pick_exemplars(pool, k):
    return pool[:k]


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        torch, tf = _imports()
        info.update(torch=torch.__version__, transformers=tf.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except RuntimeError:
        pass
    return info


def _done_ids(path):
    if not Path(path).exists():
        return set()
    with Path(path).open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def run_job(llm, eval_path, output_path, condition, exemplar_path=None, k=3, limit=None,
            max_new_tokens=128, batch_size=8, mt_questions=None):
    """Generate predictions for one eval file × condition; resumes if output exists.

    translate_test needs `mt_questions` ({id: English question}) from `mt-questions`; its
    English answers are stored as `prediction_en` with `prediction: null` until
    `back-translate` fills them in (MT runs in a separate environment, see translator.py).
    """
    rows = read_jsonl(eval_path)[:limit] if limit else read_jsonl(eval_path)
    done = _done_ids(output_path)
    rows = [r for r in rows if r["id"] not in done]
    if not rows:
        return 0
    exemplars = pick_exemplars(read_jsonl(exemplar_path), k) if exemplar_path else None

    extra = [{} for _ in rows]
    if condition == "translate_test":
        if rows[0]["task"] != "qa":
            raise ValueError("translate_test is defined for QA only")
        missing = [r["id"] for r in rows if r["id"] not in (mt_questions or {})]
        if missing:
            raise ValueError(f"No machine-translated question for {missing[0]}; run `mt-questions` first")
        en_q = [mt_questions[r["id"]] for r in rows]
        prompt_rows = [english_qa_row(r, q) for r, q in zip(rows, en_q)]
        for e, q in zip(extra, en_q):
            e["mt_question"] = q
    else:
        prompt_rows = rows
    builder = build_messages if getattr(llm, "is_chat", False) else build_prompt
    prompts = [builder(r, condition, exemplars) for r in prompt_rows]
    gens = llm.generate(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    preds = [clean_output(g["text"], rows[0]["task"]) for g in gens]
    if condition == "translate_test":
        for e, p in zip(extra, preds):
            e["prediction_en"] = p
        preds = [None] * len(preds)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with Path(output_path).open("a", encoding="utf-8") as f:
        for row, prompt, gen, pred, e in zip(rows, prompts, gens, preds, extra):
            out = dict(row, **e, prediction=pred, raw_output=gen["text"], prompt=prompt,
                       model_id=llm.model_id, model_revision=llm.revision, condition=condition,
                       k_shot=len(exemplars or []) if condition.startswith("few_shot") else 0,
                       prompt_version=PROMPT_VERSION, generated_at_utc=now,
                       input_tokens=gen["input_tokens"], output_tokens=gen["output_tokens"])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return len(rows)


def generate_file(input_path, output_path, model_id, task="translation", condition="zero_shot",
                  limit=None, max_new_tokens=128, revision=None, exemplars=None, k=3, family=None):
    """Single-file convenience wrapper used by the `generate` CLI command."""
    llm = LLM(model_id, revision=revision, family=family)
    return run_job(llm, input_path, output_path, condition, exemplars, k, limit, max_new_tokens)
