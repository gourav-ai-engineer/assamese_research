"""Exercise the real transformers code path with a tiny random model (no download)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from northeast_eval.generation import LLM, run_job  # noqa: E402

CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    path = tmp_path_factory.mktemp("tiny")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train_from_iterator(["Translate English Assamese বাক্য Sentence: Answer"] * 10,
                            trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<eos>", "<|user|>",
                                                                               "<|assistant|>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", unk_token="<unk>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    cfg = transformers.LlamaConfig(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, eos_token_id=fast.eos_token_id)
    transformers.LlamaForCausalLM(cfg).save_pretrained(path)
    return str(path)


def test_generate_batched(tiny_model):
    llm = LLM(tiny_model, dtype=torch.float32)
    assert llm.is_chat and llm.render("hi").startswith("<|user|>hi")
    outs = llm.generate(["short", "a much longer prompt here"], max_new_tokens=4, batch_size=2)
    assert len(outs) == 2 and all(o["output_tokens"] <= 4 for o in outs)


def test_run_job_with_real_model(tiny_model, tmp_path):
    import json
    data = tmp_path / "eval.jsonl"
    data.write_text(json.dumps({"id": "1", "task": "translation", "language": "as", "source_lang": "en",
                                "target_lang": "as", "source": "Sentence", "reference": "বাক্য"}) + "\n",
                    encoding="utf-8")
    llm = LLM(tiny_model, dtype=torch.float32, is_chat=False)
    assert run_job(llm, data, tmp_path / "out.jsonl", "zero_shot", max_new_tokens=4) == 1


def test_chat_few_shot_render(tiny_model):
    from northeast_eval.prompts import build_messages
    llm = LLM(tiny_model, dtype=torch.float32)
    row = {"id": "1", "task": "translation", "language": "as", "source_lang": "en", "target_lang": "as",
           "source": "Sentence", "reference": "বাক্য"}
    text = llm.render(build_messages(row, "few_shot", [row]))
    assert text.count("<|user|>") == 2 and text.endswith("<|assistant|>")
