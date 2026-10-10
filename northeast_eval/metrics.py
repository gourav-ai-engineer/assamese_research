from __future__ import annotations
import re
from collections import Counter
from sacrebleu.metrics import CHRF

_CHRF = CHRF(word_order=2)

def chrf_corpus(references: list[str], predictions: list[str]) -> float:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal lengths")
    if not references:
        raise ValueError("cannot score an empty corpus")
    return float(_CHRF.corpus_score(predictions, [references]).score)

def chrf_sentence(reference: str, prediction: str) -> float:
    return float(_CHRF.sentence_score(prediction, [reference]).score)

def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

def token_f1(reference: str, prediction: str) -> float:
    ref, pred = normalize_tokens(reference), normalize_tokens(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = sum((Counter(ref) & Counter(pred)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)

def tokenizer_fertility(token_count: int, word_count: int) -> float | None:
    """Tokens per word under a documented script-aware word segmentation rule."""
    if word_count <= 0:
        return None
    return token_count / word_count
