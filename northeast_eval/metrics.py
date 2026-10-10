import re
from collections import Counter
from sacrebleu.metrics import CHRF
_CHRF = CHRF(word_order=2)
def chrf_corpus(references, predictions):
    if len(references) != len(predictions): raise ValueError("length mismatch")
    if not references: raise ValueError("empty corpus")
    return float(_CHRF.corpus_score(predictions, [references]).score)
def chrf_sentence(reference, prediction):
    return float(_CHRF.sentence_score(prediction, [reference]).score)
def normalize_tokens(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
def token_f1(reference, prediction):
    ref, pred = normalize_tokens(reference), normalize_tokens(prediction)
    if not ref and not pred: return 1.0
    if not ref or not pred: return 0.0
    overlap = sum((Counter(ref) & Counter(pred)).values())
    if not overlap: return 0.0
    p, r = overlap/len(pred), overlap/len(ref)
    return 2*p*r/(p+r)
def tokenizer_fertility(token_count, word_count):
    """Tokens per word; document the script-aware segmentation rule."""
    return None if word_count <= 0 else token_count/word_count
