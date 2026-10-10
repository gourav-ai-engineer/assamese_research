import unicodedata
from collections import Counter

import numpy as np
from sacrebleu.metrics import CHRF

# chrF (character n-grams only, word_order=0) as reported by IndicGenBench; chrF++ available too.
_CHRF = CHRF()
_CHRFPP = CHRF(word_order=2)


def chrf_corpus(references, predictions, plus=False):
    if len(references) != len(predictions):
        raise ValueError("length mismatch")
    if not references:
        raise ValueError("empty corpus")
    metric = _CHRFPP if plus else _CHRF
    return float(metric.corpus_score(list(predictions), [list(references)]).score)


def chrf_sentence(reference, prediction):
    return float(_CHRF.sentence_score(prediction, [reference]).score)


def chrf_bootstrap(references, predictions, n_resamples=1000, seed=0, alpha=0.05):
    """Corpus chrF with a percentile bootstrap CI (resampling sentence-level statistics)."""
    stats = np.array(_CHRF._extract_corpus_statistics(list(predictions), [list(references)]))
    point = float(_CHRF._compute_score_from_stats(list(stats.sum(axis=0))).score)
    rng = np.random.default_rng(seed)
    n = len(stats)
    samples = [
        _CHRF._compute_score_from_stats(list(stats[rng.integers(0, n, n)].sum(axis=0))).score
        for _ in range(n_resamples)
    ]
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mean_bootstrap(values, n_resamples=1000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_resamples, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def normalize_tokens(text):
    """Unicode-aware tokens: runs of letters, combining marks and digits.

    Python's `\\w` excludes combining marks (viramas, vowel signs), which would split
    Indic words, so we filter by Unicode category instead.
    """
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in text.casefold())
    return kept.split()


def token_f1(reference, prediction):
    ref, pred = normalize_tokens(reference), normalize_tokens(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = sum((Counter(ref) & Counter(pred)).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)


def max_token_f1(references, prediction):
    """SQuAD-style: best F1 over all gold answers."""
    return max(token_f1(r, prediction) for r in references)


def tokenizer_fertility(token_count, word_count):
    """Tokens per whitespace-delimited word."""
    return None if word_count <= 0 else token_count / word_count


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])
