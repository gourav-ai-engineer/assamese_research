from northeast_eval.metrics import (
    chrf_bootstrap,
    chrf_corpus,
    max_token_f1,
    spearman,
    token_f1,
    tokenizer_fertility,
)


def test_token_f1_exact():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_empty():
    assert token_f1("", "") == 1.0
    assert token_f1("answer", "") == 0.0


def test_token_f1_bengali_script():
    assert token_f1("ৰজা হৰিশ্চন্দ্ৰ", "হৰিশ্চন্দ্ৰ") == 2 / 3


def test_max_token_f1():
    assert max_token_f1(["a b", "c"], "c") == 1.0


def test_chrf_exact():
    assert chrf_corpus(["bonjour"], ["bonjour"]) > 99


def test_chrf_bootstrap_matches_corpus():
    refs = ["the cat sat on the mat", "a dog barked", "it is raining today"]
    preds = ["the cat sat on a mat", "dog barks", "raining today"]
    point, lo, hi = chrf_bootstrap(refs, preds, n_resamples=200)
    assert abs(point - chrf_corpus(refs, preds)) < 1e-6
    assert lo <= point <= hi


def test_fertility():
    assert tokenizer_fertility(8, 4) == 2
    assert tokenizer_fertility(8, 0) is None


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
