from northeast_eval.metrics import token_f1, chrf_corpus, tokenizer_fertility
def test_token_f1_exact():
    assert token_f1("hello world","hello world") == 1.0
def test_token_f1_empty():
    assert token_f1("","") == 1.0
    assert token_f1("answer","") == 0.0
def test_chrf_exact():
    assert chrf_corpus(["bonjour"],["bonjour"]) > 99
def test_fertility():
    assert tokenizer_fertility(8,4) == 2
    assert tokenizer_fertility(8,0) is None
