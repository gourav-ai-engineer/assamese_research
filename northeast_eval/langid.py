"""Character n-gram language identifier for off-target (language confusion) analysis.

Trained on the FLORES-IN dev split of IndicGenBench (29 Indic languages + English) and
evaluated on the FLORES-IN test split, so no evaluation sentence is seen in training.
Used to measure how often an output meant to be Bodo is actually Hindi, Assamese is
actually Bengali, and so on.
"""
import json
import pickle
from pathlib import Path

from .data import _load_examples


def flores_language_texts(src_dir, split):
    """{lang: [sentences]} for every FLORES-IN language, plus English."""
    texts = {}
    for path in sorted(Path(src_dir, "flores_in").glob(f"flores_en_*_{split}.json")):
        lang = path.stem.split("_")[2]
        ex = _load_examples(path)
        texts[lang] = [e["target"].strip() for e in ex]
        texts.setdefault("en", []).extend(e["source"].strip() for e in ex[:40])
    # Models often write Manipuri in Meitei Mayek; add it as its own label so script choice
    # and language choice are reported separately.
    if "mni" in texts:
        from .script_norm import project
        texts["mni_Mtei"] = [project(t, "mni") for t in texts["mni"]]
    return texts


def _xy(texts):
    xs, ys = [], []
    for lang, sents in texts.items():
        xs += sents
        ys += [lang] * len(sents)
    return xs, ys


def train(src_dir, out_path=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    xs, ys = _xy(flores_language_texts(src_dir, "dev"))
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True,
                                          min_df=2, max_features=400_000), LinearSVC(C=0.5))
    model.fit(xs, ys)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(pickle.dumps(model))
    return model


def evaluate(model, src_dir, focus=("as", "mni", "mni_Mtei", "brx", "hi", "bn", "en")):
    """Held-out accuracy on FLORES-IN test, overall and for the focus languages."""
    from collections import Counter
    xs, ys = _xy(flores_language_texts(src_dir, "test"))
    pred = model.predict(xs)
    overall = sum(p == y for p, y in zip(pred, ys)) / len(ys)
    per_lang = {}
    for lang in focus:
        idx = [i for i, y in enumerate(ys) if y == lang]
        if idx:
            wrong = Counter(pred[i] for i in idx if pred[i] != lang)
            per_lang[lang] = {"accuracy": sum(pred[i] == lang for i in idx) / len(idx),
                              "n": len(idx), "top_confusions": wrong.most_common(3)}
    return {"overall_accuracy": overall, "n": len(ys), "languages": len(set(ys)), "per_language": per_lang}


def load(path):
    return pickle.loads(Path(path).read_bytes())


def main(src_dir, out_dir):
    model = train(src_dir, Path(out_dir) / "langid.pkl")
    report = evaluate(model, src_dir)
    (Path(out_dir) / "langid_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
