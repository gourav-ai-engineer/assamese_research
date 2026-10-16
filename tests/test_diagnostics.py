import json

import pytest

from northeast_eval.metrics import chrf_corpus
from northeast_eval.reporting import (
    diagnostics_table,
    language_breakdown,
    lid_group,
    score_rows,
)

pytest.importorskip("aksharamukha")
from northeast_eval.script_norm import project  # noqa: E402

MNI = "মসি চহি মরিদা মারতিলীগী মঙাশুবা।"


def test_projection_makes_scripts_comparable():
    meitei = project(MNI, "mni")
    assert any("ꯀ" <= c <= "꯿" for c in meitei)
    assert chrf_corpus([MNI], [meitei]) < 5                       # raw chrF punishes the script
    assert chrf_corpus([project(MNI, "mni")], [project(meitei, "mni")]) > 95
    assert project("শুভ প্ৰভাত", "as") == "শুভ প্ৰভাত"            # other languages untouched


def test_lid_groups():
    assert lid_group("brx", "brx") == "on_target"
    assert lid_group("mni_Mtei", "mni") == "on_target_meitei_script"
    assert lid_group("bho", "brx") == "hindi_cluster"
    assert lid_group("en", "as") == "english_or_latin"
    assert lid_group("bn", "as") == "other_northeast_or_bengali"


class FakeLid:
    def predict(self, texts):
        return ["hi" if "क" in t else "mni_Mtei" if "\uabc0" <= t[0] <= "\uabff" else "brx" for t in texts]


def test_breakdown_and_scores():
    assert language_breakdown(FakeLid(), ["क", "ब", ""], "brx") == \
        {"hindi_cluster": 0.333, "on_target": 0.333, "empty": 0.333}
    rows = [{"id": str(i), "task": "translation", "language": "mni", "direction": "en->mni", "source_lang": "en",
             "target_lang": "mni", "source": "x", "reference": MNI, "prediction": project(MNI, "mni"),
             "model_id": "m", "condition": "zero_shot"} for i in range(3)]
    s = score_rows(rows, n_resamples=10, lid=FakeLid())[0]
    assert s["score"] < 5 and s["chrf_script_projected"] > 95
    assert json.loads(s["language_breakdown"]) == {"on_target_meitei_script": 1.0}
    assert s["off_target_rate"] == 0
    assert "Script penalty" in diagnostics_table([s])
