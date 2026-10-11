import pytest

from northeast_eval.languages import script_match, script_profile
from northeast_eval.prompts import build_prompt, clean_output

ROW = {"id": "1", "task": "translation", "language": "as", "source_lang": "en", "target_lang": "as",
       "source": "Good morning.", "reference": "শুভ প্ৰভাত।"}
HI_EX = {"id": "h", "task": "translation", "language": "hi", "source_lang": "en", "target_lang": "hi",
         "source": "Thank you.", "reference": "धन्यवाद।"}


def test_script_detection():
    assert script_match("শুভ প্ৰভাত।", "Bengali")
    assert not script_match("Good morning", "Bengali")
    assert script_match("धन्यवाद", "Devanagari")
    assert script_match("ꯃꯤꯇꯩ", "Meitei")
    assert script_profile("") == {}


def test_zero_shot_prompt():
    p = build_prompt(ROW)
    assert "from English to Assamese" in p and p.endswith("English: Good morning.\nAssamese:")


def test_few_shot_hindi_keeps_labels():
    p = build_prompt(ROW, "few_shot_hindi", [HI_EX])
    assert "English: Thank you.\nHindi: धन्यवाद।" in p
    assert p.endswith("Assamese:")


def test_few_shot_requires_exemplars():
    with pytest.raises(ValueError):
        build_prompt(ROW, "few_shot")


def test_qa_prompt():
    row = {"id": "q", "task": "qa", "language": "brx", "context": "Ctx", "question": "Q?", "reference": "A"}
    assert "in Bodo" in build_prompt(row)


def test_clean_output():
    assert clean_output("Assamese: শুভ প্ৰভাত।\nEnglish: next", "translation") == "শুভ প্ৰভাত।"
    assert clean_output("<think>\n\n</think>\n\n\"ok\"", "qa") == "ok"


def test_chat_few_shot_uses_turns():
    from northeast_eval.prompts import build_messages
    msgs = build_messages(ROW, "few_shot_hindi", [HI_EX])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "to Hindi" in msgs[0]["content"] and msgs[1]["content"] == "धन्यवाद।"
    assert msgs[-1]["content"].endswith("English: Good morning.\nAssamese:")
    assert len(build_messages(ROW)) == 1
