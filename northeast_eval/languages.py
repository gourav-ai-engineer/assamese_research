"""Language, script and Unicode metadata used across the pipeline."""
import unicodedata

LANGUAGES = {
    "as": {"name": "Assamese", "script": "Bengali", "nllb": "asm_Beng", "indictrans2": "asm_Beng"},
    "mni": {"name": "Manipuri", "script": "Bengali", "nllb": "mni_Beng", "indictrans2": "mni_Beng"},
    "brx": {"name": "Bodo", "script": "Devanagari", "nllb": None, "indictrans2": "brx_Deva"},
    "hi": {"name": "Hindi", "script": "Devanagari", "nllb": "hin_Deva", "indictrans2": "hin_Deva"},
    "en": {"name": "English", "script": "Latin", "nllb": "eng_Latn", "indictrans2": "eng_Latn"},
}
# Manipuri in Meitei Mayek (IN22). IndicGenBench/FLORES Manipuri is Bengali script.
MNI_MTEI = {"name": "Manipuri", "script": "Meitei", "nllb": None, "indictrans2": "mni_Mtei"}

SCRIPT_RANGES = {
    "Bengali": [(0x0980, 0x09FF)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Meitei": [(0xABC0, 0xABFF), (0xAAE0, 0xAAFF)],
    "Latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}


def lang_name(code):
    return MNI_MTEI["name"] if code == "mni_Mtei" else LANGUAGES[code]["name"]


def lang_script(code):
    return MNI_MTEI["script"] if code == "mni_Mtei" else LANGUAGES[code]["script"]


def char_script(ch):
    cp = ord(ch)
    for script, ranges in SCRIPT_RANGES.items():
        if any(lo <= cp <= hi for lo, hi in ranges):
            return script
    return None


def script_profile(text):
    """Share of letters (category L*/M*) in each known script."""
    counts, total = {}, 0
    for ch in text:
        if not unicodedata.category(ch)[0] in "LM":
            continue
        total += 1
        s = char_script(ch) or "Other"
        counts[s] = counts.get(s, 0) + 1
    return {s: c / total for s, c in counts.items()} if total else {}


def script_match(text, expected_script, threshold=0.5):
    """True when at least `threshold` of the letters are in the expected script."""
    return script_profile(text).get(expected_script, 0.0) >= threshold
