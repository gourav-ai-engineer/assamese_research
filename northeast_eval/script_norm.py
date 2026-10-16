"""Script-projected scoring for Manipuri.

Manipuri is written in Bengali script (IndicGenBench/FLORES references) and in Meitei Mayek
(the official script, which models often produce). Raw chrF scores a perfect Meitei-script
translation near zero against a Bengali-script reference. We project both hypothesis and
reference into Meitei Mayek before scoring. Bengali->Meitei is many-to-one (e.g. ি/ী and
শ/ষ/স collapse), so the projection is deterministic and cannot be gamed by either script;
it only discards distinctions Meitei Mayek does not encode. Transliteration: Aksharamukha.
"""
from functools import lru_cache

PROJECTION = {"mni": ("Bengali", "MeeteiMayek")}


@lru_cache(maxsize=1)
def _transliterator():
    try:
        from aksharamukha import transliterate
    except ImportError as e:
        raise RuntimeError("pip install aksharamukha for script-projected scoring") from e
    return transliterate.process


def project(text, lang):
    """Map text to the canonical script of `lang`; languages without a projection pass through."""
    if lang not in PROJECTION or not text:
        return text
    src, tgt = PROJECTION[lang]
    # Characters already in the target script are left untouched by the transliterator.
    return _transliterator()(src, tgt, text)
