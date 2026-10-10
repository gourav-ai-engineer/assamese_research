"""MT systems used for translate-test and as a dedicated-MT reference line.

Backends:
  indictrans2  ai4bharat IndicTrans2 distilled 200M (covers as, mni_Beng, mni_Mtei, brx, hi).
               Needs `pip install IndicTransToolkit` and accepting the model terms on the Hub.
  nllb         facebook/nllb-200-distilled-600M (no Bodo).
"""
from .languages import LANGUAGES, MNI_MTEI


def _code(lang, backend):
    meta = MNI_MTEI if lang == "mni_Mtei" else LANGUAGES[lang]
    code = meta[backend]
    if code is None:
        raise ValueError(f"{backend} does not support language {lang!r}")
    return code


class Translator:
    def __init__(self, backend="indictrans2", batch_size=16, device=None):
        import torch
        self.torch, self.backend, self.batch_size = torch, backend, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}

    def _load(self, direction):
        if direction in self._models:
            return self._models[direction]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if self.backend == "nllb":
            name = "facebook/nllb-200-distilled-600M"
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
        else:
            name = {"en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
                    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M"}[direction]
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True)
        model = model.to(self.device).eval()
        self._models[direction] = (tok, model)
        return tok, model

    def translate(self, texts, src, tgt, max_new_tokens=256):
        direction = "en-indic" if src == "en" else "indic-en"
        tok, model = self._load(direction if self.backend == "indictrans2" else "nllb")
        s, t = _code(src, self.backend), _code(tgt, self.backend)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [x if x.strip() else "." for x in texts[i:i + self.batch_size]]
            if self.backend == "indictrans2":
                from IndicTransToolkit.processor import IndicProcessor
                ip = IndicProcessor(inference=True)
                enc = tok(ip.preprocess_batch(batch, src_lang=s, tgt_lang=t), truncation=True,
                          padding="longest", return_tensors="pt").to(self.device)
                with self.torch.inference_mode():
                    gen = model.generate(**enc, num_beams=1, max_new_tokens=max_new_tokens)
                dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                out.extend(ip.postprocess_batch(dec, lang=t))
            else:
                tok.src_lang = s
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with self.torch.inference_mode():
                    gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(t),
                                         num_beams=1, max_new_tokens=max_new_tokens)
                out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return [o.strip() for o in out]
