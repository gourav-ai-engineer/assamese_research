# Data acquisition and governance

No benchmark data is committed. `northeast-eval prepare` clones
[IndicGenBench](https://github.com/google-research-datasets/indic-gen-bench) into `data/raw/` and writes
`data/processed/manifest.json` with the source commit, seed and counts.

Verified coverage (IndicGenBench, checked 2026-10-10):

| Set | as | mni | brx | hi | Splits |
|---|---|---|---|---|---|
| FLORES-IN (en↔X) | ✓ | ✓ | ✓ | ✓ | dev 997, test 1012 |
| XorQA-IN | ✓ | ✓ | ✓ | ✓ | train/dev/test (~530 test) |
| CrossSum-IN | ✓ | ✓ | ✓ | ✓ | train/dev/test |

Manipuri text in IndicGenBench is in **Bengali script**, even though the README lists Meitei. For
Meitei Mayek use `northeast-eval prepare-in22` (IN22-Gen, requires accepting the dataset terms).

Row schema (one JSON object per line):
```json
{"id":"flores-test-en-as-0001","task":"translation","language":"as","direction":"en->as","source_lang":"en","target_lang":"as","source":"...","reference":"..."}
{"id":"xorqa-test-brx-0001","task":"qa","language":"brx","context":"English context","question":"Question in Bodo","reference":"answer","references":["answer"],"en_question":"...","en_answer":"..."}
```
Toy files in `data/examples/` are smoke tests only.
