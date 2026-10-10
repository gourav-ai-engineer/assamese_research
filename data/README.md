# Data acquisition and governance

No third-party benchmark data is bundled. Start with [IndicGenBench](https://github.com/AI4Bharat/IndicGenBench) and follow official dataset licensing/access instructions. Verify actual language and split coverage for FLORES-IN, XORQA-IN-XX and CROSSSUM-IN; do not assume all contain every target language.

Record dataset URL, version/commit, download date, license, split, language code, script (especially Meitei Mayek vs Bengali script), filtering, deduplication, sample seed and counts. Do not commit licensed test sets unless redistribution is permitted.

Translation JSONL example:
```json
{"id":"item-001","task":"translation","language":"as","direction":"en->as","source_lang":"en","target_lang":"as","script":"Bengali","source":"English source","reference":"Licensed reference"}
```
QA JSONL example:
```json
{"id":"qa-001","task":"qa","language":"brx","source":"Context","context":"Context","question":"Question","reference":"Gold answer"}
```
One JSON object per line. Keep datasets out of Git. Toy examples are smoke tests only, never research results.
