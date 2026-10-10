# Data acquisition and schema

No third-party benchmark data is bundled in this repository.

## Primary sources
- IndicGenBench: https://github.com/AI4Bharat/IndicGenBench
- FLORES-200 / FLORES+: use official repository and license terms; check which language codes and split variants are supported.
- IndicGenBench QA and summarization tasks: verify exact dataset names, language coverage, split and license from upstream documentation.

## Required data card
Before every run, record:
- dataset name and canonical URL
- commit/release/version and download date
- license and redistribution restrictions
- split (never tune prompts on test data)
- language code and script (especially Manipuri: Meitei Mayek vs Bengali/Assamese script)
- filtering/deduplication and exact sampling seed
- number of examples before and after filtering
- whether examples were human-translated, translated by a model, or collected from another source

## Local JSONL format
One object per line. Translation example:
```json
{"id":"flores-example-id","task":"translation","language":"as","direction":"en->as","source_lang":"en","target_lang":"as","script":"Bengali","source":"English source sentence","reference":"Licensed target reference"}
```
QA example:
```json
{"id":"qa-001","task":"qa","language":"brx","source":"Context passage","context":"Context passage","question":"Question in target language","reference":"Gold answer"}
```

Do not place full licensed test sets in Git. Keep data locally or in approved storage. Use small hand-written toy examples only for pipeline tests, never for research scores.
