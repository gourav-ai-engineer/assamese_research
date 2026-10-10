# Frozen protocol and publication checklist

## Main comparison
Evaluate each available model on the same frozen examples, with the same direction, prompt condition, decoding rule and output budget. Preserve example IDs across models. If a model cannot process a language or script, log the failure; do not silently replace examples.

## Conditions
1. Zero-shot.
2. One-shot same-language: example must come from a permitted development split, not test.
3. One-shot Hindi: use a documented Hindi exemplar and explain why cross-language transfer is plausible.
4. Translate-test: only as a separate experiment, using a documented translation system in both directions. Verify actual support for Assamese, Bodo, and Manipuri in the exact script. Report translation-system errors and do not compare it as if it were a native-language-only model.

## Data split and sample
The literature-review draft proposes up to 500 FLORES sentences per direction and 300 QA examples per language, but this is a target, not a guarantee of availability or compute feasibility. Freeze the sampling script and seed before model runs. Record exact counts after coverage checks.

## Metrics
- Translation: sacreBLEU implementation of chrF (record version and settings), corpus and sentence-level summaries.
- QA: token F1 with Unicode-aware tokenization. State that this is a lexical metric and may penalize valid paraphrases.
- Summarization: do not treat chrF alone as a sufficient quality measure; add blind human review or a justified complementary metric.
- Tokenizer fertility: specify tokenizer revision and script-aware segmentation rule. Report correlation descriptively and avoid causal claims.
- Error taxonomy: wrong language/script; missing information; hallucinated content; wrong word sense; grammar/inflection.

## Human evaluation
Use blinded outputs, fixed rubric, random ordering and a documented sampling method. Suggested adequacy and fluency scales are 1–5. Record rater language proficiency, instructions, disagreements and agreement statistics when multiple raters overlap. Obtain consent before collecting or publishing ratings.

## Mandatory run metadata
Commit SHA; UTC start/end; hardware; OS; Python, PyTorch and Transformers versions; model/tokenizer IDs and revisions; dataset versions; prompt version; decoding parameters; sample IDs; all errors/OOMs; exact command.

## Reporting guardrails
- No fabricated results or manually invented baselines.
- Separate upstream published scores from newly measured results.
- Do not claim “first” until the literature search is rerun and archived with search queries/date.
- Include negative results, missing coverage and failed runs.
- Check licenses and model terms; cite all datasets, models and metric tools.
- A two-week timeline is a sprint target, not a guarantee of acceptance or publication.
