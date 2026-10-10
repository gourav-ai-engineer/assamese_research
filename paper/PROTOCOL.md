# Frozen protocol and publication checklist

## Conditions
1. Zero-shot.
2. One-shot same-language using a permitted development example, never a test example.
3. One-shot Hindi with a documented exemplar.
4. Translate-test as a separate experiment only when the chosen translation system demonstrably supports the language and exact script in both directions. Log translation errors.

## Sampling
The literature-review draft proposes up to 500 FLORES sentences per direction and 300 QA items per language. These are targets, not guarantees of availability or compute feasibility. Freeze sample IDs and seed before runs; report actual counts.

## Metrics
- Translation: SacreBLEU chrF, report version/settings, corpus and sentence-level summaries.
- QA: Unicode-aware token F1; state that lexical overlap penalizes valid paraphrases.
- Summarization: chrF alone is insufficient; add blind human review or a justified complementary metric.
- Tokenizer fertility: document tokenizer revision and script-aware segmentation rule; correlation is descriptive, not causal.
- Error taxonomy: wrong language/script, missing information, hallucinated content, wrong word sense, grammar/inflection.

## Human evaluation
Blind and randomize outputs; rate adequacy and fluency from 1–5. Record rater instructions, consent, proficiency, disagreements and agreement statistics when overlapping ratings exist.

## Mandatory metadata
Commit SHA, UTC timestamps, hardware, OS, Python/PyTorch/Transformers versions, model/tokenizer IDs and revisions, dataset version, prompt version, decoding settings, sample IDs, failures and exact command.

## Integrity guardrails
Do not fabricate scores or baselines. Separate published baselines from new measurements. Do not claim "first" until a date-stamped literature search log supports it. Report missing language coverage and failed runs. Check licenses. Two weeks is a sprint goal, not a publication guarantee.
