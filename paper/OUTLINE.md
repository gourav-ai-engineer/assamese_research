# Manuscript outline (4–6 page target)

## Working title
**How Well Do Small Open LLMs Handle Northeast Indian Languages? An Empirical Study on Assamese, Manipuri and Bodo**

## Abstract (write after results)
Problem; three target languages and Hindi reference; models and benchmark subsets; tasks and conditions; main measured findings with uncertainty; limitations and artifact link. Do not draft result claims before the final tables exist.

## 1. Introduction
Low-resource language access; why 1–4B open models matter; three language/script scope; contributions stated as measured artifacts, not assumed novelty.

## 2. Related work
IndicGenBench; IndicParam; MILU; small multilingual models; Northeast language MT shared tasks; evaluation-metric limitations.

## 3. Benchmark and protocol
Dataset provenance and license; language/script coverage; sample sizes; split and sampling; model revisions; prompts; decoding; compute.

## 4. Results
Translation chrF by direction/language/model; QA token-F1 only where task coverage supports it; prompt condition comparisons; confidence intervals or paired bootstrap where appropriate; missing runs.

## 5. Error analysis
Blind sample; wrong script/language, omissions, hallucinations, wrong word sense, grammar/inflection; tokenizer fertility and cautious correlation analysis.

## 6. Limitations
Benchmark coverage, reference mismatch, metric limitations, native-speaker coverage, hardware, prompt sensitivity, language/script variants and model licensing.

## 7. Conclusion
Only conclusions supported by measured data; reproducibility statement.

## Tables to produce
1. Dataset counts by split/language/direction/script.
2. Model/revision/parameter count/license/quantization/hardware.
3. Translation corpus chrF per direction and language.
4. QA F1 only for supported benchmark-language pairs.
5. Prompt-condition deltas and error categories.
