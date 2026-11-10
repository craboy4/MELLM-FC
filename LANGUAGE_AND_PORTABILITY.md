# English source and reproduction boundaries

The current release uses English documentation, comments, docstrings, runtime messages, variable descriptions, operator guidance, and LLM prompts. Internal grouping fields are `firm_id` and `year`, matching the external panel contract. The label is `label` in the CSV and `Fraud` internally. Chinese source prose was translated or replaced with English API summaries, not hidden in encoded literals. Python identifiers and formula expressions were retained.

## Fixed-formula numerical evaluation

The release evaluates all 100 expressions in pool order, using the fixed numerical operators, classifier parameters, ten seeds, time weights, missing-value handling and metrics. All feature values and 40 fresh classifier fits match the licensed-sample calculations and saved predictions in the tested environment. Software versions and numerical checks appear in [local numerical verification](results/local_reproduction_verification.json).

## New LLM generation

Translating a prompt changes the request. English wording is intended to preserve the procedure but does **not** reproduce the original model's responses. Language-sensitive guidance, such as short strategy names and summary lengths, is expressed in English. Text-based gate recognition and prompt-replacement rules are checked against English messages. No new model requests were used to validate this translation.

Use a new work directory for generation with this source. Each run's manifest freezes its source and configuration; do not bypass checks to resume an older run. Never represent a response saved for a different prompt as a response to the English prompt. Licensed original-language case narratives supplied locally remain source evidence and are not automatically translated.

The release uses English sources. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) distinguishes archived and current hashes; [MANIFEST.json](MANIFEST.json) covers current release files.
