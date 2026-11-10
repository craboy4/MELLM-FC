# Reproduction

`reproduce.py` computes the 100 frozen formulas and evaluates Base and Base + MELLM-FC for every fixed seed in `configs/fixed_recipe.json`. It fits on 2010-2018 for validation on 2019-2021, then fits on 2010-2021 for evaluation on 2022-2024. Each validation threshold is applied to the corresponding final model. Outputs include per-seed, annual and ranking metrics.

Base medians are estimated from each training period. Formula order, focal loss, year weights, feature-gain multipliers and 1,000 boosting rounds follow the saved configuration. The runner evaluates every supplied seed with the same settings. Reference metrics are used only by `compare_results.py` after fitting.

The Chinese evaluation is retrospective. Earlier classifier and retained-subset comparisons included the existing test period. Independent evaluation requires new observations.

The original numerical check used Python 3.12, NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0 and LightGBM 4.6.0. Historical aggregate checks are recorded in [local verification](results/local_reproduction_verification.json). Numerical equality depends on the prepared sample and software build.

New LLM generation uses English prompts, development feedback and validation selection. Its binary-loss classifier is separate from the final focal-loss evaluation. See [language and portability](LANGUAGE_AND_PORTABILITY.md) for prompt and checkpoint details.

Generation runs exactly 100 rounds. A common validation-only rule compares a forward-selected subset of 10-20 formulas with the full pool and Base, then retains the highest-AUC option.
