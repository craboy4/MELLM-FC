# MELLM-FC

Feature construction and fixed-formula evaluation for financial statement fraud detection. Includes 100 retained formulas, the LightGBM configuration, generation code and reference metrics.

## Evaluation

Use Python 3.12 and the pinned dependencies.

```sh
python -m pip install -r requirements.txt
python validate_data.py --data private_data/panel.csv --out outputs/data_check.json
python reproduce.py --data private_data/panel.csv --out outputs/main
python compare_results.py outputs/main/metrics.json
```

The default evaluates all ten paper seeds with fixed formulas and parameters, producing 40 fits. Seeds are 456, 42, 123, 789, 2026, 31415, 27182, 65537, 8675309 and 20261030. Results are saved for every seed.

For a synthetic-data smoke test, use `--data data/synthetic_demo.csv` and a new output directory. Input columns are `firm_id,year,label,N1,...,N25`. Licensed CSMAR data preparation is described in the [data guide](data/CSMAR_ACCESS.md). Numerical details appear in [the configuration](configs/fixed_recipe.json) and [reproduction notes](REPRODUCIBILITY.md).

## Generation

```sh
python -m pip install -r requirements-generation.txt
python prepare_generation.py --data private_data/panel.csv --cases private_data/cases_development.json --workdir outputs/generation
python run_generation.py --workdir outputs/generation --allow-api --allow-development-context-upload
```

Set `MELLM_API_KEY` and `MELLM_BASE_URL` for an endpoint authorized to receive development data. Generation runs exactly 100 rounds with three responses per round and selects formulas on validation. English prompts and checkpoint requirements are described in [language and portability](LANGUAGE_AND_PORTABILITY.md).

## Checks and license

```sh
python -m unittest discover -s tests -v
python verify_manifest.py
```

Author contributions are licensed for noncommercial research under [LICENSE](LICENSE). CSMAR inputs require authorized access. Third-party terms appear in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
