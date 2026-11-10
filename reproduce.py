"""Evaluate the frozen formulas with the paper's fixed classifier seeds."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from fraud_alpha.expression import Feature
from fraud_alpha.parser import build_parser
from mellm_metrics import focal_objective, threshold, metrics
HERE = Path(__file__).resolve().parent
BASE = [f'N{i}' for i in range(1, 26)]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_panel(path):
    f = pd.read_csv(path, dtype={'firm_id': str}, float_precision='round_trip')
    needed = ['firm_id', 'year', 'label'] + BASE
    if set(f.columns) != set(needed):
        raise ValueError('Input must contain exactly firm_id,year,label,N1,...,N25; no extra fields.')
    if f[['firm_id', 'year', 'label']].isna().any().any():
        raise ValueError('Missing firm_id, year or label.')
    for c in ['year', 'label'] + BASE:
        f[c] = pd.to_numeric(f[c], errors='raise')
    if not (f.year == np.floor(f.year)).all() or not f.year.between(2010, 2024).all():
        raise ValueError('Years must be integers in 2010–2024.')
    if not set(f.label.unique()).issubset({0, 1}):
        raise ValueError('Labels must be 0 or 1.')
    f['year'] = f.year.astype(int)
    f['label'] = f.label.astype(int)
    if f.duplicated(['firm_id', 'year']).any():
        raise ValueError('Duplicate firm-year rows.')
    f = f.rename(columns={'firm_id': 'firm_id', 'year': 'year', 'label': 'Fraud'})
    return f[['firm_id', 'year', 'Fraud'] + BASE].sort_values(['firm_id', 'year'], kind='stable').reset_index(drop=True)

def build_matrix(raw, stage, report=None, workers=1):
    if stage not in ('development', 'final'):
        raise ValueError(stage)
    f = raw.loc[raw['year'].le(2021)].reset_index(drop=True).copy() if stage == 'development' else raw.copy()
    cutoff = 2018 if stage == 'development' else 2021
    train = f['year'].le(cutoff).to_numpy()
    medians = f.loc[train, BASE].replace([np.inf, -np.inf], np.nan).median()
    f[BASE] = f[BASE].replace([np.inf, -np.inf], np.nan).fillna(medians)
    Feature.set_available_features(BASE)
    parser = build_parser()
    formulas = read(HERE / 'formulas/retained_formulas.json')['generated_features']
    if len(formulas) != 100:
        raise ValueError('The selected recipe requires exactly 100 frozen formulas.')
    values = [f[BASE].to_numpy(float)]
    before = f.loc[train].reset_index(drop=True)
    if workers > 1:
        import sys
        sys.path.insert(0, str(HERE / 'workflow'))
        from feature_parallel import FeatureValueBackend
        with FeatureValueBackend(f, before, train, BASE, workers=workers) as backend:
            futures = [backend.submit(parser.parse(item['expression'])) for item in formulas]
            for i, future in enumerate(futures, 1):
                values.append(future.result()['values'][:, None])
                if report:
                    report({'phase': 'formulas', 'stage': stage, 'completed': i, 'total': len(formulas)})
        return (np.column_stack(values), f[['firm_id', 'year', 'Fraud']].copy())
    for i, item in enumerate(formulas, 1):
        expression = parser.parse(item['expression'])
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            a = np.asarray(expression.evaluate(f), float)
            b = np.asarray(expression.evaluate(before), float)
        if a.shape != (len(f),) or not np.allclose(a[train], b, equal_nan=True):
            raise ValueError(f"Invalid shape or temporal-prefix dependence: {item['feature']}")
        values.append(np.where(np.isfinite(a), a, np.nan)[:, None])
        if report and (i % 10 == 0 or i == len(formulas)):
            report({'phase': 'formulas', 'stage': stage, 'completed': i, 'total': len(formulas)})
    return (np.column_stack(values), f[['firm_id', 'year', 'Fraud']].copy())

def train_stage(X, keys, condition, stage, seed, recipe, output):
    if condition == 'Base':
        X = X[:, :25]
    cutoff = 2018 if stage == 'development' else 2021
    fit = keys['year'].le(cutoff).to_numpy()
    evaluate = keys['year'].between(2019, 2021).to_numpy() if stage == 'development' else keys['year'].between(2022, 2024).to_numpy()
    y = keys.Fraud.to_numpy(int)
    if not fit.any() or not evaluate.any() or set(y[fit]) != {0, 1} or (set(y[evaluate]) != {0, 1}):
        raise ValueError('Both classes and nonempty training/evaluation sets are required in each stage.')
    params = dict(recipe['underlying_classifier_parameters'], seed=seed)
    if condition == 'full':
        params['feature_contri'] = [1.0] * 25 + [recipe['generated_feature_gain_multiplier']] * 100
    weights = np.exp(-recipe['time_weight_decay'] * (2021 - keys.loc[fit, 'year'].to_numpy()))
    objective = focal_objective(recipe['focal_loss_specification'])
    dsparams = dict(params)
    dsparams.pop('objective')
    started = time.monotonic()
    model = lgb.train(dict(params, objective=objective), lgb.Dataset(X[fit], label=y[fit], weight=weights, params=dsparams), num_boost_round=recipe['num_boost_round'])
    prediction = model.predict(X[evaluate], num_threads=params['num_threads'])
    folder = Path(output) / f'seed{seed}' / condition / stage
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'model.txt').write_text(model.model_to_string(), encoding='utf-8')
    np.save(folder / 'prediction.npy', prediction)
    save(folder / 'fit.json', dict(stage=stage, condition=condition, seed=seed, features=X.shape[1], train_rows=int(fit.sum()), evaluation_rows=int(evaluate.sum()), trees=model.num_trees(), seconds=time.monotonic() - started, loss_stats=objective.stats, prediction_sha256=sha(folder / 'prediction.npy'), model_sha256=sha(folder / 'model.txt')))
    return (y[evaluate], prediction)

def run(input_path, output, seeds, matrix_observer=None, workers=1):
    recipe = read(HERE / 'configs/fixed_recipe.json')
    raw = load_panel(input_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists():
        raise FileExistsError('Use a new output directory; previous runs are never overwritten silently.')
    save(output / 'run.json', dict(status='started', input_sha256=sha(input_path), seeds=seeds, recipe_sha256=sha(HERE / 'configs/fixed_recipe.json'), formulas_sha256=sha(HERE / 'formulas/retained_formulas.json'), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, lightgbm=lgb.__version__, rows=len(raw), firms=int(raw['firm_id'].nunique()), api_requests=0))
    records = {(s, c): dict(seed=s, condition=c, features=25 if c == 'Base' else 125) for s in seeds for c in ('Base', 'full')}
    report = lambda j: print(json.dumps(j), flush=True)
    for stage in ('development', 'final'):
        X, keys = build_matrix(raw, stage, report, workers=workers)
        if matrix_observer:
            matrix_observer(stage, X, keys)
        for seed in seeds:
            for condition in ('Base', 'full'):
                report(dict(phase='fit', stage=stage, condition=condition, seed=seed))
                y, p = train_stage(X, keys, condition, stage, seed, recipe, output)
                r = records[seed, condition]
                if stage == 'development':
                    r['threshold'] = threshold(y, p)
                    r['validation'] = metrics(y, p, r['threshold'])
                else:
                    r['test'] = metrics(y, p, r['threshold'])
                    years = keys.loc[keys['year'].between(2022, 2024), 'year'].to_numpy()
                    r['annual'] = {str(t): metrics(y[years == t], p[years == t], r['threshold']) for t in sorted(set(years)) if len(set(y[years == t])) == 2}
                save(output / 'metrics.json', list(records.values()))
                report(dict(phase='fit_complete', stage=stage, condition=condition, seed=seed))
    result = list(records.values())
    pd.DataFrame([dict(seed=r['seed'], condition=r['condition'], **r['test']) for r in result]).to_csv(output / 'metrics.csv', index=False)
    save(output / 'complete.json', dict(status='complete', api_requests=0, seeds=seeds, model_fits=4 * len(seeds)))
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'outputs/reproduction')
    parser.add_argument('--seeds', default='all', help='All ten paper seeds by default; accepts a comma-separated subset.')
    parser.add_argument('--workers', type=int, default=1, help='Expression-only worker processes; training remains fixed at two threads.')
    args = parser.parse_args()
    recipe = read(HERE / 'configs/fixed_recipe.json')
    seeds = recipe['si_seeds'] if args.seeds == 'all' else [int(x) for x in args.seeds.split(',')]
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args.data, args.out, seeds, workers=args.workers)
