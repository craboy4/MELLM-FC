"""Prepare a new optional generation run from user-supplied licensed inputs. Offline only."""
import argparse
import os
from pathlib import Path
import re
import sys
import reproduce
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True, help='Prepared development-only JSON case records; see data/README.md.')
    ap.add_argument('--workdir', type=Path, default=HERE / 'outputs/generation')
    args = ap.parse_args()
    work = args.workdir.resolve()
    os.environ['MELLM_WORKDIR'] = str(work)
    sys.path.insert(0, str(HERE / 'workflow'))
    from common import save, digest, RUN, DATA, CODE, read
    if (RUN / 'config.json').exists():
        raise FileExistsError('Existing run is frozen. Resume it; do not re-prepare or overwrite it.')
    frame = reproduce.load_panel(args.data)
    if not frame['firm_id'].str.fullmatch('\\d{1,6}').all():
        raise ValueError('Generation case matching requires numeric security codes of at most six digits, encoded consistently in the case JSON.')
    frame['firm_id'] = frame['firm_id'].astype(int)
    cases = read(args.cases)
    if not isinstance(cases, list) or not cases:
        raise ValueError('Supply genuine licensed development cases; empty cases are a different experimental condition.')
    for c in cases:
        required = {'Symbol', 'linked_years', 'ViolationID', 'Activity', 'source_record'}
        if not required <= set(c) or not c['Activity'].strip():
            raise ValueError('Incomplete case record. See data/README.md.')
        if not c['linked_years'] or any((not 2010 <= int(y) <= 2021 for y in c['linked_years'])):
            raise ValueError('Case years extend outside development.')
        if any((int(y) >= 2022 for y in re.findall('(?:19|20)\\d{2}', c['Activity']))):
            raise ValueError('Case text contains later-year facts. Exclude mixed or unclear records.')
    inputs = DATA / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    frame.loc[frame['year'].le(2021)].to_pickle(inputs / 'development_2010_2021.pkl')
    save(inputs / 'cases_development.json', cases)
    cfg = read(HERE / 'configs/generation_recipe.json')
    cfg['inputs'] = {p.name: dict(sha256=digest(p), bytes=p.stat().st_size) for p in inputs.iterdir() if p.is_file()}
    save(RUN / 'config.json', cfg)
    save(RUN / 'generation_context_policy.json', read(HERE / 'configs/generation_context_policy.json'))
    sources = list((HERE / 'workflow').glob('*.py')) + list((HERE / 'fraud_alpha').rglob('*.py')) + [HERE / 'run_generation.py', HERE / 'prepare_generation.py']
    save(RUN / 'source_manifest.json', dict(config_sha256=digest(RUN / 'config.json'), generation_context_policy_sha256=digest(RUN / 'generation_context_policy.json'), files={os.path.relpath(p, CODE).replace('\\', '/'): digest(p) for p in sources}))
    print('Prepared and froze a new generation run. API requests: 0. Test rows saved: 0.')
if __name__ == '__main__':
    main()
