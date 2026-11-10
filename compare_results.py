"""Compare a completed reproduction with the archived fixed-seed aggregate references."""
import argparse
from pathlib import Path
from reproduce import read

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('metrics',type=Path)
    ap.add_argument('--tolerance',type=float,default=1e-10)
    args=ap.parse_args()
    expected=read(Path(__file__).resolve().parent/'results/reference_metrics.json')['results']
    actual=read(args.metrics)
    lookup={(r['seed'],r['condition']):r for r in expected}
    failures=[]
    for r in actual:
        identity=(r['seed'],r['condition'])
        if identity not in lookup:
            failures.append((identity,'unregistered seed/condition'));continue
        for stage in ('validation','test'):
            for metric,want in lookup[identity][stage].items():
                observed=r.get(stage,{}).get(metric)
                if observed is None or abs(observed-want)>args.tolerance:
                    failures.append((identity,stage,metric,observed,want))
    if not actual:failures.append(('empty result',))
    for row in failures:print(row)
    if failures:
        raise SystemExit('Results differ. Check licensed sample, precision and pinned software; do not tune to match test outcomes.')
    print(f'PASS: {len(actual)} condition/seed results match the archived validation and test metrics.')

if __name__=='__main__':main()
