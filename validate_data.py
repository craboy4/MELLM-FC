"""Validate a prepared local panel and report aggregate counts without fitting."""
import argparse
from pathlib import Path
from reproduce import load_panel, save, sha

EXPECTED = {
    'rows': 53825, 'firms': 5400, 'positives': 6556,
    'development_training_rows': 25294, 'validation_rows': 13072,
    'final_training_rows': 38366, 'test_rows': 15459, 'test_positives': 1670,
}


def inspect_panel(path):
    frame = load_panel(path)
    year = frame['year']
    test = year.ge(2022)
    observed = {
        'rows': len(frame), 'firms': int(frame['firm_id'].nunique()),
        'positives': int(frame['Fraud'].sum()),
        'development_training_rows': int(year.le(2018).sum()),
        'validation_rows': int(year.between(2019, 2021).sum()),
        'final_training_rows': int(year.le(2021).sum()),
        'test_rows': int(test.sum()),
        'test_positives': int(frame.loc[test, 'Fraud'].sum()),
    }
    return {
        'schema_valid': True, 'paper_counts_match': observed == EXPECTED,
        'identity_caveat': 'Matching counts does not establish row-level sample identity.',
        'input_sha256': sha(path), 'observed': observed, 'expected': EXPECTED,
        'by_year': [
            {'year': int(y), 'rows': len(group), 'positives': int(group['Fraud'].sum())}
            for y, group in frame.groupby('year', sort=True)
        ],
        'model_fits': 0, 'api_requests': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=Path('outputs/data_check.json'))
    args = parser.parse_args()
    if args.data.resolve() == args.out.resolve():
        parser.error('Input and output must be different files.')
    report = inspect_panel(args.data)
    save(args.out, report)
    print('Schema valid. Paper aggregate counts match:', report['paper_counts_match'])
    print('Aggregate report saved to:', args.out)


if __name__ == '__main__':
    main()
