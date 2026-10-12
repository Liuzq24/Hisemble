#!/usr/bin/env python3
"""Export the paper's original P metric and pre-update Q timing from fixed-W logs."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
VIEWS = ('scBasset', 'SnapATAC2', 'LSI')

def original_rows(raw, iterations=50):
    groups = defaultdict(list)
    for row in raw:
        groups[(row['Dataset'], row['View'])].append(row)
    result = []
    for (dataset, view), values in sorted(groups.items()):
        if view not in VIEWS:
            raise ValueError(f'Unexpected view: {view}')
        values.sort(key=lambda r: int(r['Iteration']))
        indices = [int(r['Iteration']) for r in values]
        if indices != list(range(1, len(values) + 1)) or len(values) < iterations:
            raise ValueError(f'{dataset}/{view}: incomplete or duplicated trajectory')
        for i, row in enumerate(values[:iterations]):
            p = float(row['legacy_scaled_absolute_L1'])
            expected = float(row['P_delta_l1']) / (int(row['n_cells']) * int(row['q_retained']))
            if not math.isclose(p, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f'{dataset}/{view}: P metric inconsistent with N*q')
            q = 0.0 if i == 0 else float(values[i - 1]['Q_delta_fro'])
            if not all(math.isfinite(v) and v >= 0 for v in (p, q)):
                raise ValueError(f'{dataset}/{view}: invalid metric')
            result.append({'Dataset': dataset, 'Iteration': i + 1, 'View': view,
                           'Relative Change': row['legacy_scaled_absolute_L1'],
                           'Delta_Q': '0' if i == 0 else values[i - 1]['Q_delta_fro']})
    for dataset in {key[0] for key in groups}:
        if {key[1] for key in groups if key[0] == dataset} != set(VIEWS):
            raise ValueError(f'{dataset}: all three views required')
    return result

def write_csv(path, rows):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--require-all', action='store_true', help='Require all 24 benchmark datasets')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    files = sorted(args.input_dir.glob('*/convergence_history.csv'))
    if (args.input_dir / 'convergence_history.csv').is_file():
        files.append(args.input_dir / 'convergence_history.csv')
    if not files:
        parser.error('No convergence_history.csv files found')
    raw = []
    for path in files:
        with path.open(newline='') as f:raw.extend(csv.DictReader(f))
    rows = original_rows(raw, args.iterations)
    datasets = {row['Dataset'] for row in rows}
    if args.require_all and datasets != set(json.loads((HERE / 'datasets.json').read_text())):
        parser.error('Dataset set differs from the 24 benchmark datasets')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'all_convergence_results.csv', rows)
    checkpoints = [r for r in rows if r['Iteration'] in (20, 50)]
    if checkpoints:write_csv(args.output_dir / 'convergence_checkpoints.csv', checkpoints)
    stats = {}
    for t in sorted({r['Iteration'] for r in checkpoints}):
        values = [r for r in checkpoints if r['Iteration'] == t]
        stats[t] = {metric: {'min': min(x := [float(r[metric]) for r in values]),
                            'median': median(x), 'max': max(x)}
                    for metric in ('Relative Change', 'Delta_Q')}
    (args.output_dir / 'summary_statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(f'Exported {len(rows)} records: {len(datasets)} datasets, 3 views, {args.iterations} iterations; original metrics.')

if __name__ == '__main__':
    main()
