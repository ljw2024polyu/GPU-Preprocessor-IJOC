#!/usr/bin/env python3
"""Verify archived Section 4 tables or generate CSVs for a new run; no solver is run."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import argparse
import csv
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = re.compile(r'problem information:\s*nRow\s*=\s*(\d+),\s*nCol\s*=\s*(\d+),\s*nnz A\s*=\s*(\d+)')
GROUPS = ['Mittelmann', 'MIPLIB Large', 'QAP20']
TABLES = {
    3: ('Table3_Ablation', ['None', 'Folding only', 'GPU-Preprocessor']),
    4: ('Table4_Preprocessing', ['Gurobi', 'PSLP', 'GPU-Preprocessor']),
    5: ('Table5_EndToEnd', ['None', 'Gurobi', 'PSLP', 'GPU-Preprocessor']),
    6: ('Table6_QAPScalability', ['Gurobi', 'PSLP', 'GPU-Preprocessor']),
}
METRICS = {
    3: ['solved', 'preprocessing_time_s', 'solve_time_s', 'total_time_s', 'r_m', 'r_n', 'r_nnz'],
    4: ['preprocessing_time_s', 'r_m', 'r_n', 'r_nnz'],
    5: ['solved', 'preprocessing_time_s', 'solve_time_s', 'total_time_s'],
    6: ['preprocessing_time_s', 'solve_time_s', 'total_time_s', 'r_m', 'r_n', 'r_nnz'],
}
TABLE_FILES = {
    3: 'table_3_ablation_results.csv',
    4: 'table_4_preprocessing_results.csv',
    5: 'table_5_end_to_end_results.csv',
    6: 'table_6_scalability_results.csv',
}

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def rounded(value):
    return str(Decimal(str(value)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))

def sgm10(values):
    return math.expm1(math.fsum(math.log((x + 10) / 10) for x in values) / len(values)) * 10

def scale_log(path, final_fields=False):
    text = path.read_text(errors='replace')
    def value(pattern):
        matches = re.findall(pattern, text, re.M)
        if not matches:
            raise ValueError(f'Missing {pattern} in {path}')
        return matches[-1] if final_fields else matches[0]
    patterns = [r'^GPU-Presolver-C preprocessing time: ([\d.eE+-]+) seconds',
                r'^Preprocessing Time:\s*([\d.eE+-]+) seconds',
                r'^Presolve Time:\s*([\d.eE+-]+) seconds']
    if final_fields:
        patterns[0], patterns[1] = patterns[1], patterns[0]
    prep = next((matches[-1] for p in patterns if (matches := re.findall(p, text, re.M))), None)
    assert prep is not None, path
    solve = value(r'^Solve Time:\s*([\d.eE+-]+) seconds')
    dimensions = [tuple(map(int, m)) for m in DIMENSIONS.findall(text)]
    assert len(dimensions) >= 2, path
    original, reduced = dimensions[0], dimensions[-1]
    return {
        'status': value(r'^Status:\s*(\S+)'),
        'iterations': value(r'^Iterations:\s*(\d+)'),
        'primal_objective': value(r'^Primal Objective:\s*([\d.eE+-]+)'),
        'original_dimensions': json.dumps(original),
        'reduced_dimensions': json.dumps(reduced),
        'preprocessing_time_s': prep, 'solve_time_s': solve,
        'total_time_s': str(Decimal(prep) + Decimal(solve)),
        'r_m': str(Decimal(reduced[0]) / Decimal(original[0])),
        'r_n': str(Decimal(reduced[1]) / Decimal(original[1])),
        'r_nnz': str(Decimal(reduced[2]) / Decimal(original[2])),
        'postsolve_original_kkt_warning': 'Warning: postsolve original KKT check failed' in text,
        'gurobi_original_solution_not_recovered': 'original_solution_recovery: NOT_PERFORMED' in text,
    }

def check_archived_table(number, path, records, fields):
    expected = read_csv(path)
    assert len(records) == len(expected), (
        f'Table {number}: recomputed {len(records)} rows, archived CSV has {len(expected)}')
    for index, (actual, saved) in enumerate(zip(records, expected), start=1):
        for field in fields:
            assert str(actual[field]) == saved[field], (
                f'Table {number}, row {index}, {field}: '
                f'recomputed {actual[field]!r}, archived CSV has {saved[field]!r}')

def main():
    global ROOT
    archive = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        help='Reconstruct a complete new run instead of checking archived paper results')
    args = parser.parse_args()
    if args.run_dir:
        ROOT = args.run_dir.expanduser().resolve()
        if ROOT == archive or archive / 'results' == ROOT or archive / 'results' in ROOT.parents:
            parser.error('--run-dir must be a separate experiment output directory')
    plan = None
    if args.run_dir:
        plan = json.loads((ROOT / 'commands.json').read_text())
        # Summarize exactly the selected run: unlicensed methods and unselected
        # instances/sizes are absent, rather than copied from the paper archive.
        if not plan:
            parser.error('The run command plan is empty')
    runs = read_csv(archive / 'docs/tables_3_4_5_batch_index.csv')
    aggregates = {}
    instance_rows = []
    for run in runs:
        batch = run['paper_batch_id']
        selected = [j for j in plan if j['summary'] == run['local_summary']] if plan is not None else None
        if selected == []:
            continue
        rows = [r for r in read_csv(ROOT / run['local_summary']) if r['name'] not in {'SGM10', 'solved'}]
        count = len(selected) if selected is not None else int(run['instances'])
        assert len(rows) == count, f'Incomplete {batch}: expected {count}, found {len(rows)}'
        if selected is not None:
            assert sorted(r['name'] for r in rows) == sorted(j['instance'] for j in selected), batch
        ratios = []
        prep_times, solve_times, total_times = [], [], []
        for row in rows:
            name = row['name']
            stem = re.sub(r'\.mps(?:\.gz)?$', '', name, flags=re.I)
            stem = re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')
            log = ROOT / run['local_directory'] / 'logs' / f'{batch}_{stem}.log'
            text = log.read_text(errors='replace')
            dimensions = [tuple(map(int, m)) for m in DIMENSIONS.findall(text)]
            assert dimensions, log
            original = dimensions[0]
            reduced = dimensions[1] if len(dimensions) > 1 else original
            ratio = tuple(b / a for a, b in zip(original, reduced))
            ratios.append(ratio)
            prep = float(row.get('preprocessing_time') or row.get('presolve_time') or 0)
            solve = float(row['solve_time']) if row['status'] == 'OPTIMAL' else 1000.0
            prep_times.append(prep)
            solve_times.append(solve)
            total_times.append(prep + solve)
            instance_rows.append({
                'dataset': run['dataset'], 'variant': run['variant'], 'instance': name,
                'status': row['status'], 'preprocessing_time_s': prep,
                'reported_solve_time_s': row['solve_time'], 'statistical_solve_time_s': solve,
                'statistical_total_time_s': prep + solve,
                **dict(zip(['original_rows', 'original_columns', 'original_nnz'], original)),
                **dict(zip(['reduced_rows', 'reduced_columns', 'reduced_nnz'], reduced)),
                **dict(zip(['r_m', 'r_n', 'r_nnz'], ratio)),
                'raw_log': str(log.relative_to(ROOT)), 'source_batch': batch,
                **({'postsolve_original_kkt_warning': 'Warning: postsolve original KKT check failed' in text}
                   if args.run_dir else {}),
            })
        result = {
            'dataset': run['dataset'], 'variant': run['variant'],
            'solved': sum(r['status'] == 'OPTIMAL' for r in rows),
            'preprocessing_time_s': rounded(sgm10(prep_times)),
            'solve_time_s': rounded(sgm10(solve_times)),
            'total_time_s': rounded(sgm10(total_times)),
            'source_batch': batch,
        }
        result.update({k: rounded(math.fsum(r[i] for r in ratios) / len(ratios))
                       for i, k in enumerate(['r_m', 'r_n', 'r_nnz'])})
        aggregates[run['dataset'], run['variant']] = result
    if instance_rows and args.run_dir:
        write_csv(ROOT / 'results/tables_3_4_5_instance_results.csv', instance_rows, list(instance_rows[0]))
    scale_index = read_csv(archive / 'docs/table_6_figure_2_log_index.csv')
    scale_rows = []
    scale_detailed = []
    for entry in scale_index:
        if plan is not None and entry['local_log'] not in {j['log'] for j in plan}:
            continue
        log = ROOT / entry['local_log']
        parsed = scale_log(log, final_fields=bool(args.run_dir))
        record = {'N': entry['N'], 'method': entry['method'], 'status': parsed['status'],
                  **{k: rounded(parsed[k]) for k in METRICS[6]},
                  'raw_log': str(log.relative_to(ROOT))}
        scale_rows.append(record)
        scale_detailed.append({'N': entry['N'], 'method': entry['method'],
                               **parsed, 'raw_log': str(log.relative_to(ROOT))})
    if scale_detailed and args.run_dir:
        write_csv(ROOT / 'results/Table6_QAPScalability/table_6_full_precision_results.csv',
                  scale_detailed, list(scale_detailed[0]))
    checks = []
    for number, (name, variants) in TABLES.items():
        folder = ROOT / 'results' / name
        if number == 6:
            records = scale_rows
            keys = ['N', 'method'] + (['status'] if args.run_dir else [])
            source = ['raw_log']
        else:
            records = [aggregates[group, variant] for group in GROUPS for variant in variants
                       if (group, variant) in aggregates]
            if number == 3:
                records = [dict(r, variant='Folding + presolve') if r['variant'] == 'GPU-Preprocessor' else r for r in records]
            keys = ['dataset', 'variant']
            if number == 4:
                records = [dict(r, preprocessor=r['variant']) for r in records]
                keys = ['dataset', 'preprocessor']
            elif number == 5:
                records = [dict(r, method='HPR-LP' + ('' if r['variant'] == 'None' else ' + ' + r['variant'])) for r in records]
                keys = ['dataset', 'method']
            source = ['source_batch']
        fields = keys + METRICS[number] + source
        if not records:
            continue
        if args.run_dir:
            write_csv(folder / TABLE_FILES[number], records, fields)
        else:
            check_archived_table(number, folder / TABLE_FILES[number], records, fields)
        checks.append({'table': number, 'rows': len(records),
                       'matches_archived_csv': None if args.run_dir else True})
    figure_data = [{'N': r['N'], 'method': r['method'],
                    'preprocessing_time_s': r['preprocessing_time_s'], 'raw_log': r['raw_log']}
                   for r in scale_rows]
    if figure_data and args.run_dir:
        write_csv(ROOT / 'results/Figure2_QAPScalability/figure_2_preprocessing_times.csv', figure_data, list(figure_data[0]))
    report = {'benchmark_instance_rows': len(instance_rows), 'scale_results': len(scale_rows),
              'table_rows_matching_archived_csv': None if args.run_dir else sum(c['rows'] for c in checks),
              'tables': checks}
    if args.run_dir:
        report['postsolve_original_kkt_warnings'] = sum(
            r['postsolve_original_kkt_warning'] for r in instance_rows + scale_detailed)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
