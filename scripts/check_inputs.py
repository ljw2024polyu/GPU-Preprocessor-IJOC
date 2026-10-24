#!/usr/bin/env python3
"""Read prepared LPs on CPU and compare their dimensions with the paper inputs."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {'Mittelmann': 'Mittelmann', 'MIPLIB Large': 'MIPLIB_Large',
           'QAP20': 'QAP20', 'QAP wil100 scale': 'QAP_wil100_scale'}


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--solver', type=Path, default=ROOT / 'build-experiments/bin/hprlp_solve')
    parser.add_argument('--available-only', action='store_true', help='Check a prepared subset instead of requiring all 92 models.')
    args = parser.parse_args()
    expected = {}
    for row in read_csv(ROOT / 'results/tables_3_4_5_instance_results.csv'):
        if row['variant'] == 'None':
            expected[row['dataset'], row['instance']] = tuple(
                int(row['original_' + k]) for k in ['rows', 'columns', 'nnz'])
    for row in read_csv(ROOT / 'results/Table6_QAPScalability/table_6_full_precision_results.csv'):
        expected['QAP wil100 scale', f"qap-wil100-n{int(row['N']):03d}.mps.gz"] = tuple(
            json.loads(row['original_dimensions']))
    checked = 0
    failures = []
    for row in read_csv(ROOT / 'data/section_4_input_inventory.csv'):
        path = args.input_root / FOLDERS[row['dataset']] / row['name']
        if args.available_only and not path.is_file():
            continue
        if not path.is_file():
            failures.append(f'Missing {path}')
            continue
        result = subprocess.run([str(args.solver.resolve()), '-i', str(path.resolve()),
                                 '--inspect-input'], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        match = re.search(r'^INPUT_INFO (.+)$', result.stdout, re.M)
        if result.returncode or not match:
            failures.append(f'Cannot parse {path}: {result.stdout[-1000:]}')
            continue
        info = json.loads(match.group(1))
        actual = tuple(info[k] for k in ['rows', 'columns', 'nonzeros'])
        target = expected[row['dataset'], row['name']]
        if actual != target:
            failures.append(f'{row["name"]}: got {actual}, expected {target}')
        else:
            checked += 1
            print(f'OK {row["name"]}: rows={actual[0]}, columns={actual[1]}, nonzeros={actual[2]}', flush=True)
    if failures:
        raise ValueError('\n'.join(failures))
    if not checked:
        raise ValueError('No prepared inputs were found')
    print(f'{checked} inputs match the paper dimensions. No GPU was used.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f'ERROR: {exc}')
