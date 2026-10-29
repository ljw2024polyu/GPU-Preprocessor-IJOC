#!/usr/bin/env python3
# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""Run or resume Section 4 experiments, preserving completed solver logs."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from input_paths import model_path
import check_gurobi

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'hans': 'Mittelmann', 'miplib-large': 'MIPLIB Large', 'qap20': 'QAP20'}
METHODS = {'None': 'none', 'Folding only': 'folding',
           'GPU-Preprocessor': 'gpu', 'PSLP': 'pslp', 'Gurobi': 'gurobi'}
DIMENSIONS = re.compile(r'problem information:\s*nRow\s*=\s*(\d+),\s*nCol\s*=\s*(\d+),\s*nnz A\s*=\s*(\d+)')
FINAL_SUMMARY = '=== External preprocessing + official HPR-LP v0.1.3 ==='
FAILURE_STATUSES = {'ERROR', 'PRESOLVE_FAILED', 'POSTSOLVE_FAILED',
                    'UNSUPPORTED_PRESOLVER', 'REDUCED_MODEL_UNSUPPORTED'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def instance_stem(name):
    return re.sub(r'\.mps(?:\.gz)?$', '', name, flags=re.I)


def select_methods(args):
    methods = list(args.methods) if args.methods is not None else list(METHODS.values())
    mode = os.environ.get('WITH_GUROBI') or 'AUTO'
    if mode not in {'AUTO', 'ON', 'OFF'}:
        raise ValueError('WITH_GUROBI must be AUTO, ON, or OFF')
    if 'gurobi' not in methods:
        return methods
    if mode == 'OFF':
        if args.methods is not None:
            print('Gurobi skipped: WITH_GUROBI=OFF', file=sys.stderr)
        return [method for method in methods if method != 'gurobi']
    # hprlp_solve is installed in BUILD_DIR/bin by the CMake build.
    enabled, reason = check_gurobi.available(os.environ.get('GUROBI_HOME'),
                                           args.solver.parent.parent)
    print(('Gurobi enabled: ' if enabled else 'Gurobi skipped: ') + reason, file=sys.stderr)
    if not enabled:
        if mode == 'ON':
            raise ValueError('WITH_GUROBI=ON requires a usable SDK, license, and adapter build; no experiments were started')
        methods = [method for method in methods if method != 'gurobi']
    return methods


def jobs_for(args):
    inventory = read_csv(ROOT / 'data/section_4_input_inventory.csv')
    config = json.loads((ROOT / 'scripts/experiment_config.json').read_text())
    methods = set(args.methods)
    jobs = []
    if args.suite in {'benchmarks', 'all'}:
        selected = {DATASETS[key] for key in args.datasets}
        allowed = {instance_stem(row['name']) for row in inventory if row['dataset'] in selected}
        if args.instances and not set(args.instances) <= allowed:
            raise ValueError('Unknown instance in the selected datasets: ' +
                             ', '.join(sorted(set(args.instances) - allowed)))
        for run in config['benchmarks']:
            method = METHODS[run['variant']]
            if run['dataset'] not in selected or method not in methods:
                continue
            for entry in inventory:
                if entry['dataset'] != run['dataset']:
                    continue
                if args.instances and instance_stem(entry['name']) not in args.instances:
                    continue
                input_mps = model_path(args.input_root, entry['dataset'], entry['name'])
                stem = re.sub(r'[^a-z0-9]+', '_', instance_stem(entry['name']).lower()).strip('_')
                log = f"{run['local_directory']}/logs/{run['paper_batch_id']}_{stem}.log"
                command = [x.replace('{solver_executable}', str(args.solver))
                           .replace('{input_mps}', str(input_mps))
                           for x in run['command']]
                jobs.append(dict(dataset=run['dataset'], method=method, instance=entry['name'],
                                 input=str(input_mps), log=log, command=command, environment={},
                                 summary=run['local_summary']))
    if args.suite in {'scalability', 'all'}:
        if args.instances and args.suite == 'scalability':
            raise ValueError('--instances is for the benchmarks suite; use --sizes for scalability')
        for entry in config['scalability']:
            method = METHODS[entry['method']]
            if method not in methods or int(entry['N']) not in args.sizes:
                continue
            name = f"qap-wil100-n{int(entry['N']):03d}.mps.gz"
            input_mps = model_path(args.input_root, 'QAP wil100 scale', name)
            command = [x.replace('{solver_executable}', str(args.solver))
                       .replace('{input_dir}/' + name, str(input_mps))
                       for x in entry['command']]
            jobs.append(dict(dataset='QAP scalability', method=method, instance=name,
                             input=str(input_mps), log=entry['local_log'], command=command,
                             environment=entry['environment'], summary=None))
    for job in jobs:
        command = job['command']
        command[command.index('--device') + 1] = str(args.device)
    if not jobs:
        raise ValueError('The selected suite, datasets, and methods contain no experiments')
    return jobs


def parse_result(text, name):
    def field(label, fallback=None):
        matches = re.findall(r'^' + re.escape(label) + r':\s*(\S+)', text, re.M)
        if not matches:
            if fallback is not None:
                return fallback
            raise ValueError(f'Missing final {label} field in the solver output for {name}')
        return matches[-1]
    result = {'name': name, 'status': field('Status')}
    prep_label = 'Preprocessing Time' if re.search(r'^Preprocessing Time:', text, re.M) else 'Presolve Time'
    for key, label in [('preprocessing_time', prep_label), ('solve_time', 'Solve Time')]:
        value = field(label)
        if not float(value) >= 0 or not float(value) < float('inf'):
            raise ValueError(f'Invalid {label} for {name}: {value}')
        result[key] = value
    dimensions = [tuple(map(int, match)) for match in DIMENSIONS.findall(text)]
    if not dimensions:
        raise ValueError(f'Missing model dimensions in the solver output for {name}')
    original = dimensions[0]
    reduced = dimensions[1] if len(dimensions) > 1 else original
    result.update(zip(['original_m', 'original_n', 'original_nnz'], original))
    result.update(zip(['reduced_m', 'reduced_n', 'reduced_nnz'], reduced))
    return result


def resume_results(jobs, output):
    saved = json.loads((output / 'commands.json').read_text())
    if jobs != saved:
        raise ValueError('The selected commands differ from commands.json; resume with the original settings')
    completed = {}
    for job in jobs:
        log = output / job['log']
        if not log.is_file():
            continue
        text = log.read_text(errors='replace')
        if FINAL_SUMMARY not in text:
            continue
        footer = text.rsplit(FINAL_SUMMARY, 1)[1]
        if not all(re.search(r'^' + label + r':\s*\S+', footer, re.M)
                   for label in ['Status', 'Preprocessing Time', 'Solve Time', 'Total Time']):
            continue
        exit_codes = re.findall(r'^# Exit code:\s*(-?\d+)\s*$', text, re.M)
        if exit_codes and exit_codes[-1] != '0':
            continue
        try:
            result = parse_result(text, job['instance'])
        except ValueError:
            continue
        if result['status'] not in FAILURE_STATUSES:
            completed[job['log']] = result
    return completed


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_jobs(jobs, output, resume=False):
    if resume:
        if not output.is_dir():
            raise ValueError('The resume output directory does not exist')
    else:
        validate_inputs(jobs)
        output.mkdir(parents=True, exist_ok=False)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another experiment runner is using this output directory') from None
        completed = resume_results(jobs, output) if resume else {}
        pending = [job for job in jobs if job['log'] not in completed]
        if pending and resume:
            validate_inputs(pending)
        if not resume:
            (output / 'commands.json').write_text(json.dumps(jobs, indent=2) + '\n')
        else:
            print(f'Resume: {len(completed)} completed; {len(pending)} remaining', flush=True)
        execute_jobs(jobs, output, completed, resume)


def validate_inputs(jobs):
    if not Path(jobs[0]['command'][0]).is_file() or not os.access(jobs[0]['command'][0], os.X_OK):
        raise ValueError('Solver executable is missing; build hprlp_solve or set --solver')
    missing = [job['input'] for job in jobs if not Path(job['input']).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} input files are missing; first missing file: {missing[0]}')


def execute_jobs(jobs, output, completed, resume):
    summaries = {}
    for job in jobs:
        if job['summary'] and job['log'] in completed:
            summaries.setdefault(job['summary'], []).append(completed[job['log']])
    for name, rows in summaries.items():
        write_summary(output / name, rows)
    for index, job in enumerate(jobs, 1):
        if job['log'] in completed:
            continue
        log = output / job['log']
        log.parent.mkdir(parents=True, exist_ok=True)
        if resume and log.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            previous = output / 'interrupted' / job['log']
            previous = previous.with_name(previous.stem + '.' + stamp + previous.suffix)
            previous.parent.mkdir(parents=True, exist_ok=True)
            log.rename(previous)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('HPRLP_', 'GPUPREPROCESSOR_'))}
        env.update(job['environment'])
        print(f"[{index}/{len(jobs)}] {job['dataset']} / {job['method']} / {job['instance']}", flush=True)
        with log.open('x') as handle:
            handle.write('# Command: ' + shlex.join(job['command']) + '\n')
            handle.flush()
            process = subprocess.run(job['command'], stdout=handle, stderr=subprocess.STDOUT, env=env)
            handle.write(f'\n# Exit code: {process.returncode}\n')
        if process.returncode:
            raise RuntimeError(f'Solver exited with code {process.returncode}; see {log}')
        result = parse_result(log.read_text(errors='replace'), job['instance'])
        if result['status'] in FAILURE_STATUSES:
            raise RuntimeError(f"Solver returned {result['status']}; see {log}")
        completed[job['log']] = result
        if job['summary']:
            rows = [completed[item['log']] for item in jobs
                    if item['summary'] == job['summary'] and item['log'] in completed]
            write_summary(output / job['summary'], rows)
        print(f"  {result['status']}; log: {log}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['benchmarks', 'scalability', 'all'], default='all',
                        help='Experiment groups to run (default: all)')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='New output directory, or an existing one with --resume')
    parser.add_argument('--resume', action='store_true', help='Reuse completed logs in the same saved command plan')
    parser.add_argument('--solver', type=Path, default=ROOT / 'build-experiments/bin/hprlp_solve')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--methods', nargs='+', choices=list(METHODS.values()),
                        help='Methods to compare (default: all; Gurobi requires SDK/license preflight)')
    parser.add_argument('--instances', nargs='+', help='Benchmark names without .mps.gz; does not filter scalability')
    parser.add_argument('--sizes', nargs='+', type=int, choices=[20, 40, 60, 80, 100], default=[20, 40, 60, 80, 100])
    parser.add_argument('--device', type=int, default=0, help='Index among visible CUDA devices')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running benchmark models; Gurobi preflight still runs')
    args = parser.parse_args(argv)
    try:
        if args.device < 0:
            raise ValueError('--device must be nonnegative')
        args.input_root = args.input_root.expanduser().resolve()
        args.solver = args.solver.expanduser().resolve()
        args.methods = select_methods(args)
        jobs = jobs_for(args)
        if args.dry_run:
            reused = {}
            if args.resume:
                if args.output_dir is None:
                    raise ValueError('--output-dir is required with --resume')
                reused = resume_results(jobs, args.output_dir.expanduser().resolve())
                jobs = [job for job in jobs if job['log'] not in reused]
            print(json.dumps(jobs, indent=2))
            print(f'{len(jobs)} experiments selected' +
                  (f'; {len(reused)} completed logs reused' if args.resume else ''), file=sys.stderr)
        else:
            if args.output_dir is None:
                raise ValueError('--output-dir is required unless --dry-run is used')
            output = args.output_dir.expanduser().resolve()
            if output == ROOT / 'results' or ROOT / 'results' in output.parents:
                raise ValueError('Use runs/ or an external directory; the archived results are read-only')
            run_jobs(jobs, output, resume=args.resume)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
