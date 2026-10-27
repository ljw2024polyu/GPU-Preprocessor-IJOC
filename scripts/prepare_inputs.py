#!/usr/bin/env python3
"""Download and prepare the four Section 4 input groups."""
import argparse
import bz2
import csv
import gzip
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MIPLIB = 'https://miplib.zib.de/WebData/instances/'
QAPLIB = 'https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/'
HINDER_COMMIT = 'b44862990e19ddaeab5568a826bd4ced13f23819'
HINDER = f'https://raw.githubusercontent.com/ohinder/large-scale-LP-test-problems/{HINDER_COMMIT}/'
GROUPS = {'hans': 'Mittelmann', 'miplib-large': 'MIPLIB Large',
          'qap20': 'QAP20', 'qap-scale': 'QAP wil100 scale'}
FOLDERS = {'hans': 'Mittelmann', 'miplib-large': 'MIPLIB_Large',
           'qap20': 'QAP20', 'qap-scale': 'QAP_wil100_scale'}


def inventory(group, selected):
    with (ROOT / 'data/mittelmann_sources.csv').open(newline='') as stream:
        hans = {r['name']: r for r in csv.DictReader(stream)}
    with (ROOT / 'data/section_4_input_inventory.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['dataset'] == GROUPS[group]]
    for row in rows:
        row['instance'] = row['name'].removesuffix('.mps.gz')
        if group == 'hans':
            row.update(hans[row['name']])
            row['source_name'] = row['url'].rsplit('/', 1)[1]
        elif group == 'miplib-large':
            row['source_name'] = row['instance'].removesuffix('_relax') + '.mps.gz'
            row['url'] = MIPLIB + row['source_name']
        elif group == 'qap20':
            row['source_name'] = row['instance'] + '.dat'
            row['url'] = QAPLIB + row['source_name']
        else:
            row['source_name'] = 'wil100.dat'
            row['url'] = QAPLIB + 'wil100.dat'
    if selected:
        unknown = set(selected) - {r['instance'] for r in rows}
        if unknown:
            raise ValueError('Unknown paper instance(s): ' + ', '.join(sorted(unknown)))
        rows = [r for r in rows if r['instance'] in selected]
    return rows


def download(url, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    os.close(handle)
    try:
        print(f'Download {url}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--show-error', '--silent',
                        '--retry', '3', '--connect-timeout', '30', '--output', temporary, url], check=True)
        if Path(temporary).stat().st_size == 0:
            raise ValueError(f'Empty download: {url}')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_dat(path):
    values = [int(v) for v in path.read_text().split()]
    if not values or values[0] < 2:
        raise ValueError(f'Invalid QAPLIB DAT: {path}')
    n = values[0]
    if len(values) != 1 + 2*n*n:
        raise ValueError(f'Expected two {n}-by-{n} matrices in {path}')
    matrices = [[[values[offset + i*n + j] for j in range(n)] for i in range(n)]
                for offset in (1, 1+n*n)]
    return n, matrices


def require_executable(command):
    if shutil.which(command) is None:
        raise RuntimeError(f'Executable not found: {command}. See data/README.md.')


def export_mip(source, output):
    import gurobipy as gp
    if gp.gurobi.version() != (12, 0, 2):
        raise RuntimeError('MIPLIB export requires gurobipy==12.0.2, as used in the paper.')
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.read(str(source), env=env) as original:
            if original.NumQConstrs or original.NumQNZs or original.NumSOS or original.NumGenConstrs:
                raise ValueError(f'{source.name} is not a linear MIP without SOS/general constraints')
            with original.relax() as relaxed:
                # Deliberately no optimize() or presolve(): export the original LP relaxation.
                relaxed.write(str(output))
                print(f'{output.name}: rows={relaxed.NumConstrs}, columns={relaxed.NumVars}, '
                      f'nonzeros={relaxed.DNumNZs:.0f}', flush=True)


def prepare_hans(source, output, row, args, cache, work):
    if row['format'] == 'mip-gz':
        export_mip(source, output)
        return
    unpack = bz2.open if row['format'].endswith('bz2') else gzip.open
    if row['format'].startswith('emps-'):
        require_executable(args.cc)
        emps_source = cache / 'emps.c'
        download('https://netlib.org/lp/data/emps.c', emps_source)
        emps = cache / 'emps'
        if not emps.is_file():
            subprocess.run([args.cc, '-O2', str(emps_source), '-o', str(emps)], check=True)
        packed = work / 'packed.lp'
        with unpack(source, 'rb') as inp, packed.open('wb') as out:
            shutil.copyfileobj(inp, out, 1 << 20)
        # EMPS keeps extension lines by default and expands names/numbers only.
        with (work / 'expanded.mps').open('wb') as out:
            subprocess.run([str(emps), str(packed)], stdout=out, check=True)
        inp_context = (work / 'expanded.mps').open('rb')
    else:
        inp_context = unpack(source, 'rb')
    with inp_context as inp, output.open('wb') as raw_out:
        with gzip.GzipFile(filename='', fileobj=raw_out, mode='wb', mtime=0) as out:
            shutil.copyfileobj(inp, out, 1 << 20)


def build_scale_tools(args, cache):
    require_executable(args.fc)
    require_executable(args.cxx)
    source = cache / ('hinder-' + HINDER_COMMIT[:12])
    for name in ['newlp-300.f', 'LICENSE']:
        download(HINDER + name, source / name)
    generator = source / 'qap-gen'
    scaler = source / 'rescale-qap'
    subprocess.run([args.fc, *shlex.split(args.fflags), '-o', str(generator), str(source / 'newlp-300.f')], check=True)
    subprocess.run([args.cxx, '-O3', '-std=c++17', '-o', str(scaler),
                    str(ROOT / 'scripts/qap_scale/rescale.cpp')], check=True)
    return generator, scaler


def generate_scale(source, n, output, tools, work):
    master_n, (flow, distance) = read_dat(source)
    if master_n != 100 or n > master_n:
        raise ValueError('The scale experiment requires QAPLIB wil100 (dimension 100).')
    for matrix in (flow, distance):
        if any(matrix[i][i] != 0 for i in range(n)) or any(
                matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n)):
            raise ValueError('The Hinder generator requires symmetric matrices with zero diagonal.')
    tj = work / 'cropped.tj'
    with tj.open('w') as stream:
        stream.write(str(n) + '\n')
        for i in range(n):
            stream.write(','.join(str(flow[i][j] if i > j else distance[i][j] if i < j else 0)
                                  for j in range(n)) + '\n')
    raw = work / 'unscaled.mps'
    with tj.open('rb') as inp, raw.open('wb') as out:
        subprocess.run([str(tools[0])], stdin=inp, stdout=out, check=True)
    # Stream column scaling and gzip compression; do not materialize a second full MPS.
    with output.open('wb') as out:
        producer = subprocess.Popen([str(tools[1]), str(raw), str(n)], stdout=subprocess.PIPE)
        try:
            compressor = subprocess.run(['gzip', '-n', '-c'], stdin=producer.stdout, stdout=out)
        finally:
            producer.stdout.close()
            code = producer.wait()
        if code or compressor.returncode:
            raise RuntimeError(f'Scale generation failed: scaler={code}, gzip={compressor.returncode}')
    print(f'{output.name}: rows={2*n + 2*n*n*(n-1)}, '
          f'columns={n*n + n*n*(n-1)**2//2}, nonzeros={2*n**3 + 2*n*n*(n-1)**2}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=GROUPS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/inputs', help='Input root; datasets get separate subdirectories.')
    parser.add_argument('--instances', nargs='+', help='Paper instance names without .mps.gz; use --list to see them.')
    parser.add_argument('--list', action='store_true', help='Print exact URLs and output names; do not download or write files.')
    parser.add_argument('--download-only', action='store_true', help='Download source MPS/DAT files without generating LPs.')
    parser.add_argument('--resume', action='store_true', help='Keep already completed output models and prepare the missing ones.')
    parser.add_argument('--source-dir', type=Path, help='Use source files from this directory, without downloading them.')
    parser.add_argument('--julia', default='julia', help='Julia executable for QAP20.')
    parser.add_argument('--fc', default=os.environ.get('FC', 'gfortran'), help='Fortran compiler executable for scale.')
    parser.add_argument('--fflags', default='-O3 -std=legacy -mcmodel=large', help='Fortran compiler flags (Linux x86-64 defaults).')
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'), help='C++17 compiler for scale.')
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'), help='C compiler for the Hans EMPS utility.')
    args = parser.parse_args()
    rows = inventory(args.group, args.instances)
    if args.list:
        for row in rows:
            print(f"{row['instance']}\t{row['url']}\t{FOLDERS[args.group]}/{row['name']}")
        if args.group == 'qap-scale':
            print(f'Generator: {HINDER}newlp-300.f')
        return
    destination = args.output_dir.resolve() / FOLDERS[args.group]
    raw_dir = args.source_dir.resolve() if args.source_dir else args.output_dir.resolve() / 'sources' / args.group
    require_executable('curl') if not args.source_dir or args.group == 'qap-scale' else None
    if not args.download_only:
        require_executable('gzip')
        if args.group == 'qap20':
            require_executable(args.julia)
        if args.group == 'miplib-large':
            import gurobipy as gp
            if gp.gurobi.version() != (12, 0, 2):
                raise RuntimeError('Install gurobipy==12.0.2 before preparing MIPLIB Large.')
    tools = None
    for row in rows:
        output = destination / row['name']
        if output.exists() and not args.download_only:
            if args.resume:
                with gzip.open(output, 'rb') as stream:
                    if not stream.read(1):
                        raise ValueError(f'Empty output: {output}')
                print(f'Keep {output}', flush=True)
                continue
            raise FileExistsError(f'Refusing to overwrite {output}; choose a new --output-dir.')
        source = raw_dir / row['source_name']
        if args.source_dir:
            if not source.is_file():
                raise FileNotFoundError(source)
        else:
            download(row['url'], source)
        if args.group == 'hans':
            unpack = bz2.open if row['format'].endswith('bz2') else gzip.open
            with unpack(source, 'rb') as stream:
                if not stream.read(1):
                    raise ValueError(f'Empty source: {source}')
        elif args.group == 'miplib-large':
            with gzip.open(source, 'rb') as stream:
                stream.read(1)
        else:
            read_dat(source)
        if args.download_only:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if args.group == 'qap-scale' and tools is None:
            tools = build_scale_tools(args, args.output_dir.resolve() / '.tools')
        with tempfile.TemporaryDirectory(prefix='.generate-', dir=destination) as tmp:
            work = Path(tmp)
            temporary = work / output.name
            if args.group == 'hans':
                prepare_hans(source, temporary, row, args,
                             args.output_dir.resolve() / '.tools', work)
            elif args.group == 'miplib-large':
                export_mip(source, temporary)
            elif args.group == 'qap20':
                subprocess.run([args.julia, str(ROOT / 'scripts/generate_qap20.jl'),
                                str(source), str(temporary)], check=True)
            else:
                n = int(row['instance'].rsplit('n', 1)[1])
                generate_scale(source, n, temporary, tools, work)
            temporary.rename(output)
            print(f'Created {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        sys.exit(f'ERROR: {exc}')
