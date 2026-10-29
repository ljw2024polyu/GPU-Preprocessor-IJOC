#!/usr/bin/env python3
# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""Check the separately installed C SDK and license without exposing credentials."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys

VERSION = (12, 0, 2)
RESULT = 'GUROBI_CHECK_RESULT '


class Unavailable(RuntimeError):
    pass


def sdk_library(home):
    if not home:
        raise Unavailable('set GUROBI_HOME to the Gurobi 12.0.2 C SDK platform directory')
    root = Path(home).expanduser()
    header = root / 'include/gurobi_c.h'
    if not header.is_file():
        raise Unavailable('GUROBI_HOME does not contain include/gurobi_c.h')
    text = header.read_text()
    version = []
    for key in ['MAJOR', 'MINOR', 'TECHNICAL']:
        match = re.search(r'^\s*#\s*define\s+GRB_VERSION_' + key + r'\s+(\d+)\b', text, re.M)
        if not match:
            raise Unavailable('cannot determine the C SDK header version')
        version.append(int(match[1]))
    if tuple(version) != VERSION:
        raise Unavailable('the paper experiments require Gurobi 12.0.2')
    for name in ['lib/libgurobi120.so', 'lib/libgurobi120.dylib', 'bin/gurobi120.dll']:
        library = root / name
        if library.is_file():
            return library.resolve()
    raise Unavailable('GUROBI_HOME does not contain the Gurobi 12.0 C shared library')


def check_build(build_dir):
    cache = Path(build_dir).expanduser() / 'CMakeCache.txt'
    if not cache.is_file() or not re.search(r'^HPRLP_GUROBI:BOOL=(ON|TRUE|YES|1)$',
                                          cache.read_text(), re.M):
        raise Unavailable('this build has no Gurobi adapter; rerun scripts/reproduce.sh setup with the SDK and license configured')


def probe_library(path):
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    signatures = {
        'GRBversion': ([ctypes.POINTER(integer)] * 3, None),
        'GRBemptyenvinternal': ([ctypes.POINTER(pointer), integer, integer, integer], integer),
        'GRBsetintparam': ([pointer, ctypes.c_char_p, integer], integer),
        'GRBsetdblparam': ([pointer, ctypes.c_char_p, ctypes.c_double], integer),
        'GRBstartenv': ([pointer], integer),
        'GRBnewmodel': ([pointer, ctypes.POINTER(pointer), ctypes.c_char_p, integer] + [pointer] * 5, integer),
        'GRBoptimize': ([pointer], integer),
        'GRBgetintattr': ([pointer, ctypes.c_char_p, ctypes.POINTER(integer)], integer),
        'GRBfreemodel': ([pointer], None),
        'GRBfreeenv': ([pointer], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    major, minor, patch = integer(), integer(), integer()
    lib.GRBversion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    if (major.value, minor.value, patch.value) != VERSION:
        raise Unavailable('the C runtime version must also be Gurobi 12.0.2')

    def checked(code, operation):
        if code:
            # Runtime error details can contain server or licensing information.
            raise Unavailable(f'{operation} failed (Gurobi error {code})')

    env, model = pointer(), pointer()
    try:
        # The SDK header expands GRBemptyenv to this versioned C entry point.
        checked(lib.GRBemptyenvinternal(ctypes.byref(env), *VERSION), 'environment creation')
        checked(lib.GRBsetintparam(env, b'OutputFlag', 0), 'quiet license check')
        checked(lib.GRBsetintparam(env, b'Threads', 1), 'license probe configuration')
        checked(lib.GRBsetintparam(env, b'Presolve', 0), 'license probe configuration')
        checked(lib.GRBsetdblparam(env, b'TimeLimit', 5.0), 'license probe configuration')
        checked(lib.GRBstartenv(env), 'license initialization')
        # A trivial LP just above the 2,000-variable restricted-license limit.
        # This CPU-only probe is separate from every paper experiment.
        checked(lib.GRBnewmodel(env, ctypes.byref(model), b'license_probe', 2001,
                                None, None, None, None, None), 'license probe model creation')
        checked(lib.GRBoptimize(model), 'full-size license check')
        status = integer()
        checked(lib.GRBgetintattr(model, b'Status', ctypes.byref(status)), 'license probe status')
        if status.value != 2:
            raise Unavailable('the license probe did not finish successfully')
    finally:
        if model:
            lib.GRBfreemodel(model)
        if env:
            lib.GRBfreeenv(env)


def available(home, build_dir=None, timeout=30):
    try:
        library = sdk_library(home)
        if build_dir is not None:
            check_build(build_dir)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    '--probe', str(library)], capture_output=True,
                                   text=True, timeout=timeout)
        replies = [json.loads(line[len(RESULT):]) for line in completed.stdout.splitlines()
                   if line.startswith(RESULT)]
        if completed.returncode or len(replies) != 1:
            raise Unavailable('the isolated SDK/license probe could not complete')
        return replies[0]['available'], replies[0]['reason']
    except subprocess.TimeoutExpired:
        return False, 'the SDK/license check timed out'
    except (Unavailable, OSError, ValueError, KeyError) as error:
        return False, str(error) if isinstance(error, Unavailable) else 'the SDK/license check could not complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, help='Also require a build configured with HPRLP_GUROBI=ON')
    parser.add_argument('--probe', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        try:
            probe_library(args.probe)
            result = dict(available=True, reason='Gurobi 12.0.2 SDK and full-size license check passed')
        except Unavailable as error:
            result = dict(available=False, reason=str(error))
        except (OSError, AttributeError):
            result = dict(available=False, reason='the C runtime could not be loaded or is incompatible')
        print(RESULT + json.dumps(result))
        return 0
    enabled, reason = available(os.environ.get('GUROBI_HOME'), args.build_dir)
    print(('Gurobi enabled: ' if enabled else 'Gurobi skipped: ') + reason,
          file=sys.stdout if enabled else sys.stderr)
    return 0 if enabled else 1


if __name__ == '__main__':
    sys.exit(main())
