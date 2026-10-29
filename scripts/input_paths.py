# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""Locate prepared models without copying or converting the input files."""
from pathlib import Path


DIRECTORIES = {
    'Mittelmann': ('Mittelmann', 'Hans49_original'),
    'MIPLIB Large': ('MIPLIB_Large', 'MIP_large18_original'),
    'QAP20': ('QAP20', 'QAP20_original'),
    'QAP wil100 scale': ('QAP_wil100_scale', 'QAP_wil100_nested/mps'),
}


def model_path(root, dataset, name):
    """Accept compressed or plain MPS files in prepared or collection folders."""
    root = Path(root)
    names = (name, name[:-3]) if name.endswith('.gz') else (name,)
    for folder in DIRECTORIES[dataset]:
        for filename in names:
            candidate = root / folder / filename
            if candidate.is_file():
                return candidate
    return root / DIRECTORIES[dataset][0] / name
