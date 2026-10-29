# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""CPU coverage of optional SDK detection and full-size license preflight."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import ANY, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_gurobi as check


class GurobiAvailabilityTests(unittest.TestCase):
    def sdk(self, root, version=(12, 0, 2)):
        (root / 'include').mkdir()
        (root / 'lib').mkdir()
        (root / 'include/gurobi_c.h').write_text('\n'.join(
            f'#define GRB_VERSION_{key} {value}' for key, value in
            zip(['MAJOR', 'MINOR', 'TECHNICAL'], version)))
        library = root / 'lib/libgurobi120.so'
        library.touch()
        return library

    def library(self, version=(12, 0, 2), start_error=0, solve_error=0):
        library = Mock()
        def version_call(*pointers):
            for pointer, value in zip(pointers, version):
                pointer._obj.value = value
        library.GRBversion.side_effect = version_call
        def empty(env, *version):
            env._obj.value = 1
            return 0
        def model(env, pointer, *args):
            pointer._obj.value = 2
            return 0
        def status(model, attr, pointer):
            pointer._obj.value = 2
            return 0
        library.GRBemptyenvinternal.side_effect = empty
        library.GRBnewmodel.side_effect = model
        library.GRBgetintattr.side_effect = status
        library.GRBsetintparam.return_value = library.GRBsetdblparam.return_value = 0
        library.GRBstartenv.return_value = start_error
        library.GRBoptimize.return_value = solve_error
        return library

    def test_sdk_requires_headers_and_exact_library_version(self):
        with self.assertRaises(check.Unavailable):
            check.sdk_library(None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(check.Unavailable):
                check.sdk_library(root)
            library = self.sdk(root)
            self.assertEqual(check.sdk_library(root), library.resolve())
            library.unlink()
            with self.assertRaises(check.Unavailable):
                check.sdk_library(root)
        with tempfile.TemporaryDirectory() as temporary:
            self.sdk(Path(temporary), (12, 0, 1))
            with self.assertRaisesRegex(check.Unavailable, '12.0.2'):
                check.sdk_library(temporary)

    def test_license_probe_checks_a_full_size_model_and_releases_resources(self):
        library = self.library()
        with patch.object(ctypes, 'CDLL', return_value=library):
            check.probe_library('/SDK/library')
        self.assertEqual(library.GRBnewmodel.call_args.args[3], 2001)
        library.GRBemptyenvinternal.assert_called_once_with(ANY, 12, 0, 2)
        library.GRBsetintparam.assert_any_call(ANY, b'OutputFlag', 0)
        library.GRBoptimize.assert_called_once()
        library.GRBfreemodel.assert_called_once()
        library.GRBfreeenv.assert_called_once()

    def test_missing_or_expired_license_does_not_attempt_a_model(self):
        library = self.library(start_error=10009)
        with patch.object(ctypes, 'CDLL', return_value=library), \
             self.assertRaisesRegex(check.Unavailable, 'license initialization.*10009'):
            check.probe_library('/SDK/library')
        library.GRBnewmodel.assert_not_called()
        library.GRBfreemodel.assert_not_called()
        library.GRBfreeenv.assert_called_once()

    def test_restricted_license_is_rejected_and_cleaned_up(self):
        library = self.library(solve_error=10010)
        with patch.object(ctypes, 'CDLL', return_value=library), \
             self.assertRaisesRegex(check.Unavailable, 'full-size license check.*10010'):
            check.probe_library('/SDK/library')
        library.GRBfreemodel.assert_called_once()
        library.GRBfreeenv.assert_called_once()

    def test_wrong_runtime_is_rejected_before_license_checkout(self):
        library = self.library(version=(13, 0, 0))
        with patch.object(ctypes, 'CDLL', return_value=library), self.assertRaises(check.Unavailable):
            check.probe_library('/SDK/library')
        library.GRBemptyenvinternal.assert_not_called()

    def test_missing_adapter_build_does_not_activate_license(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.sdk(Path(temporary))
            with patch.object(check.subprocess, 'run') as run:
                available, reason = check.available(temporary, temporary)
            self.assertFalse(available)
            self.assertIn('rerun', reason)
            run.assert_not_called()

    def test_probe_uses_a_timeout_and_does_not_forward_license_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.sdk(Path(temporary))
            output = 'private runtime text\n' + check.RESULT + json.dumps({'available': True, 'reason': 'ready'})
            with patch.object(check.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output, 'private error')) as run:
                self.assertEqual(check.available(temporary), (True, 'ready'))
                self.assertEqual(run.call_args.kwargs['timeout'], 30)
                self.assertTrue(run.call_args.kwargs['capture_output'])
            with patch.object(check.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 30)):
                self.assertEqual(check.available(temporary), (False, 'the SDK/license check timed out'))


if __name__ == '__main__':
    unittest.main()
