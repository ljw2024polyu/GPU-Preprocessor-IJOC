"""CPU checks for experiment selection, output protection, and log parsing."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_experiments.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExperimentRunnerTests(unittest.TestCase):
    def plan(self, *options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            runner.main(['--input-root', '/unused-inputs', '--dry-run', *options])
        return json.loads(output.getvalue())

    def test_full_plan_matches_every_archived_log(self):
        jobs = self.plan('--suite', 'all', '--methods', 'none', 'folding', 'gpu', 'pslp', 'gurobi')
        actual = [job['log'] for job in jobs]
        expected = {str(path.relative_to(ROOT)) for path in (ROOT / 'results/raw').rglob('*.log')}
        self.assertEqual(len(actual), 450)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)
        for job in jobs:
            self.assertNotIn('{', ' '.join(job['command']))
            if job['method'] == 'pslp':
                self.assertEqual(job['command'][job['command'].index('--presolver') + 1], 'pslp0011')

    def test_default_excludes_gurobi_and_subset_has_recorded_options(self):
        jobs = self.plan('--suite', 'all')
        self.assertEqual(len(jobs), 358)
        self.assertNotIn('gurobi', {job['method'] for job in jobs})
        subset = self.plan('--datasets', 'qap20', '--instances', 'tho40',
                           '--methods', 'folding', '--device', '1')
        self.assertEqual(len(subset), 1)
        command = subset[0]['command']
        for flag, expected in [('--presolver', 'gpu'), ('--gpu-folding', 'true'),
                               ('--gpu-presolve', 'false'), ('--device', '1')]:
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_reject_invalid_selection_and_archive_output(self):
        for options in [('--instances', 'does-not-exist'), ('--device', '-1'),
                        ('--suite', 'scalability', '--methods', 'folding')]:
            with self.assertRaises(SystemExit):
                self.plan(*options)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(['--input-root', '/unused-inputs', '--output-dir', str(ROOT / 'results/new')])

    def test_parse_all_archived_final_records(self):
        for path in (ROOT / 'results/raw').rglob('*.log'):
            result = runner.parse_result(path.read_text(errors='replace'), path.name)
            self.assertTrue(result['status'])
        with self.assertRaises(ValueError):
            runner.parse_result('Status: OPTIMAL\n', 'incomplete')
        for invalid in ['nan', 'inf', '-1']:
            with self.assertRaises(ValueError):
                runner.parse_result('Status: OPTIMAL\nPreprocessing Time: ' + invalid +
                                    '\nPresolve Time: 0\nSolve Time: 1\n', 'invalid')

    def test_run_writes_summary_and_preserves_environment_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'small.mps'
            model.write_text('fixture')
            fake = root / 'solver.py'
            fake.write_text(
                'import os\n'
                'assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"\n'
                'assert os.environ["HPRLP_TEST_SETTING"] == "recorded"\n'
                'assert "GPUPRESOLVER_UNRECORDED" not in os.environ\n'
                'print("Status: OPTIMAL\\nPreprocessing Time: 0.2 seconds\\n"'
                ' "Presolve Time: 0.1 seconds\\nSolve Time: 0.3 seconds")\n')
            job = dict(command=[sys.executable, str(fake)], input=str(model),
                       log='results/raw/test/logs/test_small.log',
                       summary='results/raw/test/summary.csv', dataset='fixture',
                       method='pslp', instance='small.mps',
                       environment={'HPRLP_TEST_SETTING': 'recorded'})
            output = root / 'run'
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2',
                                         'HPRLP_TEST_SETTING': 'inherited',
                                         'GPUPRESOLVER_UNRECORDED': '1'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_jobs([job], output)
            with (output / job['summary']).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['preprocessing_time'], '0.2')
            saved = (output / job['log']).read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run_jobs([job], output)
            self.assertEqual((output / job['log']).read_bytes(), saved)
            model.unlink()
            with self.assertRaises(ValueError):
                runner.run_jobs([job], root / 'missing-input')
            self.assertFalse((root / 'missing-input').exists())

    def test_failure_does_not_create_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'small.mps'
            model.touch()
            job = dict(command=[sys.executable, '-c', 'raise SystemExit(7)'], input=str(model),
                       log='results/raw/logs/test.log', summary='results/raw/summary.csv',
                       dataset='fixture', method='gpu', instance='small.mps', environment={})
            with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                runner.run_jobs([job], root / 'run')
            self.assertTrue((root / 'run' / job['log']).is_file())
            self.assertFalse((root / 'run' / job['summary']).exists())


if __name__ == '__main__':
    unittest.main()
