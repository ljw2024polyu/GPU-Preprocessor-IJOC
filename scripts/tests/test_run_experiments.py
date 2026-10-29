# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""CPU checks for experiment selection, output protection, and log parsing."""
import contextlib
import csv
import fcntl
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
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_experiments.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'WITH_GUROBI': 'AUTO', 'GUROBI_HOME': '/SDK'})
        environment.start()
        self.addCleanup(environment.stop)
        probe = patch.object(runner.check_gurobi, 'available', return_value=(False, 'unavailable'))
        self.probe = probe.start()
        self.addCleanup(probe.stop)

    def plan(self, *options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            runner.main(['--input-root', '/unused-inputs', '--dry-run', *options])
        return json.loads(output.getvalue())

    def test_full_plan_matches_every_archived_log(self):
        self.probe.return_value = (True, 'ready')
        jobs = self.plan()
        self.assertEqual(jobs, self.plan('--suite', 'all', '--methods', 'none', 'folding', 'gpu', 'pslp', 'gurobi'))
        actual = [job['log'] for job in jobs]
        expected = {str(path.relative_to(ROOT)) for path in (ROOT / 'results/raw').rglob('*.log')}
        self.assertEqual(len(actual), 450)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)
        for job in jobs:
            self.assertNotIn('{', ' '.join(job['command']))
            if job['method'] == 'pslp':
                self.assertEqual(job['command'][job['command'].index('--presolver') + 1], 'pslp0011')

    def test_default_runs_all_groups_without_gurobi_when_unavailable(self):
        jobs = self.plan()
        self.assertEqual(len(jobs), 358)
        self.assertNotIn('gurobi', {job['method'] for job in jobs})
        self.assertEqual({job['dataset'] for job in jobs},
                         {'Mittelmann', 'MIPLIB Large', 'QAP20', 'QAP scalability'})
        self.assertEqual(jobs, self.plan('--suite', 'all', '--methods', 'none', 'folding', 'gpu', 'pslp'))

    def test_benchmark_subset_has_recorded_options_and_does_not_probe_gurobi(self):
        subset = self.plan('--suite', 'benchmarks', '--datasets', 'qap20', '--instances', 'tho40',
                           '--methods', 'folding', '--device', '1')
        self.assertEqual(len(subset), 1)
        self.probe.assert_not_called()
        command = subset[0]['command']
        for flag, expected in [('--presolver', 'gpu'), ('--gpu-folding', 'true'),
                               ('--gpu-presolve', 'false'), ('--device', '1')]:
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_explicit_gurobi_selection_is_checked_and_skipped_when_unavailable(self):
        jobs = self.plan('--methods', 'gpu', 'gurobi')
        self.assertEqual(len(jobs), 92)
        self.assertEqual({job['method'] for job in jobs}, {'gpu'})
        with self.assertRaises(SystemExit):
            self.plan('--methods', 'gurobi')

    def test_gurobi_off_disables_the_probe_for_default_and_explicit_selection(self):
        with patch.dict(os.environ, {'WITH_GUROBI': 'OFF'}):
            self.assertEqual(len(self.plan()), 358)
            jobs = self.plan('--methods', 'gpu', 'gurobi')
        self.assertEqual({job['method'] for job in jobs}, {'gpu'})
        self.probe.assert_not_called()

    def test_explicit_gurobi_requirement_and_invalid_mode_fail_before_execution(self):
        for mode in ['ON', 'invalid']:
            with self.subTest(mode=mode), patch.dict(os.environ, {'WITH_GUROBI': mode}), \
                 patch.object(runner, 'run_jobs') as execute, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(['--input-root', '/unused', '--output-dir', '/unused-output'])
            execute.assert_not_called()

    def test_gurobi_probe_uses_the_selected_solver_build(self):
        self.plan('--solver', '/custom build/bin/hprlp_solve')
        self.probe.assert_called_once_with('/SDK', Path('/custom build'))

    def test_existing_collections_are_read_in_place_without_changing_experiment_options(self):
        options = ('--suite', 'all', '--instances', 'FOME13', 'a2864-99blp_relax',
                   'tho40', '--sizes', '20')
        expected = self.plan(*options)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files = {
                'FOME13.mps.gz': 'Hans49_original/FOME13.mps',
                'a2864-99blp_relax.mps.gz': 'MIP_large18_original/a2864-99blp_relax.mps',
                'tho40.mps.gz': 'QAP20_original/tho40.mps',
                'qap-wil100-n020.mps.gz': 'QAP_wil100_nested/mps/qap-wil100-n020.mps.gz',
            }
            for filename in files.values():
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('existing model fixture')
            before = sorted(root.rglob('*'))
            actual = self.plan('--input-root', str(root), *options)
            self.assertEqual(len(actual), 14)
            self.assertEqual(sorted(root.rglob('*')), before)
            for old, new in zip(expected, actual):
                model = str(root / files[new['instance']])
                self.assertEqual(new['input'], model)
                self.assertEqual(new['command'][new['command'].index('-i') + 1], model)
                expected_job = dict(old, input=model,
                                    command=[model if arg == old['input'] else arg
                                             for arg in old['command']])
                self.assertEqual(new, expected_job)

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

    def test_summary_has_two_times_and_original_and_preprocessed_dimensions(self):
        text = ('problem information: nRow = 10, nCol = 20, nnz A = 30\n'
                'problem information: nRow = 5, nCol = 12, nnz A = 18\n'
                'problem information: nRow = 4, nCol = 11, nnz A = 16\n'
                'Warning: postsolve original KKT check failed\n'
                'Status: OPTIMAL\nPresolve Time: 0.1 seconds\n'
                'Preprocessing Time: 0.25 seconds\nSolve Time: 1.5 seconds\n'
                'Status: TIME_LIMIT\nPreprocessing Time: 0.333333333333 seconds\n'
                'Solve Time: 1000.123456789 seconds\n')
        self.assertEqual(runner.parse_result(text, 'model.mps.gz'), {
            'name': 'model.mps.gz', 'status': 'TIME_LIMIT',
            'preprocessing_time': '0.333333333333', 'solve_time': '1000.123456789',
            'original_m': 10, 'original_n': 20, 'original_nnz': 30,
            'reduced_m': 5, 'reduced_n': 12, 'reduced_nnz': 18,
        })
        baseline = ('problem information: nRow = 10, nCol = 20, nnz A = 30\n'
                    'Status: OPTIMAL\nPresolve Time: 0 seconds\nSolve Time: 1 seconds\n')
        result = runner.parse_result(baseline, 'baseline.mps')
        self.assertEqual(result['preprocessing_time'], '0')
        self.assertEqual([result[k] for k in ['reduced_m', 'reduced_n', 'reduced_nnz']],
                         [10, 20, 30])

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
                'print("problem information: nRow = 3, nCol = 4, nnz A = 5")\n'
                'print("problem information: nRow = 2, nCol = 3, nnz A = 4")\n'
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
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames,
                                 ['name', 'status', 'preprocessing_time', 'solve_time',
                                  'original_m', 'original_n', 'original_nnz',
                                  'reduced_m', 'reduced_n', 'reduced_nnz'])
            self.assertEqual(rows[0]['preprocessing_time'], '0.2')
            self.assertEqual(rows[0]['reduced_nnz'], '4')
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

    def interrupted_run(self, root):
        model = root / 'model.mps'
        model.write_text('fixture')
        failure = root / 'interrupt'
        failure.touch()
        calls = root / 'solver-calls'
        solver = root / 'solver.py'
        solver.write_text(
            'import pathlib, sys\n'
            f'with pathlib.Path({str(calls)!r}).open("a") as out: out.write(sys.argv[1] + "\\n")\n'
            'print("problem information: nRow = 3, nCol = 4, nnz A = 5")\n'
            f'if sys.argv[1] == "1" and pathlib.Path({str(failure)!r}).exists():\n'
            '    print("Status: OPTIMAL\\nSolve Time: 0.1 seconds")\n'
            '    raise SystemExit(7)\n'
            'if sys.argv[1] == "0": print("Warning: postsolve original KKT check failed")\n'
            f'print({runner.FINAL_SUMMARY!r})\n'
            'print("Status: " + ("TIME_LIMIT" if sys.argv[1] == "0" else "OPTIMAL"))\n'
            'print("Preprocessing Time: 0.2 seconds\\nSolve Time: 1.3 seconds\\nTotal Time: 1.5 seconds")\n')
        jobs = [dict(command=[sys.executable, str(solver), str(index)], input=str(model),
                     log=f'results/raw/logs/model_{index}.log', summary='results/raw/summary.csv',
                     dataset='fixture', method='gpu', instance=f'model_{index}.mps', environment={})
                for index in range(3)]
        output = root / 'run'
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            runner.run_jobs(jobs, output)
        return jobs, output, failure, calls

    def test_resume_reuses_completed_logs_and_preserves_the_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs, output, failure, calls = self.interrupted_run(Path(directory))
            first_log = output / jobs[0]['log']
            saved = first_log.read_bytes()
            partial = (output / jobs[1]['log']).read_bytes()
            plan = (output / 'commands.json').read_bytes()
            # Older runs did not record exit codes in their completed logs.
            first_log.write_bytes(saved.replace(b'\n# Exit code: 0\n', b''))
            saved = first_log.read_bytes()
            failure.unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_jobs(jobs, output, resume=True)
            self.assertEqual(calls.read_text().splitlines(), ['0', '1', '1', '2'])
            self.assertEqual(first_log.read_bytes(), saved)
            self.assertEqual((output / 'commands.json').read_bytes(), plan)
            previous = list((output / 'interrupted').rglob('*.log'))
            self.assertEqual(len(previous), 1)
            self.assertEqual(previous[0].read_bytes(), partial)
            with (output / jobs[0]['summary']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['name'] for row in rows], [job['instance'] for job in jobs])
            self.assertEqual(rows[0]['status'], 'TIME_LIMIT')
            self.assertEqual(len(runner.resume_results(jobs, output)), 3)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_jobs(jobs, output, resume=True)
            self.assertEqual(calls.read_text().splitlines(), ['0', '1', '1', '2'])

    def test_resume_rejects_changed_commands_and_a_concurrent_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs, output, _, calls = self.interrupted_run(Path(directory))
            changed = json.loads(json.dumps(jobs))
            changed[0]['method'] = 'different-method'
            with self.assertRaisesRegex(ValueError, 'differ from commands.json'):
                runner.run_jobs(changed, output, resume=True)
            with (output / '.run.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'Another experiment runner'):
                    runner.run_jobs(jobs, output, resume=True)
            self.assertEqual(calls.read_text().splitlines(), ['0', '1'])

    def test_resume_does_not_reuse_a_failed_exit_even_with_a_final_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs, output, _, _ = self.interrupted_run(Path(directory))
            text = (output / jobs[0]['log']).read_text()
            (output / jobs[0]['log']).write_text(text.replace('# Exit code: 0', '# Exit code: 7'))
            self.assertEqual(runner.resume_results(jobs, output), {})


if __name__ == '__main__':
    unittest.main()
