"""Check result reconstruction on a real selected run without Gurobi."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rebuild', ROOT / 'scripts/rebuild_results.py')
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


class RebuildResultsTests(unittest.TestCase):
    def selected_run(self, directory):
        plan = json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / 'scripts/run_experiments.py'),
             '--input-root', '/unused', '--suite', 'all', '--dry-run',
             '--instances', 'FOME13', 'scpn2_relax', 'tho40', '--sizes', '20'],
            text=True, stderr=subprocess.DEVNULL))
        directory.joinpath('commands.json').write_text(json.dumps(plan))
        summaries = {}
        for job in plan:
            log = directory / job['log']
            log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / job['log'], log)
            if job['summary']:
                summaries.setdefault(job['summary'], set()).add(job['instance'])
        for name, instances in summaries.items():
            with (ROOT / name).open(newline='') as source:
                reader = csv.DictReader(source)
                rows = [r for r in reader if r['name'] in instances]
                with (directory / name).open('w', newline='') as out:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        return plan

    def test_selected_methods_and_instances_reconstruct_without_gurobi(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.selected_run(run)
            completed = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/rebuild_results.py'), '--run-dir', str(run)],
                text=True, capture_output=True, check=True)
            report = json.loads(completed.stdout)
            self.assertEqual(report['benchmark_instance_rows'], 12)
            self.assertEqual(report['scale_results'], 2)
            self.assertEqual({r['table']: r['rows'] for r in report['tables']},
                             {3: 9, 4: 6, 5: 9, 6: 2})
            with (run / 'results/Figure2_QAPScalability/figure_2_preprocessing_times.csv').open() as stream:
                self.assertEqual({r['method'] for r in csv.DictReader(stream)},
                                 {'PSLP', 'GPU-Preprocessor'})

    def test_an_incomplete_selected_run_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            plan = self.selected_run(run)
            (run / plan[0]['log']).unlink()
            completed = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/rebuild_results.py'), '--run-dir', str(run)],
                text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)

    def test_final_status_and_precision_are_read_from_the_external_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'scale.log'
            log.write_text('problem information: nRow = 3, nCol = 4, nnz A = 5\n'
                           'problem information: nRow = 2, nCol = 3, nnz A = 4\n'
                           'Status: OPTIMAL\nIterations: 10\nPrimal Objective: 2\n'
                           'Preprocessing Time: 0.1 seconds\nSolve Time: 0.2 seconds\n'
                           'Status: TIME_LIMIT\nIterations: 20\nPrimal Objective: 3\n'
                           'Preprocessing Time: 0.123456789 seconds\nSolve Time: 0.3 seconds\n')
            result = rebuild.scale_log(log, final_fields=True)
            self.assertEqual(result['status'], 'TIME_LIMIT')
            self.assertEqual(result['iterations'], '20')
            self.assertEqual(result['solve_time_s'], '0.3')
            self.assertEqual(result['preprocessing_time_s'], '0.123456789')


if __name__ == '__main__':
    unittest.main()
