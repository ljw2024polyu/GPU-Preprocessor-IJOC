"""Check archived CSV verification and reconstruction of a selected run."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rebuild', ROOT / 'scripts/rebuild_results.py')
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


class RebuildResultsTests(unittest.TestCase):
    def archived_results(self, directory):
        (directory / 'scripts').symlink_to(ROOT / 'scripts', target_is_directory=True)
        (directory / 'results').mkdir()
        (directory / 'results/raw').symlink_to(ROOT / 'results/raw', target_is_directory=True)
        paths = []
        for number, (folder, _) in rebuild.TABLES.items():
            path = directory / 'results' / folder / rebuild.TABLE_FILES[number]
            path.parent.mkdir()
            shutil.copyfile(ROOT / 'results' / folder / path.name, path)
            paths.append(path)
        return paths

    def verify_archive(self, directory):
        output = io.StringIO()
        with patch.object(rebuild, 'ROOT', directory), \
             patch.object(sys, 'argv', ['rebuild_results.py']), \
             contextlib.redirect_stdout(output):
            rebuild.main()
        return json.loads(output.getvalue())

    def test_archive_verification_preserves_csvs_without_tex_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            paths = self.archived_results(archive)
            saved = {p: p.read_bytes() for p in paths}
            report = self.verify_archive(archive)
            self.assertEqual(report['benchmark_instance_rows'], 435)
            self.assertEqual(report['scale_results'], 15)
            self.assertEqual(report['table_rows_matching_archived_csv'], 45)
            self.assertTrue(all(t['matches_archived_csv'] for t in report['tables']))
            self.assertEqual({p: p.read_bytes() for p in paths}, saved)
            self.assertEqual(set((archive / 'results').rglob('*.csv')), set(paths))

    def test_archive_mismatch_fails_without_overwriting_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            paths = self.archived_results(archive)
            reference = archive / 'results/Table4_Preprocessing' / rebuild.TABLE_FILES[4]
            rows = rebuild.read_csv(reference)
            rows[0]['preprocessing_time_s'] = '999999.000'
            rebuild.write_csv(reference, rows, list(rows[0]))
            saved = {p: p.read_bytes() for p in paths}
            with self.assertRaisesRegex(AssertionError, 'Table 4, row 1, preprocessing_time_s'):
                self.verify_archive(archive)
            self.assertEqual({p: p.read_bytes() for p in paths}, saved)
            self.assertEqual(set((archive / 'results').rglob('*.csv')), set(paths))

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
