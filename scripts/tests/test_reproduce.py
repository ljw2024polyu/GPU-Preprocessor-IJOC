# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

"""CPU checks for portable builds, license gating, and reproduction selection."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which('cmake')
CTEST = shutil.which('ctest')
TOOLCHAIN = ('CC', 'CXX', 'FC', 'CUDACXX', 'CUDAHOSTCXX', 'CUDAToolkit_ROOT',
             'CUDAARCHS', 'CMAKE_BUILD_PARALLEL_LEVEL')
CONFIGURATION = TOOLCHAIN + ('CUDA_ARCH', 'BUILD_JOBS', 'BUILD_DIR', 'CMAKE_ARGS',
                            'WORK_ROOT', 'RUN_DIR', 'INPUT_ROOT', 'WITH_GUROBI',
                            'GUROBI_HOME', 'JULIA', 'FFLAGS', 'MAKEFLAGS', 'MFLAGS',
                            'PYTHON', 'TEST_GUROBI_PROBE_EXIT')


class ReproduceTests(unittest.TestCase):
    def run_stage(self, stage='setup', overrides=None, make=False, extra_args=()):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            project = directory / 'project with spaces'
            (project / 'scripts').mkdir(parents=True)
            (project / 'build-experiments').mkdir()
            shutil.copyfile(ROOT / 'scripts/reproduce.sh', project / 'scripts/reproduce.sh')
            shutil.copyfile(ROOT / 'Makefile', project / 'Makefile')
            tools = directory / 'tools with spaces'
            tools.mkdir()
            log = directory / 'calls.jsonl'
            fake_tool = (
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                f'keys = {TOOLCHAIN!r}\n'
                'with open(os.environ["TEST_CALL_LOG"], "a") as stream:\n'
                '    stream.write(json.dumps({"tool": os.path.basename(sys.argv[0]),\n'
                '        "args": sys.argv[1:],\n'
                '        "environment": {k: os.environ[k] for k in keys if k in os.environ}}) + "\\n")\n'
                'if len(sys.argv) > 1 and sys.argv[1] == "scripts/check_gurobi.py":\n'
                '    sys.exit(int(os.environ.get("TEST_GUROBI_PROBE_EXIT", "0")))\n'
            )
            for name in ['python3', 'custom-python', 'cmake', 'ctest']:
                executable = tools / name
                executable.write_text(fake_tool)
                executable.chmod(0o755)
            environment = {k: v for k, v in os.environ.items() if k not in CONFIGURATION}
            environment.update(PATH=str(tools) + os.pathsep + os.environ['PATH'],
                               CMAKE=str(tools / 'cmake'),
                               CTEST=str(tools / 'ctest'),
                               TEST_CALL_LOG=str(log))
            environment.update(overrides or {})
            command = ['make'] if make else ['bash', 'scripts/reproduce.sh', stage, *extra_args]
            subprocess.run(command, cwd=project, env=environment,
                           text=True, capture_output=True, check=True)
            return [json.loads(line) for line in log.read_text().splitlines()]

    def test_default_builds_do_not_force_toolchain_target_or_parallelism(self):
        for make in [False, True]:
            with self.subTest(make=make):
                calls = self.run_stage(make=make)
                self.assertEqual([c['tool'] for c in calls], ['cmake', 'cmake'])
                self.assertEqual(len(calls), 2)
                configure, build = calls
                self.assertIn('-S', configure['args'])
                if not make:
                    self.assertIn('-DHPRLP_GUROBI=OFF', configure['args'])
                self.assertFalse(any('CMAKE_CUDA_ARCHITECTURES' in a for a in configure['args']))
                self.assertEqual(configure['environment'], {})
                self.assertIn('--build', build['args'])
                self.assertNotIn('--parallel', build['args'])
                self.assertNotIn('-j', build['args'])

    def test_explicit_toolchain_target_and_parallelism_are_preserved(self):
        settings = dict(CC='custom-cc', CXX='custom-cxx', FC='custom-fc',
                        CUDACXX='/custom toolkit/bin/nvcc', CUDAHOSTCXX='custom-host-cxx',
                        CUDAToolkit_ROOT='/custom toolkit', CUDA_ARCH='75;86', BUILD_JOBS='2')
        for make in [False, True]:
            with self.subTest(make=make):
                calls = [c for c in self.run_stage(overrides=settings, make=make)
                         if c['tool'] == 'cmake']
                configure, build = calls
                self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=75;86', configure['args'])
                self.assertEqual(build['args'][build['args'].index('--parallel') + 1], '2')
                for call in calls:
                    self.assertEqual(call['environment'], {k: settings[k] for k in TOOLCHAIN if k in settings})

    def test_fresh_setup_preserves_the_selected_toolchain(self):
        settings = dict(CUDACXX='/selected toolkit/bin/nvcc', CUDAHOSTCXX='selected-cxx')
        configure, build = self.run_stage(overrides=settings, extra_args=['--fresh'])
        self.assertIn('--fresh', configure['args'])
        self.assertNotIn('--fresh', build['args'])
        self.assertEqual(configure['environment'], settings)

    @unittest.skipUnless(CTEST, 'CTest is needed to check missing test registration')
    def test_no_cpp_tests_is_an_error(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.run_stage(stage='test', overrides={'CTEST': CTEST})
        self.assertIn('No tests were found', failure.exception.stdout + failure.exception.stderr)

    def test_standard_cmake_environment_is_not_overridden(self):
        settings = dict(CUDAARCHS='75;86', CMAKE_BUILD_PARALLEL_LEVEL='2')
        for make in [False, True]:
            with self.subTest(make=make):
                calls = [c for c in self.run_stage(overrides=settings, make=make)
                         if c['tool'] == 'cmake']
                self.assertFalse(any('CMAKE_CUDA_ARCHITECTURES' in a for a in calls[0]['args']))
                self.assertNotIn('--parallel', calls[1]['args'])
                for call in calls:
                    self.assertEqual(call['environment'], settings)

    def test_data_preparation_uses_generic_compilers_and_accepts_overrides(self):
        for settings in [{}, dict(CC='custom-cc', CXX='custom-cxx', FC='custom-fc')]:
            with self.subTest(settings=settings):
                calls = self.run_stage(stage='data', overrides=settings)
                prepare = [c for c in calls if c['args'][0] == 'scripts/prepare_inputs.py']
                self.assertEqual(len(prepare), 4)
                for call in prepare:
                    for flag, variable, default in [('--cc', 'CC', 'cc'),
                                                     ('--cxx', 'CXX', 'c++'),
                                                     ('--fc', 'FC', 'gfortran')]:
                        self.assertEqual(call['args'][call['args'].index(flag) + 1],
                                         settings.get(variable, default))

    def test_data_preparation_uses_the_current_or_selected_python(self):
        for settings, expected in [({}, 'python3'),
                                   ({'PYTHON': 'custom-python'}, 'custom-python')]:
            with self.subTest(settings=settings):
                calls = self.run_stage(stage='data', overrides=settings)
                self.assertEqual([c['tool'] for c in calls], [expected] * 4)
                self.assertTrue(all(c['args'][0] == 'scripts/prepare_inputs.py' for c in calls))

    def test_resume_uses_the_run_entry_with_resume_enabled(self):
        call, = self.run_stage(stage='resume')
        self.assertEqual(call['args'][0], 'scripts/run_experiments.py')
        self.assertIn('--resume', call['args'])
        methods = call['args'].index('--methods')
        self.assertEqual(call['args'][methods + 1:methods + 5], ['none', 'folding', 'gpu', 'pslp'])

    def test_auto_gurobi_build_requires_a_successful_sdk_and_license_probe(self):
        for status, enabled in [('0', 'ON'), ('1', 'OFF')]:
            with self.subTest(status=status):
                calls = self.run_stage(overrides={'GUROBI_HOME': '/SDK with spaces',
                                                 'TEST_GUROBI_PROBE_EXIT': status})
                probe, configure, build = calls
                self.assertEqual(probe['args'], ['scripts/check_gurobi.py'])
                self.assertIn('-DHPRLP_GUROBI=' + enabled, configure['args'])
                self.assertEqual('-DGUROBI_ROOT=/SDK with spaces' in configure['args'], enabled == 'ON')
                self.assertIn('--build', build['args'])

    def test_auto_run_includes_gurobi_only_when_probe_and_build_are_usable(self):
        for status in ['0', '1']:
            with self.subTest(status=status):
                probe, run = self.run_stage(stage='run', overrides={'GUROBI_HOME': '/SDK',
                                                 'TEST_GUROBI_PROBE_EXIT': status})
                self.assertEqual(probe['args'][0], 'scripts/check_gurobi.py')
                self.assertIn('--build-dir', probe['args'])
                selected = run['args'][run['args'].index('--methods') + 1:run['args'].index('--output-dir')]
                self.assertEqual(selected, ['none', 'folding', 'gpu', 'pslp'] + (['gurobi'] if status == '0' else []))

    def test_gurobi_off_skips_the_probe_even_with_an_sdk(self):
        configure, build = self.run_stage(overrides={'GUROBI_HOME': '/SDK', 'WITH_GUROBI': 'OFF'})
        self.assertIn('-DHPRLP_GUROBI=OFF', configure['args'])
        self.assertIn('--build', build['args'])

    def test_explicit_gurobi_requirement_fails_before_build_or_experiments(self):
        for stage in ['setup', 'run']:
            for settings in [{'WITH_GUROBI': 'ON'},
                             {'WITH_GUROBI': 'ON', 'GUROBI_HOME': '/SDK', 'TEST_GUROBI_PROBE_EXIT': '1'}]:
                with self.subTest(stage=stage, settings=settings), self.assertRaises(subprocess.CalledProcessError) as failure:
                    self.run_stage(stage=stage, overrides=settings)
                self.assertIn('WITH_GUROBI=ON requires', failure.exception.stderr)

    def test_data_preparation_does_not_check_the_benchmark_license(self):
        calls = self.run_stage(stage='data', overrides={'GUROBI_HOME': '/SDK', 'TEST_GUROBI_PROBE_EXIT': '1'})
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c['args'][0] == 'scripts/prepare_inputs.py' for c in calls))

    def test_test_stage_skips_only_gurobi_checks_when_license_is_unavailable(self):
        for status in ['0', '1']:
            with self.subTest(status=status):
                probe, python_tests, cpp_tests = self.run_stage(stage='test',
                    overrides={'GUROBI_HOME': '/SDK', 'TEST_GUROBI_PROBE_EXIT': status})
                self.assertIn('--build-dir', probe['args'])
                self.assertEqual(python_tests['args'][:3], ['-m', 'unittest', 'discover'])
                self.assertEqual(cpp_tests['tool'], 'ctest')
                self.assertIn('--no-tests=error', cpp_tests['args'])
                self.assertEqual('-E' in cpp_tests['args'], status == '1')
                if status == '1':
                    self.assertEqual(cpp_tests['args'][-2:], ['-E', 'gurobi'])

    @unittest.skipUnless(CMAKE, 'CMake is needed to check architecture initialization')
    def test_cmake_architecture_default_and_overrides(self):
        prefix = (ROOT / 'CMakeLists.txt').read_text().split('project(', 1)[0]
        probe = prefix + '\nif(DEFINED CMAKE_CUDA_ARCHITECTURES)\n' \
            '  message(STATUS "target=${CMAKE_CUDA_ARCHITECTURES}")\n' \
            'else()\n  message(STATUS "environment=$ENV{CUDAARCHS}")\nendif()\n'
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / 'architecture.cmake'
            script.write_text(probe)
            cases = [({}, [], 'target=native'),
                     ({'CUDAARCHS': '75;86'}, [], 'environment=75;86'),
                     ({'CUDAARCHS': '75;86'}, ['-DCMAKE_CUDA_ARCHITECTURES=89'], 'target=89')]
            for settings, arguments, expected in cases:
                with self.subTest(settings=settings, arguments=arguments):
                    environment = {k: v for k, v in os.environ.items() if k != 'CUDAARCHS'}
                    environment.update(settings)
                    completed = subprocess.run([CMAKE, *arguments, '-P', str(script)],
                                               env=environment, text=True,
                                               capture_output=True, check=True)
                    self.assertIn(expected, completed.stdout + completed.stderr)


if __name__ == '__main__':
    unittest.main()
