"""CPU checks for portable build defaults and user toolchain overrides."""
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
                            'PYTHON')


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
            )
            for name in ['python3', 'custom-python', 'cmake']:
                executable = tools / name
                executable.write_text(fake_tool)
                executable.chmod(0o755)
            environment = {k: v for k, v in os.environ.items() if k not in CONFIGURATION}
            environment.update(PATH=str(tools) + os.pathsep + os.environ['PATH'],
                               CMAKE=str(tools / 'cmake'),
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
                self.assertEqual([c['tool'] for c in calls], [expected] * 5)
                self.assertEqual(calls[-1]['args'][0], 'scripts/check_inputs.py')

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
