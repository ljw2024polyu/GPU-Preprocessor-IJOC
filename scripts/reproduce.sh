#!/usr/bin/env bash
# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

# Run from any directory. See docs/REPRODUCING.md for prerequisites and outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
entry_script="$PWD/scripts/reproduce.sh"

stage=${1:-help}
WORK_ROOT=${WORK_ROOT:-"$PWD/runs/reproduction"}
INPUT_ROOT=${INPUT_ROOT:-"$WORK_ROOT/inputs"}
RUN_DIR=${RUN_DIR:-"$WORK_ROOT/experiment"}
PYTHON=${PYTHON:-python3}
BUILD_DIR=${BUILD_DIR:-"$PWD/build-experiments"}
SOLVER="$BUILD_DIR/bin/hprlp_solve"
WITH_GUROBI=${WITH_GUROBI:-AUTO}
GUROBI_ENABLED=OFF
methods=(none folding gpu pslp)

select_gurobi() {
  GUROBI_ENABLED=OFF
  case "$WITH_GUROBI" in
    OFF) return ;;
    AUTO|ON) ;;
    *) printf '%s\n' 'WITH_GUROBI must be AUTO, ON, or OFF.' >&2; exit 2 ;;
  esac
  local probe_args=(scripts/check_gurobi.py)
  if [[ "${1:-}" == built ]]; then probe_args+=(--build-dir "$BUILD_DIR"); fi
  if [[ -n "${GUROBI_HOME:-}" ]] && "$PYTHON" "${probe_args[@]}"; then
    GUROBI_ENABLED=ON
    methods+=(gurobi)
  else
    if [[ -z "${GUROBI_HOME:-}" ]]; then
      printf '%s\n' 'Gurobi skipped: set GUROBI_HOME to the Gurobi 12.0.2 C SDK and configure a full license.' >&2
    fi
    if [[ "$WITH_GUROBI" == ON ]]; then
      printf '%s\n' 'WITH_GUROBI=ON requires a usable SDK, license, and adapter build; no experiments were started.' >&2
      exit 1
    fi
  fi
}

prepare() {
  "$PYTHON" scripts/prepare_inputs.py "$1" --output-dir "$INPUT_ROOT" --resume \
    --julia "${JULIA:-julia}" --cc "${CC:-cc}" --cxx "${CXX:-c++}" \
    --fc "${FC:-gfortran}" --fflags="${FFLAGS:--O3 -std=legacy -mcmodel=large}" "${@:2}"
}

summarize() {
  "$PYTHON" scripts/rebuild_results.py --run-dir "$RUN_DIR"
  local figure_data="$RUN_DIR/results/Figure2_QAPScalability/figure_2_preprocessing_times.csv"
  if [[ -f "$figure_data" ]]; then
    "$PYTHON" scripts/plot_figure2.py --data "$figure_data"
  fi
}

case "$stage" in
  setup)
    if (( $# > 2 )) || [[ -n "${2:-}" && "${2:-}" != --fresh ]]; then
      printf '%s\n' 'Usage: bash scripts/reproduce.sh setup [--fresh]' >&2
      exit 2
    fi
    select_gurobi
    cmake_args=(-DCMAKE_BUILD_TYPE=Release
      -DGPUPREPROCESSOR_BUILD_HPRLP=ON -DGPUPREPROCESSOR_BUILD_TESTS=ON
      "-DHPRLP_GUROBI=$GUROBI_ENABLED")
    if [[ "${2:-}" == --fresh ]]; then
      cmake_args+=(--fresh)
    fi
    if [[ -n "${CUDA_ARCH:-}" ]]; then
      cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
    fi
    if [[ "$GUROBI_ENABLED" == ON ]]; then
      cmake_args+=("-DGUROBI_ROOT=${GUROBI_HOME:?Set GUROBI_HOME to the Gurobi 12.0.2 SDK}")
    fi
    "${CMAKE:-cmake}" -S . -B "$BUILD_DIR" "${cmake_args[@]}"
    build_args=(--build "$BUILD_DIR")
    if [[ -n "${BUILD_JOBS:-}" ]]; then
      build_args+=(--parallel "$BUILD_JOBS")
    fi
    "${CMAKE:-cmake}" "${build_args[@]}"
    ;;
  data)
    for group in hans miplib-large qap20 qap-scale; do prepare "$group"; done
    ;;
  test)
    select_gurobi built
    "$PYTHON" -m unittest discover -s scripts/tests -v
    test_args=(--test-dir "$BUILD_DIR" --output-on-failure --no-tests=error)
    if [[ "$GUROBI_ENABLED" == OFF ]]; then test_args+=(-E gurobi); fi
    "${CTEST:-ctest}" "${test_args[@]}"
    ;;
  smoke)
    select_gurobi built
    prepare hans --instances FOME13
    prepare miplib-large --instances scpn2_relax
    prepare qap20 --instances tho40
    prepare qap-scale --instances qap-wil100-n020
    "$PYTHON" scripts/run_experiments.py --input-root "$INPUT_ROOT" --solver "$SOLVER" \
      --suite all --methods "${methods[@]}" --instances FOME13 scpn2_relax tho40 \
      --sizes 20 --output-dir "$RUN_DIR"
    summarize
    ;;
  run|resume)
    select_gurobi built
    run_args=(scripts/run_experiments.py --input-root "$INPUT_ROOT" --solver "$SOLVER"
      --suite all --methods "${methods[@]}" --output-dir "$RUN_DIR")
    if [[ "$stage" == resume ]]; then run_args+=(--resume); fi
    "$PYTHON" "${run_args[@]}"
    ;;
  results)
    summarize
    ;;
  all)
    for step in setup data test run results; do bash "$entry_script" "$step"; done
    ;;
  help|-h|--help)
    printf '%s\n' 'Usage: bash scripts/reproduce.sh STAGE' \
      '  setup [--fresh]  Build the solver/tools/tests; --fresh resets CMake configuration.' \
      '  data     Download/generate all 92 paper input models (CPU).' \
      '  test     Run Python and C++/CUDA regression tests (requires one GPU).' \
      '  smoke    Prepare one model per group, run all selected methods, and summarize.' \
      '  run      Run the complete Section 4 plan in a new RUN_DIR (one GPU).' \
      '  resume   Continue RUN_DIR, reusing completed logs from its saved command plan.' \
      '  results  Rebuild tables and Figure 2 from that run (CPU).' \
      '  all      Execute setup, data, test, run, and results in order.' \
      'CMake discovers the toolchain and GPU architecture; set CUDA_ARCH for a target override.' \
      'Set BUILD_JOBS to override CMake build parallelism.' \
      'The solver uses logical device 0 by default.' \
      'Gurobi defaults to AUTO: include it only with a usable 12.0.2 C SDK and full license.' \
      'Set GUROBI_HOME and, if needed, GRB_LICENSE_FILE. WITH_GUROBI=OFF skips it; ON requires it.' \
      'Set WORK_ROOT/RUN_DIR for outputs; see docs/REPRODUCING.md.'
    ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
