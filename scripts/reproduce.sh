#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Run from any directory. See docs/REPRODUCING.md for prerequisites and outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
entry_script="$PWD/scripts/reproduce.sh"

stage=${1:-help}
WORK_ROOT=${WORK_ROOT:-"$PWD/runs/reproduction"}
INPUT_ROOT=${INPUT_ROOT:-"$WORK_ROOT/inputs"}
RUN_DIR=${RUN_DIR:-"$WORK_ROOT/experiment"}
PYTHON=${PYTHON:-"$PWD/.venv/bin/python"}
BUILD_DIR=${BUILD_DIR:-"$PWD/build-experiments"}
SOLVER="$BUILD_DIR/bin/hprlp_solve"
WITH_GUROBI=${WITH_GUROBI:-OFF}
methods=(none folding gpu pslp)
if [[ "$WITH_GUROBI" == ON ]]; then methods+=(gurobi); fi

prepare() {
  "$PYTHON" scripts/prepare_inputs.py "$1" --output-dir "$INPUT_ROOT" --resume \
    --julia "${JULIA:-julia}" --cc "${CC:-cc}" --cxx "${CXX:-g++}" \
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
    python3 -m venv .venv
    "$PYTHON" -m pip install -r requirements.txt
    cmake_args=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-80}"
      -DGPUPREPROCESSOR_BUILD_HPRLP=ON -DGPUPREPROCESSOR_BUILD_TESTS=ON
      "-DHPRLP_GUROBI=$WITH_GUROBI")
    if [[ "$WITH_GUROBI" == ON ]]; then
      cmake_args+=("-DGUROBI_ROOT=${GUROBI_HOME:?Set GUROBI_HOME to the Gurobi 12.0.2 SDK}")
    fi
    "${CMAKE:-cmake}" -S . -B "$BUILD_DIR" "${cmake_args[@]}"
    "${CMAKE:-cmake}" --build "$BUILD_DIR" -j "${BUILD_JOBS:-4}"
    ;;
  data)
    for group in hans miplib-large qap20 qap-scale; do prepare "$group"; done
    "$PYTHON" scripts/check_inputs.py --input-root "$INPUT_ROOT" --solver "$SOLVER"
    ;;
  test)
    "$PYTHON" -m unittest discover -s scripts/tests -v
    "${CTEST:-ctest}" --test-dir "$BUILD_DIR" --output-on-failure
    ;;
  smoke)
    prepare hans --instances FOME13
    prepare miplib-large --instances scpn2_relax
    prepare qap20 --instances tho40
    prepare qap-scale --instances qap-wil100-n020
    "$PYTHON" scripts/check_inputs.py --input-root "$INPUT_ROOT" --solver "$SOLVER" --available-only
    "$PYTHON" scripts/run_experiments.py --input-root "$INPUT_ROOT" --solver "$SOLVER" \
      --suite all --methods "${methods[@]}" --instances FOME13 scpn2_relax tho40 \
      --sizes 20 --output-dir "$RUN_DIR"
    summarize
    ;;
  run)
    "$PYTHON" scripts/run_experiments.py --input-root "$INPUT_ROOT" --solver "$SOLVER" \
      --suite all --methods "${methods[@]}" --output-dir "$RUN_DIR"
    ;;
  results)
    summarize
    ;;
  all)
    for step in setup data test run results; do bash "$entry_script" "$step"; done
    ;;
  help|-h|--help)
    printf '%s\n' 'Usage: bash scripts/reproduce.sh STAGE' \
      '  setup    Install pinned Python packages and build the solver/tests (CPU).' \
      '  data     Download/generate and verify all 92 paper input models (CPU).' \
      '  test     Run Python and C++/CUDA regression tests (requires one GPU).' \
      '  smoke    Prepare one model per group, run all selected methods, and summarize.' \
      '  run      Run the complete Section 4 plan in a new RUN_DIR (one GPU).' \
      '  results  Rebuild tables and Figure 2 from that run (CPU).' \
      '  all      Execute setup, data, test, run, and results in order.' \
      'Set CUDA_VISIBLE_DEVICES before GPU stages. The solver uses logical device 0.' \
      'Set WORK_ROOT/RUN_DIR for outputs; see docs/REPRODUCING.md.'
    ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
