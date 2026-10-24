# Build rules live outside the byte-identical official v0.1.3 source tree.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(GPUPREPROCESSOR_USE_INT64_NNZ)
  message(FATAL_ERROR "The HPR-LP-C 0.1.3 interface requires int32 sparse offsets")
endif()
find_package(ZLIB REQUIRED)
option(HPRLP_HDF5 "Enable optional HDF5 input support" OFF)
if(HPRLP_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()
option(HPRLP_GUROBI "Enable the separately licensed Gurobi 12.0.2 adapter" OFF)
if(HPRLP_GUROBI)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
  find_package(Gurobi REQUIRED)
endif()
set(HPRLP_SOURCE "${PROJECT_SOURCE_DIR}/third_party/hprlp")
set(PSLP_SOURCE "${PROJECT_SOURCE_DIR}/third_party/pslp")
function(build_pslp target source version)
  file(GLOB sources CONFIGURE_DEPENDS "${source}/src/core/*.c" "${source}/src/explorers/*.c")
  list(REMOVE_ITEM sources "${source}/src/core/Debugger.c")
  add_library(${target} STATIC ${sources})
  target_include_directories(${target} PRIVATE "${source}/include/PSLP" "${source}/include/core"
    "${source}/include/data_structures" "${source}/include/explorers")
  target_compile_definitions(${target} PRIVATE PSLP_VERSION="${version}" NDEBUG _POSIX_C_SOURCE=200809L)
  target_compile_options(${target} PRIVATE -O3)
  set_target_properties(${target} PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
endfunction()
# Preserve the release's own dependency; the external comparison uses 0.0.11.
build_pslp(ijoc_release_pslp "${HPRLP_SOURCE}/third_party/PSLP" "0.0.8")
build_pslp(ijoc_pslp "${PSLP_SOURCE}" "0.0.11")
target_compile_options(ijoc_pslp PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/adapters/pslp0011_symbols.h")

set(HPRLP_SOURCES
  src/presolve/pslp_integration.cpp
  src/io/mps_reader.cpp
  src/gpu/memory/compressible_memory.cu
  src/support/cuda_utils.cu
  src/solver/scaling.cu
  src/gpu/preprocessing/preprocess.cu
  src/solver/power_iteration.cu
  src/solver/restart_control.cpp
  src/solver/progress_monitor.cu
  src/solver/reduced_matrix.cu
  src/solver/iteration/main_iterate.cu
  src/solver/solver_output.cpp
  src/solver/cuda_graph.cu
  src/solver/solve.cu
  src/api/model.cpp
  src/batch/batched_solver.cu
  src/cuda_kernels/shared/vector_kernels.cu
  src/cuda_kernels/shared/scaling_kernels.cu
  src/cuda_kernels/shared/residual_kernels.cu
  src/cuda_kernels/shared/halpern_kernels.cu
  src/cuda_kernels/backends/simple/simple_update_kernels.cu
  src/cuda_kernels/backends/generic/generic_fused_kernels.cu
  src/cuda_kernels/backends/unit/unit_kernels.cu
  src/cuda_kernels/backends/unit/signed_unit_kernels.cu
  src/cuda_kernels/backends/dictionary/dictionary_kernels.cu
  src/cuda_kernels/backends/structured/structured_kernels.cu
  src/cuda_kernels/backends/structured/factorized_stencil_kernels.cu
  src/cuda_kernels/backends/structured/grid_slack_laplacian_kernels.cu)
list(TRANSFORM HPRLP_SOURCES PREPEND "${HPRLP_SOURCE}/")
add_library(ijoc_hprlp_release STATIC ${HPRLP_SOURCES})
target_compile_features(ijoc_hprlp_release PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(ijoc_hprlp_release PUBLIC "${HPRLP_SOURCE}/include"
  PRIVATE "${HPRLP_SOURCE}/include/cuda_kernels" "${HPRLP_SOURCE}/third_party/PSLP/include/PSLP")
target_compile_definitions(ijoc_hprlp_release PRIVATE CUSPARSE_ENABLE_EXPERIMENTAL_API)
# Parse CUDA's header before the release's numThreads macro; no source patch.
target_compile_options(ijoc_hprlp_release PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--pre-include=cooperative_groups.h>")
target_link_libraries(ijoc_hprlp_release PUBLIC ijoc_release_pslp CUDA::cublas CUDA::cusolver
  CUDA::cusparse CUDA::curand CUDA::cuda_driver CUDA::cudart ZLIB::ZLIB Threads::Threads)
set_target_properties(ijoc_hprlp_release PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(ijoc_hprlp_pipeline STATIC integration.cpp adapters/pslp.cpp adapters/gpu.cpp adapters/gurobi.cpp)
target_include_directories(ijoc_hprlp_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE "${PSLP_SOURCE}/include/PSLP")
target_compile_definitions(ijoc_hprlp_pipeline PRIVATE HPRLP_HAS_GUROBI=$<BOOL:${HPRLP_GUROBI}>)
target_link_libraries(ijoc_hprlp_pipeline PUBLIC ijoc_hprlp_release GPUPreprocessor::core ijoc_pslp)
if(HPRLP_GUROBI)
  target_link_libraries(ijoc_hprlp_pipeline PRIVATE Gurobi::C)
endif()
add_executable(hprlp_solve solve_mps_file.cpp)
target_link_libraries(hprlp_solve PRIVATE ijoc_hprlp_pipeline)
set_target_properties(hprlp_solve PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
if(HPRLP_HDF5)
  target_compile_definitions(hprlp_solve PRIVATE HPRLP_HAS_HDF5=1)
  target_include_directories(hprlp_solve PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(hprlp_solve PRIVATE ${HDF5_LIBRARIES})
endif()
if(BUILD_TESTING)
  # Unmodified official CLI for comparison against the external no-presolve path.
  add_executable(hprlp_release_reference "${HPRLP_SOURCE}/src/cli/solve_mps_file.cpp")
  target_link_libraries(hprlp_release_reference PRIVATE ijoc_hprlp_release)
  foreach(test IN ITEMS pslp0011_integration_test gpu_presolver_integration_test)
    add_executable(hprlp_${test} "tests/${test}.cpp")
    target_link_libraries(hprlp_${test} PRIVATE ijoc_hprlp_pipeline)
    target_compile_options(hprlp_${test} PRIVATE -UNDEBUG)
    add_test(NAME hprlp_${test} COMMAND hprlp_${test})
    set_tests_properties(hprlp_${test} PROPERTIES TIMEOUT 120)
  endforeach()
  add_executable(hprlp_restart_control_test "${HPRLP_SOURCE}/tests/restart_control_test.cpp")
  target_link_libraries(hprlp_restart_control_test PRIVATE ijoc_hprlp_release)
  target_compile_options(hprlp_restart_control_test PRIVATE -UNDEBUG)
  add_test(NAME hprlp_restart_control_test COMMAND hprlp_restart_control_test)
  add_test(NAME hprlp_pslp_cpu_integration_test COMMAND hprlp_pslp0011_integration_test --presolve-only)
  set_tests_properties(hprlp_pslp_cpu_integration_test hprlp_restart_control_test PROPERTIES LABELS cpu TIMEOUT 120)
  if(HPRLP_GUROBI)
    add_executable(hprlp_gurobi_presolver_integration_test tests/gurobi_presolver_integration_test.cpp)
    target_link_libraries(hprlp_gurobi_presolver_integration_test PRIVATE ijoc_hprlp_pipeline)
    target_compile_options(hprlp_gurobi_presolver_integration_test PRIVATE -UNDEBUG)
    add_test(NAME hprlp_gurobi_presolver_integration_test COMMAND hprlp_gurobi_presolver_integration_test)
    set_tests_properties(hprlp_gurobi_presolver_integration_test PROPERTIES TIMEOUT 120)
  endif()
endif()
message(STATUS "HPR-LP official release v0.1.3 (89d3cc0); external PSLP 0.0.11; Gurobi ${HPRLP_GUROBI}")
