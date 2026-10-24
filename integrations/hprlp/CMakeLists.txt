# HPR-LP-C 0.1.3 experiment implementation; see docs/SOLVER_INTEGRATION.md.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(GPUPREPROCESSOR_USE_INT64_NNZ)
  message(FATAL_ERROR "The HPR-LP-C 0.1.3 interface requires int32 sparse offsets")
endif()
find_package(ZLIB REQUIRED)
option(HPRLP_HDF5 "Enable optional HDF5 input support (paper experiments use MPS)" OFF)
if(HPRLP_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()
set(HPRLP_GUROBI "OFF" CACHE STRING "Optional Gurobi 12.0.2 presolve backend: ON or OFF")
set_property(CACHE HPRLP_GUROBI PROPERTY STRINGS ON OFF)
string(TOUPPER "${HPRLP_GUROBI}" HPRLP_GUROBI)
if(NOT HPRLP_GUROBI MATCHES "^(ON|OFF)$")
  message(FATAL_ERROR "HPRLP_GUROBI must be ON or OFF")
endif()
if(HPRLP_GUROBI STREQUAL "ON")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
  find_package(Gurobi REQUIRED)
endif()

set(HPRLP_SOURCE "${PROJECT_SOURCE_DIR}/third_party/hprlp")
set(PSLP_SOURCE "${PROJECT_SOURCE_DIR}/third_party/pslp")
file(GLOB PSLP_SOURCES CONFIGURE_DEPENDS
  "${PSLP_SOURCE}/src/core/*.c" "${PSLP_SOURCE}/src/explorers/*.c")
list(REMOVE_ITEM PSLP_SOURCES "${PSLP_SOURCE}/src/core/Debugger.c")
add_library(ijoc_pslp STATIC ${PSLP_SOURCES})
target_include_directories(ijoc_pslp PUBLIC "${PSLP_SOURCE}/include/PSLP"
  PRIVATE "${PSLP_SOURCE}/include/core" "${PSLP_SOURCE}/include/data_structures"
          "${PSLP_SOURCE}/include/explorers")
target_compile_definitions(ijoc_pslp PRIVATE PSLP_VERSION="0.0.11" NDEBUG _POSIX_C_SOURCE=200809L)
target_compile_options(ijoc_pslp PRIVATE -O3)
set_target_properties(ijoc_pslp PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)

set(HPRLP_SOURCES
  src/presolve/pslp_integration.cpp
  src/presolve/pslp0011_integration.cpp
  src/presolve/gurobi_presolver_integration.cpp
  src/presolve/gpu_presolver_integration.cpp
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
add_library(ijoc_hprlp STATIC ${HPRLP_SOURCES})
target_compile_features(ijoc_hprlp PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(ijoc_hprlp PUBLIC "${HPRLP_SOURCE}/include"
  PRIVATE "${HPRLP_SOURCE}/include/cuda_kernels")
target_compile_definitions(ijoc_hprlp PRIVATE
  HPRLP_HAS_GPU_PRESOLVER=1 HPRLP_HAS_GUROBI=$<BOOL:${HPRLP_GUROBI}>
  CUSPARSE_ENABLE_EXPERIMENTAL_API)
target_link_libraries(ijoc_hprlp PUBLIC GPUPreprocessor::core ijoc_pslp
  CUDA::cublas CUDA::cusolver CUDA::cusparse CUDA::curand CUDA::cuda_driver
  CUDA::cudart ZLIB::ZLIB Threads::Threads)
set_target_properties(ijoc_hprlp PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(HPRLP_GUROBI STREQUAL "ON")
  target_link_libraries(ijoc_hprlp PRIVATE Gurobi::C)
endif()

add_executable(hprlp_solve "${HPRLP_SOURCE}/src/cli/solve_mps_file.cpp")
target_link_libraries(hprlp_solve PRIVATE ijoc_hprlp)
set_target_properties(hprlp_solve PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
if(HPRLP_HDF5)
  target_compile_definitions(hprlp_solve PRIVATE HPRLP_HAS_HDF5=1)
  target_include_directories(hprlp_solve PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(hprlp_solve PRIVATE ${HDF5_LIBRARIES})
endif()

if(BUILD_TESTING)
  foreach(test IN ITEMS pslp0011_integration_test gpu_presolver_integration_test restart_control_test)
    add_executable(hprlp_${test} "${HPRLP_SOURCE}/tests/${test}.cpp")
    target_link_libraries(hprlp_${test} PRIVATE ijoc_hprlp)
    target_compile_options(hprlp_${test} PRIVATE -UNDEBUG)
    add_test(NAME hprlp_${test} COMMAND hprlp_${test})
    set_tests_properties(hprlp_${test} PROPERTIES TIMEOUT 120)
  endforeach()
  add_test(NAME hprlp_pslp_cpu_integration_test
    COMMAND hprlp_pslp0011_integration_test --presolve-only)
  set_tests_properties(hprlp_pslp_cpu_integration_test hprlp_restart_control_test
    PROPERTIES LABELS cpu TIMEOUT 120)
  if(HPRLP_GUROBI STREQUAL "ON")
    add_executable(hprlp_gurobi_presolver_integration_test
      "${HPRLP_SOURCE}/tests/gurobi_presolver_integration_test.cpp")
    target_link_libraries(hprlp_gurobi_presolver_integration_test PRIVATE ijoc_hprlp)
    target_compile_options(hprlp_gurobi_presolver_integration_test PRIVATE -UNDEBUG)
    add_test(NAME hprlp_gurobi_presolver_integration_test COMMAND hprlp_gurobi_presolver_integration_test)
    set_tests_properties(hprlp_gurobi_presolver_integration_test PROPERTIES TIMEOUT 120)
  endif()
endif()
message(STATUS "Experiment solver: HPR-LP-C 0.1.3 / PSLP 0.0.11 / Gurobi ${HPRLP_GUROBI}")
