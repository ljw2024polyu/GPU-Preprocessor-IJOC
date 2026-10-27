#ifndef IJOC_GPU_ADAPTER_HPP
#define IJOC_GPU_ADAPTER_HPP

#include "integration.hpp"

#include <chrono>

bool ijoc_run_gpu_presolve(const LP_info_cpu *model,
                               const IntegrationParameters *param,
                               LP_info_cpu *reduced_model,
                               void **presolver_handle_out,
                               HPRLP_FLOAT *presolve_time_out,
                               HPRLP_FLOAT *folding_time_out,
                               HPRLP_FLOAT *presolve_core_time_out = nullptr,
                               HPRLP_FLOAT *preprocessing_time_out = nullptr);

bool ijoc_run_gpu_presolve_device(const LP_info_cpu *model,
                                      const IntegrationParameters *param,
                                      LP_info_gpu *reduced_model,
                                      void **presolver_handle_out,
                                      HPRLP_FLOAT *presolve_time_out,
                                      HPRLP_FLOAT *folding_time_out,
                                      std::chrono::steady_clock::time_point
                                          *first_device_ready_out,
                                      HPRLP_FLOAT *presolve_core_time_out = nullptr,
                                      HPRLP_FLOAT *preprocessing_time_out = nullptr);

bool ijoc_apply_gpu_postsolve(HPRLP_results *result,
                                  void *presolver_handle,
                                  int original_m,
                                  int original_n);

bool ijoc_gpu_postsolve_and_validate(HPRLP_results *result,
                                             const LP_info_cpu *original_model,
                                             void *presolver_handle,
                                             const IntegrationParameters *param);

void ijoc_free_gpu(void *presolver_handle);

#endif
