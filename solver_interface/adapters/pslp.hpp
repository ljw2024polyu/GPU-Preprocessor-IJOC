#ifndef IJOC_PSLP_ADAPTER_HPP
#define IJOC_PSLP_ADAPTER_HPP

#include "integration.hpp"

bool ijoc_run_pslp_presolve(const LP_info_cpu *model,
                                const IntegrationParameters *param,
                                LP_info_cpu *reduced_model,
                                void **presolver_handle_out,
                                HPRLP_FLOAT *presolve_core_time_out = nullptr);

bool ijoc_apply_pslp_postsolve(HPRLP_results *result,
                                   void *presolver_handle,
                                   int original_m,
                                   int original_n);

bool ijoc_pslp_postsolve_and_validate(HPRLP_results *result,
                                         const LP_info_cpu *original_model,
                                         void *presolver_handle,
                                         const IntegrationParameters *param);

void ijoc_free_pslp(void *presolver_handle);

#endif
