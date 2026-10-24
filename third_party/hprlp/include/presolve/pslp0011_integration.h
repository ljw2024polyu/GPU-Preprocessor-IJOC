#ifndef HPRLP_PSLP0011_INTEGRATION_H
#define HPRLP_PSLP0011_INTEGRATION_H

#include "HPRLP.h"

// PSLP v0.0.11. The worker retains recovery data until the handle is freed.
bool run_embedded_pslp0011_presolve(const LP_info_cpu *model,
                                  const HPRLP_parameters *param,
                                  LP_info_cpu *reduced_model,
                                  void **presolver_handle_out,
                                  HPRLP_FLOAT *presolve_core_time_out = nullptr);

bool apply_embedded_pslp0011_postsolve(HPRLP_results *result,
                                     void *presolver_handle,
                                     int original_m,
                                     int original_n);

bool pslp0011_postsolve_and_validate_original_kkt(
    HPRLP_results *result,
    const LP_info_cpu *original_model,
    void *presolver_handle,
    const HPRLP_parameters *param);

void free_embedded_pslp0011_presolver(void *presolver_handle);

#endif
