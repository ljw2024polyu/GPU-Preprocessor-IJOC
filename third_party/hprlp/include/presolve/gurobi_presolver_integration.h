#ifndef HPRLP_GUROBI_PRESOLVER_INTEGRATION_H
#define HPRLP_GUROBI_PRESOLVER_INTEGRATION_H

#include "HPRLP.h"

// Same Presolve=-1 / unlimited presolve configuration as the paper project.
// The reduced model has no public original-space postsolve mapping.
bool run_embedded_gurobi_presolve(const LP_info_cpu *model,
                                  const HPRLP_parameters *param,
                                  LP_info_cpu *reduced_model,
                                  void **presolver_handle_out,
                                  HPRLP_FLOAT *presolve_time_out,
                                  bool *timed_out_out);

void free_embedded_gurobi_presolver(void *presolver_handle);

#endif
