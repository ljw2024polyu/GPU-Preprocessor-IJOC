#ifndef IJOC_GUROBI_ADAPTER_HPP
#define IJOC_GUROBI_ADAPTER_HPP

#include "integration.hpp"

// Automatic presolve (Presolve=-1), without a presolve time limit.
// The reduced model has no public original-space postsolve mapping.
bool ijoc_run_gurobi_presolve(const LP_info_cpu *model,
                                  const IntegrationParameters *param,
                                  LP_info_cpu *reduced_model,
                                  void **presolver_handle_out,
                                  HPRLP_FLOAT *presolve_time_out,
                                  bool *timed_out_out);

void ijoc_free_gurobi(void *presolver_handle);

#endif
