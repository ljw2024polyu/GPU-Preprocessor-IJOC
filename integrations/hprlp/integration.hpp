#pragma once
#include "HPRLP.h"

enum IjocPresolver {
    IJOC_PRESOLVER_PSLP,
    IJOC_PRESOLVER_GPU,
    IJOC_PRESOLVER_NONE,
    IJOC_PRESOLVER_PSLP0011,
    IJOC_PRESOLVER_GUROBI
};

// Options belonging to the external pipeline, not to the official solver API.
struct IntegrationParameters : HPRLP_parameters {
    IjocPresolver backend = IJOC_PRESOLVER_GPU;
    bool enable_gpu_presolve = true;
    bool enable_gpu_new_rules = true;
};

struct ExperimentResult : HPRLP_results {
    double preprocessing_time = 0;
    double presolve_core_time = 0;
};

ExperimentResult solve_preprocessed(const LP_info_cpu* model,
                                    const IntegrationParameters* options);
void print_experiment_summary(const ExperimentResult& result);
