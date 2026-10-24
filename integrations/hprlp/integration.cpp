#include "integration.hpp"
#include "adapters/gpu.hpp"
#include "adapters/pslp.hpp"
#include "adapters/gurobi.hpp"
#include "gpu/preprocessing/preprocess.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace {
using Clock = std::chrono::steady_clock;

struct PresolvedModel {
    LP_info_cpu model{};
    void* handle = nullptr;
    IjocPresolver backend;
    explicit PresolvedModel(IjocPresolver value) : backend(value) {}
    ~PresolvedModel() {
        if (backend == IJOC_PRESOLVER_GPU) ijoc_free_gpu(handle);
        else if (backend == IJOC_PRESOLVER_GUROBI) ijoc_free_gurobi(handle);
        else ijoc_free_pslp(handle);
        free_lp_info_cpu(&model);
    }
};

void discard_vectors(HPRLP_results& result) {
    std::free(result.x); std::free(result.y); std::free(result.z);
    result.x = result.y = result.z = nullptr;
}

void status(ExperimentResult& result, const char* value) {
    std::strncpy(result.status, value, sizeof(result.status) - 1);
    result.status[sizeof(result.status) - 1] = '\0';
}
}  // namespace

ExperimentResult solve_preprocessed(const LP_info_cpu* model,
                                    const IntegrationParameters* options) {
    const IntegrationParameters p = options ? *options : IntegrationParameters{};
    // Copy only the official parameter structure. The solver receives no
    // experimental fields and never performs a second presolve on the reduced LP.
    HPRLP_parameters solver_parameters = p;
    solver_parameters.presolver = HPRLP_PRESOLVER_NONE;
    solver_parameters.use_presolve = false;
    ExperimentResult result{};
    if (!model || !model->A) {
        status(result, "ERROR");
        return result;
    }
    std::cout << "HPR-LP source: official v0.1.3 (89d3cc06806b4d31d029f7b5c8591db58219ff6d)\n";
    if (p.backend == IJOC_PRESOLVER_NONE) {
        static_cast<HPRLP_results&>(result) = ::solve(model, &solver_parameters);
        return result;
    }

    const auto start = Clock::now();
    PresolvedModel reduced(p.backend);
    double presolve = 0, folding = 0, preprocessing = 0;
    bool ok = false;
    if (p.backend == IJOC_PRESOLVER_GPU) {
        ok = ijoc_run_gpu_presolve(model, &p, &reduced.model, &reduced.handle,
                                  &presolve, &folding, &result.presolve_core_time,
                                  &preprocessing);
    } else if (p.backend == IJOC_PRESOLVER_GUROBI) {
        bool timed_out = false;
        ok = ijoc_run_gurobi_presolve(model, &p, &reduced.model, &reduced.handle,
                                     &preprocessing, &timed_out);
        // GRBpresolvemodel exposes a wall interval, not an internal core timer.
        presolve = preprocessing;
        result.presolve_core_time = std::numeric_limits<double>::quiet_NaN();
    } else {
        ok = ijoc_run_pslp_presolve(model, &p, &reduced.model, &reduced.handle,
                                   &result.presolve_core_time);
        presolve = result.presolve_core_time;
        preprocessing = std::chrono::duration<double>(Clock::now() - start).count();
    }
    result.preprocessing_time = preprocessing;
    if (!ok) {
        status(result, "PRESOLVE_FAILED");
        return result;
    }

    if (reduced.model.m == 0 && reduced.model.n == 0 && p.backend == IJOC_PRESOLVER_GUROBI) {
        status(result, "OPTIMAL");
        result.primal_obj = reduced.model.obj_constant;
        result.residuals = result.gap = 0;
    } else if (reduced.model.m <= 0 || reduced.model.n <= 0 ||
               !reduced.model.A || reduced.model.A->numElements <= 0) {
        status(result, "REDUCED_MODEL_UNSUPPORTED");
        return result;
    } else {
        static_cast<HPRLP_results&>(result) = ::solve(&reduced.model, &solver_parameters);
    }

    if (p.backend == IJOC_PRESOLVER_GUROBI) {
        discard_vectors(result);
        std::cout << "original_solution_recovery: NOT_PERFORMED (Gurobi reduced-space result)\n";
    } else if (result.x && result.y && result.z) {
        const bool recovered = p.backend == IJOC_PRESOLVER_GPU
            ? ijoc_gpu_postsolve_and_validate(&result, model, reduced.handle, &p)
            : ijoc_pslp_postsolve_and_validate(&result, model, reduced.handle, &p);
        if (!recovered) {
            discard_vectors(result);
            status(result, "POSTSOLVE_FAILED");
        }
    }
    result.preprocessing_time = preprocessing;
    result.folding_time = folding;
    result.presolve_time = result.timing.presolve_time = presolve;
    result.time = result.timing.total_time =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

void print_experiment_summary(const ExperimentResult& result) {
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << "\n=== External preprocessing + official HPR-LP v0.1.3 ===\n"
              << "Status: " << result.status << '\n'
              << "Iterations: " << result.iter << '\n'
              << std::setprecision(12)
              << "Primal Objective: " << result.primal_obj << '\n'
              << "Preprocessing Time: " << result.preprocessing_time << " seconds\n"
              << "Presolve Time: " << result.presolve_time << " seconds\n"
              << "Solve Time: " << result.timing.solve_time << " seconds\n"
              << "Folding Time: " << result.folding_time << " seconds\n"
              << "Total Time: " << result.timing.total_time << " seconds\n";
    std::cout.flags(flags);
    std::cout.precision(precision);
}
