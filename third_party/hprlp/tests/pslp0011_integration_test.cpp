#include "HPRLP.h"
#include "presolve/pslp_integration.h"
#include "presolve/pslp0011_integration.h"
#include "gpu/preprocessing/preprocess.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

// Duplicate columns and a fixed variable exercise primal/dual recovery.
// A nonzero input objective constant checks both constant contributions.
static void check_pipeline(HPRLP_presolve_backend backend, bool presolve_only) {
    int row[] = {0, 5, 10};
    int col[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    double a[] = {1, 1, 2, 2, 1, 3, 3, 1, 1, 1};
    double al[] = {-INFINITY, -INFINITY}, au[] = {12, 14};
    double lo[] = {0, 0, 0, 0, 2}, up[] = {INFINITY, INFINITY, INFINITY, INFINITY, 2};
    double c[] = {-3, -3, -5, -5, 7};
    auto *model = create_model_from_arrays_with_obj_constant(
        2, 5, 10, row, col, a, al, au, lo, up, c, 3.5, false);
    require(model != nullptr, "model creation failed");
    HPRLP_parameters p;
    p.presolver = backend;
    p.stop_tol = 1e-8;
    p.time_limit = 30;
    p.print_debug_info = true;

    // Require actual presolve success: an original-model fallback cannot pass.
    LP_info_cpu reduced{};
    void *handle = nullptr;
    const bool latest = backend == HPRLP_PRESOLVER_PSLP0011;
    double core_time = NAN;
    require(latest
        ? run_embedded_pslp0011_presolve(model, &p, &reduced, &handle, &core_time)
        : run_embedded_pslp_presolve(model, &p, &reduced, &handle, &core_time),
        "CPU presolve handoff failed");
    require(std::isfinite(core_time) && core_time > 0,
            "native PSLP core timer did not cross the worker transport");
    require(reduced.n < model->n, "test did not reduce the model");
    require(std::abs(reduced.obj_constant - 17.5) < 1e-10, "objective constant was lost");
    free_lp_info_cpu(&reduced);
    if (latest) free_embedded_pslp0011_presolver(handle);
    else free_embedded_pslp_presolver(handle);
    if (presolve_only) {
        free_model(model);
        std::cout << "CPU presolve handoff passed (backend=" << backend << ")\n";
        return;
    }

    auto result = solve(model, &p);
    require(std::strcmp(result.status, "OPTIMAL") == 0, "solver did not return OPTIMAL");
    require(result.x && result.y && result.z, "postsolve did not return original vectors");
    require(std::abs(result.x[0] + result.x[1] - 2.8) < 1e-5, "first variable pair recovery failed");
    require(std::abs(result.x[2] + result.x[3] - 3.6) < 1e-5, "second variable pair recovery failed");
    require(std::abs(result.x[4] - 2) < 1e-10, "fixed variable recovery failed");
    double objective = model->obj_constant;
    for (int j = 0; j < 5; ++j) {
        objective += c[j] * result.x[j];
        require(std::isfinite(result.x[j]) && result.x[j] >= lo[j] - 1e-6 &&
                result.x[j] <= up[j] + 1e-6, "original variable bound violation");
        double residual = c[j] - a[j] * result.y[0] - a[5+j] * result.y[1] - result.z[j];
        require(std::isfinite(residual) && std::abs(residual) < 1e-6, "original stationarity failed");
        if (j < 4) {
            require(result.z[j] >= -1e-6, "wrong bound dual sign");
            require(std::abs(result.z[j] * result.x[j]) < 1e-5, "bound complementarity failed");
        }
    }
    require(std::abs(objective + 8.9) < 1e-5, "original objective is incorrect");
    require(std::abs(result.primal_obj - objective) < 1e-5,
            "reported objective differs from original-space objective");
    require(result.presolve_time > 0 && result.time >= result.presolve_time,
            "CPU presolve timing was lost");
    require(std::isfinite(result.timing.presolve_core_time) &&
            result.timing.presolve_core_time > 0 &&
            result.timing.presolve_core_time == result.presolve_time &&
            result.timing.presolve_time == result.presolve_time,
            "PSLP core and outer timing boundaries are inconsistent");
    require(result.timing.preprocessing_time > result.presolve_time &&
            result.timing.total_time >= result.timing.preprocessing_time,
            "PSLP wall time was lost when presolve time became core time");
    for (int i = 0; i < 2; ++i) {
        double activity = 0;
        for (int j = 0; j < 5; ++j) activity += a[5*i+j] * result.x[j];
        require(activity <= au[i] + 1e-6, "original row feasibility failed");
        require(result.y[i] <= 1e-6, "wrong row dual sign");
        require(std::abs(result.y[i] * (activity - au[i])) < 1e-5, "row complementarity failed");
    }
    std::free(result.x); std::free(result.y); std::free(result.z);
    free_model(model);
    std::cout << "Original-space recovery passed (backend=" << backend << ")\n";
}

int main(int argc, char **argv) {
    const bool presolve_only = argc == 2 && std::strcmp(argv[1], "--presolve-only") == 0;
    if (argc > 1 && !presolve_only) return 2;
    try {
        // Both public selectors must use PSLP 0.0.11 and recover original solutions.
        check_pipeline(HPRLP_PRESOLVER_PSLP, presolve_only);
        check_pipeline(HPRLP_PRESOLVER_PSLP0011, presolve_only);
        check_pipeline(HPRLP_PRESOLVER_PSLP, presolve_only);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
