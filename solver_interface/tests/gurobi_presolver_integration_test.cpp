// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "integration.hpp"
#include "adapters/gurobi.hpp"
#include "gpu/preprocessing/preprocess.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void check_reduced_pipeline() {
    int row[] = {0, 5, 10};
    int col[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    double a[] = {1, 1, 2, 2, 1, 3, 3, 1, 1, 1};
    double al[] = {-INFINITY, -INFINITY}, au[] = {12, 14};
    double lo[] = {0, 0, 0, 0, 2}, up[] = {INFINITY, INFINITY, INFINITY, INFINITY, 2};
    double c[] = {-3, -3, -5, -5, 7};
    auto *model = create_model_from_arrays_with_obj_constant(
        2, 5, 10, row, col, a, al, au, lo, up, c, 3.5, false);
    require(model != nullptr, "model creation failed");
    IntegrationParameters p;
    p.backend = IJOC_PRESOLVER_GUROBI;
    p.stop_tol = 1e-8;
    p.print_debug_info = true;

    // The HPRLP time limit must not limit the Gurobi presolve phase.
    p.time_limit = 1e-12;
    LP_info_cpu reduced{};
    void *handle = nullptr;
    double presolve_time = 0;
    bool timed_out = true;
    require(ijoc_run_gurobi_presolve(model, &p, &reduced, &handle,
                                       &presolve_time, &timed_out),
            "Gurobi presolve handoff failed");
    require(!timed_out && presolve_time > p.time_limit,
            "HPRLP time limit incorrectly limited Gurobi presolve");
    require(reduced.n < model->n && reduced.n > 0 && reduced.m > 0 &&
            reduced.A && reduced.A->numElements > 0,
            "test did not produce a nonempty reduced LP");
    require(std::abs(reduced.obj_constant - 17.5) < 1e-10,
            "fixed-variable or input objective constant was lost");
    free_lp_info_cpu(&reduced);
    ijoc_free_gurobi(handle);

    p.time_limit = 30;
    auto result = solve_preprocessed(model, &p);
    require(std::strcmp(result.status, "OPTIMAL") == 0, "reduced solve failed");
    require(std::abs(result.primal_obj + 8.9) < 1e-5, "reduced objective is incorrect");
    require(!result.x && !result.y && !result.z,
            "reduced vectors escaped through the original-space API");
    require(result.preprocessing_time > 0 &&
            result.time >= result.preprocessing_time &&
            result.presolve_time == result.preprocessing_time &&
            std::isnan(result.presolve_core_time),
            "Gurobi wall time was lost or mislabeled as native core");
    free_model(model);
}

static void check_fixed_model(bool infeasible) {
    int row[] = {0, 1}, col[] = {0};
    double a[] = {1}, al[] = {-INFINITY}, au[] = {infeasible ? 1.0 : 3.0};
    double lo[] = {2}, up[] = {2}, c[] = {7};
    auto *model = create_model_from_arrays_with_obj_constant(
        1, 1, 1, row, col, a, al, au, lo, up, c, 3.5, false);
    require(model != nullptr, "fixed model creation failed");
    IntegrationParameters p;
    p.backend = IJOC_PRESOLVER_GUROBI;
    auto result = solve_preprocessed(model, &p);
    require(!result.x && !result.y && !result.z, "unexpected original vectors");
    if (infeasible) {
        require(std::strcmp(result.status, "PRESOLVE_FAILED") == 0,
                "infeasible Gurobi model silently fell back to HPRLP");
    } else {
        require(std::strcmp(result.status, "OPTIMAL") == 0 && result.iter == 0,
                "fully eliminated LP was not recognized as solved");
        require(std::abs(result.primal_obj - 17.5) < 1e-10,
                "fully eliminated objective constant was lost");
        require(result.preprocessing_time > 0 &&
                result.time >= result.preprocessing_time && std::isnan(result.presolve_core_time),
                "fully eliminated LP timing is incorrect");
    }
    free_model(model);
}

int main() {
    try {
        check_reduced_pipeline();
        check_fixed_model(false);
        check_fixed_model(true);
        std::cout << "Gurobi reduced-space integration passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
