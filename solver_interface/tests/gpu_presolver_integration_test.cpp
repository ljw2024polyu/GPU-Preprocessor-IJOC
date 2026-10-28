#include "integration.hpp"
#include "adapters/gpu.hpp"
#include "gpu/preprocessing/preprocess.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

// Duplicate columns exercise folding/unfolding. The fixed fifth variable
// exercises presolve recovery and its +14 objective constant shift.
static void check_pipeline(bool folding, bool presolve = true, bool new_rules = true) {
    int row[] = {0, 5, 10};
    int col[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    double a[] = {1, 1, 2, 2, 1, 3, 3, 1, 1, 1};
    double al[] = {-INFINITY, -INFINITY}, au[] = {12, 14};
    double lo[] = {0, 0, 0, 0, 2}, up[] = {INFINITY, INFINITY, INFINITY, INFINITY, 2};
    double c[] = {-3, -3, -5, -5, 7};
    auto *model = create_model_from_arrays(2, 5, 10, row, col, a, al, au, lo, up, c, false);
    require(model != nullptr, "model creation failed");
    IntegrationParameters p;
    p.backend = IJOC_PRESOLVER_GPU;
    p.enable_gpu_folding = folding;
    p.enable_gpu_presolve = presolve;
    p.enable_gpu_new_rules = new_rules;
    p.stop_tol = 1e-8;
    p.time_limit = 30;
    p.print_debug_info = true;

    // Check the external preprocessor handoff and the official release solve.
    LP_info_cpu reduced{};
    void *handle = nullptr;
    double presolve_time = 0, presolve_core_time = 0, folding_time = 0;
    double preprocessing_time = 0;
    require(ijoc_run_gpu_presolve(model, &p, &reduced, &handle,
                                     &presolve_time, &folding_time,
                                     &presolve_core_time, &preprocessing_time), "CPU presolve handoff failed");
    require(preprocessing_time > 0 && preprocessing_time >= folding_time + presolve_time,
            "CPU preprocessing timer did not include folding and presolve");
    if (presolve) {
        require(presolve_core_time > 0 && presolve_time == presolve_core_time,
                "CPU handoff did not report native core as presolve time");
    }
    require(reduced.n < model->n, "test did not reduce the model");
    require(std::abs(reduced.obj_constant - (presolve ? 14 : 0)) < 1e-10,
            "objective constant was lost");
    if (!presolve) {
        require(presolve_time == 0, "folding only unexpectedly ran presolve");
        require(presolve_core_time == 0, "folding only unexpectedly measured core time");
        require(reduced.n == 3, "folding only did not retain the fixed column");
    }
    free_lp_info_cpu(&reduced);
    ijoc_free_gpu(handle);

    auto result = solve_preprocessed(model, &p);
    require(std::strcmp(result.status, "OPTIMAL") == 0, "solver did not return OPTIMAL");
    if (presolve) {
        require(result.presolve_core_time > 0 &&
                result.presolve_time == result.presolve_core_time,
                "GPU presolve time is not the native core measurement");
    } else {
        require(result.presolve_time == 0, "device folding only ran presolve");
        require(result.presolve_core_time == 0, "device folding only measured core time");
    }
    require(result.timing.presolve_time == result.presolve_time,
            "GPU result outer timer fields disagree");
    require(result.preprocessing_time > 0 &&
            result.preprocessing_time >= result.folding_time + result.presolve_time,
            "GPU preprocessing timer did not cover folding through presolve");
    require(result.x && result.y && result.z, "postsolve did not return original vectors");
    require(std::abs(result.x[0] + result.x[1] - 2.8) < 1e-5, "first variable pair recovery failed");
    require(std::abs(result.x[2] + result.x[3] - 3.6) < 1e-5, "second variable pair recovery failed");
    require(std::abs(result.x[4] - 2) < 1e-10, "fixed variable recovery failed");
    double objective = 0;
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
    require(std::abs(objective + 12.4) < 1e-5, "original objective is incorrect");
    for (int i = 0; i < 2; ++i) {
        double activity = 0;
        for (int j = 0; j < 5; ++j) activity += a[5*i+j] * result.x[j];
        require(activity <= au[i] + 1e-6, "original row feasibility failed");
        require(result.y[i] <= 1e-6, "wrong row dual sign");
        require(std::abs(result.y[i] * (activity - au[i])) < 1e-5, "row complementarity failed");
    }
    std::free(result.x); std::free(result.y); std::free(result.z);
    free_model(model);
    std::cout << "Original-space recovery passed (folding=" << folding
              << ", presolve=" << presolve << ", new_rules=" << new_rules << ")\n";
}

// Each zero-cost z is a certified McCormick auxiliary. Coupling rows keep
// x/y in the remaining LP, so classical rules cannot erase the comparison.
static void check_new_rule_switch() {
    constexpr int blocks = 128;  // Production zero-cost product candidate threshold.
    constexpr int n = 3 * blocks;
    std::vector<int> row{0}, col;
    std::vector<double> a, al, au;
    auto add_row = [&](std::initializer_list<int> columns,
                       std::initializer_list<double> values,
                       double lower, double upper) {
        col.insert(col.end(), columns.begin(), columns.end());
        a.insert(a.end(), values.begin(), values.end());
        row.push_back(static_cast<int>(col.size()));
        al.push_back(lower);
        au.push_back(upper);
    };
    for (int i = 0; i < blocks; ++i) {
        const int x = 2*i, y = 2*i+1, z = 2*blocks+i;
        add_row({x, y, z}, {-1, -1, 1}, -1, INFINITY);
        add_row({x, z}, {-1, 1}, -INFINITY, 0);
        add_row({y, z}, {-1, 1}, -INFINITY, 0);
    }
    for (int i = 0; i < blocks; ++i) {
        const int x = 2*i, y = 2*((i+1)%blocks)+1;
        if (x < y) add_row({x, y}, {1, 1}, 0.5, 1.5);
        else add_row({y, x}, {1, 1}, 0.5, 1.5);
    }
    for (int parity = 0; parity < 2; ++parity) {
        for (int i = 0; i < blocks; ++i) {
            col.push_back(2*i+parity);
            a.push_back(1);
        }
        row.push_back(static_cast<int>(col.size()));
        al.push_back(blocks/4.0);
        au.push_back(INFINITY);
    }
    std::vector<double> lo(n, 0), up(n, 1), c(n, 0);
    for (int j = 0; j < 2*blocks; ++j) c[j] = 1;
    auto *model = create_model_from_arrays(static_cast<int>(al.size()), n,
        static_cast<int>(a.size()), row.data(), col.data(), a.data(),
        al.data(), au.data(), lo.data(), up.data(), c.data(), false);
    require(model != nullptr, "Zero-cost product fixture creation failed");
    int reduced_columns[2]{};
    for (int enabled = 0; enabled <= 1; ++enabled) {
        IntegrationParameters p;
        p.backend = IJOC_PRESOLVER_GPU;
        p.enable_gpu_folding = false;
        p.enable_gpu_new_rules = enabled != 0;
        LP_info_cpu reduced{};
        void *handle = nullptr;
        double presolve_time = 0, folding_time = 0;
        require(ijoc_run_gpu_presolve(model, &p, &reduced, &handle,
            &presolve_time, &folding_time), "Zero-cost product presolve handoff failed");
        require(folding_time == 0, "rule ablation unexpectedly ran folding");
        reduced_columns[enabled] = reduced.n;
        free_lp_info_cpu(&reduced);
        ijoc_free_gpu(handle);
    }
    require(reduced_columns[0] == n, "classical rules unexpectedly removed the fixture auxiliaries");
    require(reduced_columns[1] <= 2*blocks, "new-rule switch did not enable zero-cost product variable elimination");
    free_model(model);
    std::cout << "New-rule switch changed reduced columns: "
              << reduced_columns[0] << " -> " << reduced_columns[1] << '\n';
}

int main() {
    try {
        check_pipeline(false);
        check_pipeline(true);
        check_pipeline(true, false);
        check_pipeline(false, true, false);
        check_new_rule_switch();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
