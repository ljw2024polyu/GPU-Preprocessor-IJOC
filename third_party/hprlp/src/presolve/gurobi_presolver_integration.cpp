#include "presolve/gurobi_presolver_integration.h"

#include "io/mps_reader.h"
#include "gpu/preprocessing/preprocess.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#if HPRLP_HAS_GUROBI
#include <gurobi_c.h>
#endif

namespace {

struct GurobiPresolverHandle {
    int original_m = 0;
    int original_n = 0;
    int reduced_m = 0;
    int reduced_n = 0;
    int reduced_nnz = 0;
};

void initialize_empty_model(LP_info_cpu *model) {
    if (!model) {
        return;
    }
    model->m = 0;
    model->n = 0;
    model->A = nullptr;
    model->AL = nullptr;
    model->AU = nullptr;
    model->c = nullptr;
    model->l = nullptr;
    model->u = nullptr;
    model->obj_constant = 0.0;
}

#if HPRLP_HAS_GUROBI

struct GurobiResources {
    GRBenv *env = nullptr;
    GRBmodel *original = nullptr;
    GRBmodel *presolved = nullptr;

    ~GurobiResources() {
        if (presolved) {
            GRBfreemodel(presolved);
        }
        if (original) {
            GRBfreemodel(original);
        }
        if (env) {
            GRBfreeenv(env);
        }
    }
};

void check_gurobi(int error, GRBenv *env, const char *operation) {
    if (error == 0) {
        return;
    }
    std::string message(operation);
    if (env) {
        const char *detail = GRBgeterrormsg(env);
        if (detail && detail[0] != '\0') {
            message += ": ";
            message += detail;
        }
    }
    message += " (Gurobi error ";
    message += std::to_string(error);
    message += ")";
    throw std::runtime_error(message);
}

template <typename T>
T *data_or_null(std::vector<T> &values) {
    return values.empty() ? nullptr : values.data();
}

HPRLP_FLOAT hprlp_bound(double value) {
    if (value >= GRB_INFINITY) {
        return std::numeric_limits<HPRLP_FLOAT>::infinity();
    }
    if (value <= -GRB_INFINITY) {
        return -std::numeric_limits<HPRLP_FLOAT>::infinity();
    }
    return static_cast<HPRLP_FLOAT>(value);
}

double gurobi_bound(HPRLP_FLOAT value) {
    if (std::isinf(value)) {
        return value < 0.0 ? -GRB_INFINITY : GRB_INFINITY;
    }
    return static_cast<double>(value);
}

void build_gurobi_model(const LP_info_cpu *model, GurobiResources *resources) {
    const int m = model->m;
    const int n = model->n;

    std::vector<int> first_row(static_cast<size_t>(m), -1);
    std::vector<int> second_row(static_cast<size_t>(m), -1);
    std::vector<char> sense;
    std::vector<double> rhs;
    sense.reserve(static_cast<size_t>(m) * 2);
    rhs.reserve(static_cast<size_t>(m) * 2);

    for (int row = 0; row < m; ++row) {
        const bool has_lower = std::isfinite(model->AL[row]);
        const bool has_upper = std::isfinite(model->AU[row]);
        if (has_lower && has_upper && model->AL[row] == model->AU[row]) {
            first_row[row] = static_cast<int>(sense.size());
            sense.push_back(GRB_EQUAL);
            rhs.push_back(static_cast<double>(model->AL[row]));
        } else {
            if (has_lower) {
                first_row[row] = static_cast<int>(sense.size());
                sense.push_back(GRB_GREATER_EQUAL);
                rhs.push_back(static_cast<double>(model->AL[row]));
            }
            if (has_upper) {
                const int mapped = static_cast<int>(sense.size());
                if (first_row[row] < 0) {
                    first_row[row] = mapped;
                } else {
                    second_row[row] = mapped;
                }
                sense.push_back(GRB_LESS_EQUAL);
                rhs.push_back(static_cast<double>(model->AU[row]));
            }
        }
    }

    long long expanded_nnz = 0;
    for (int row = 0; row < m; ++row) {
        const int copies = (first_row[row] >= 0 ? 1 : 0) + (second_row[row] >= 0 ? 1 : 0);
        expanded_nnz += static_cast<long long>(copies) *
                        static_cast<long long>(model->A->rowPtr[row + 1] - model->A->rowPtr[row]);
    }
    if (expanded_nnz > std::numeric_limits<int>::max()) {
        throw std::runtime_error("expanded Gurobi matrix exceeds the 32-bit C API limit");
    }

    std::vector<int> column_counts(static_cast<size_t>(n), 0);
    for (int row = 0; row < m; ++row) {
        const int copies = (first_row[row] >= 0 ? 1 : 0) + (second_row[row] >= 0 ? 1 : 0);
        for (int entry = model->A->rowPtr[row]; entry < model->A->rowPtr[row + 1]; ++entry) {
            const int col = model->A->colIndex[entry];
            if (col < 0 || col >= n) {
                throw std::runtime_error("invalid column index in HPRLP matrix");
            }
            column_counts[col] += copies;
        }
    }

    std::vector<int> vbeg(static_cast<size_t>(n), 0);
    std::vector<int> vlen = column_counts;
    int offset = 0;
    for (int col = 0; col < n; ++col) {
        vbeg[col] = offset;
        offset += column_counts[col];
    }

    std::vector<int> cursor = vbeg;
    std::vector<int> vind(static_cast<size_t>(expanded_nnz));
    std::vector<double> vval(static_cast<size_t>(expanded_nnz));
    for (int row = 0; row < m; ++row) {
        const int mapped_rows[2] = {first_row[row], second_row[row]};
        for (int copy = 0; copy < 2; ++copy) {
            if (mapped_rows[copy] < 0) {
                continue;
            }
            for (int entry = model->A->rowPtr[row]; entry < model->A->rowPtr[row + 1]; ++entry) {
                const int col = model->A->colIndex[entry];
                const int destination = cursor[col]++;
                vind[destination] = mapped_rows[copy];
                vval[destination] = static_cast<double>(model->A->value[entry]);
            }
        }
    }

    std::vector<double> objective(static_cast<size_t>(n));
    std::vector<double> lower(static_cast<size_t>(n));
    std::vector<double> upper(static_cast<size_t>(n));
    for (int col = 0; col < n; ++col) {
        objective[col] = static_cast<double>(model->c[col]);
        lower[col] = gurobi_bound(model->l[col]);
        upper[col] = gurobi_bound(model->u[col]);
    }

    const int error = GRBloadmodel(
        resources->env,
        &resources->original,
        "hprlp_gurobi_presolve",
        n,
        static_cast<int>(sense.size()),
        1,
        static_cast<double>(model->obj_constant),
        data_or_null(objective),
        data_or_null(sense),
        data_or_null(rhs),
        data_or_null(vbeg),
        data_or_null(vlen),
        data_or_null(vind),
        data_or_null(vval),
        data_or_null(lower),
        data_or_null(upper),
        nullptr,
        nullptr,
        nullptr);
    check_gurobi(error, resources->env, "GRBloadmodel");
}

void copy_presolved_model(GRBmodel *model,
                          int reduced_m,
                          int reduced_n,
                          int reduced_nnz,
                          LP_info_cpu *output) {
    GRBenv *env = GRBgetenv(model);
    double objective_constant = 0.0;
    check_gurobi(GRBgetdblattr(model, GRB_DBL_ATTR_OBJCON, &objective_constant),
                  env, "get presolved objective constant");
    output->obj_constant = static_cast<HPRLP_FLOAT>(objective_constant);

    if (reduced_m <= 0 || reduced_n <= 0 || reduced_nnz <= 0) {
        output->m = reduced_m;
        output->n = reduced_n;
        return;
    }

    std::vector<double> objective(static_cast<size_t>(reduced_n));
    std::vector<double> lower_raw(static_cast<size_t>(reduced_n));
    std::vector<double> upper_raw(static_cast<size_t>(reduced_n));
    std::vector<double> rhs(static_cast<size_t>(reduced_m));
    std::vector<char> sense(static_cast<size_t>(reduced_m));

    check_gurobi(GRBgetdblattrarray(model, GRB_DBL_ATTR_OBJ, 0, reduced_n, objective.data()),
                  env, "get presolved objective");
    check_gurobi(GRBgetdblattrarray(model, GRB_DBL_ATTR_LB, 0, reduced_n, lower_raw.data()),
                  env, "get presolved lower bounds");
    check_gurobi(GRBgetdblattrarray(model, GRB_DBL_ATTR_UB, 0, reduced_n, upper_raw.data()),
                  env, "get presolved upper bounds");
    check_gurobi(GRBgetdblattrarray(model, GRB_DBL_ATTR_RHS, 0, reduced_m, rhs.data()),
                  env, "get presolved right-hand sides");
    check_gurobi(GRBgetcharattrarray(model, GRB_CHAR_ATTR_SENSE, 0, reduced_m, sense.data()),
                  env, "get presolved constraint senses");

    std::vector<int> vbeg(static_cast<size_t>(reduced_n));
    std::vector<int> row_index(static_cast<size_t>(reduced_nnz));
    std::vector<double> csc_value(static_cast<size_t>(reduced_nnz));
    int actual_nnz = 0;
    check_gurobi(GRBgetvars(model, &actual_nnz, vbeg.data(), row_index.data(),
                            csc_value.data(), 0, reduced_n),
                  env, "get presolved constraint matrix");
    if (actual_nnz != reduced_nnz) {
        throw std::runtime_error("Gurobi returned an inconsistent presolved nonzero count");
    }

    std::vector<int> row_ptr(static_cast<size_t>(reduced_m + 1), 0);
    for (int entry = 0; entry < reduced_nnz; ++entry) {
        const int row = row_index[entry];
        if (row < 0 || row >= reduced_m) {
            throw std::runtime_error("invalid row index in Gurobi presolved matrix");
        }
        ++row_ptr[static_cast<size_t>(row + 1)];
    }
    for (int row = 0; row < reduced_m; ++row) {
        row_ptr[static_cast<size_t>(row + 1)] += row_ptr[static_cast<size_t>(row)];
    }

    std::vector<int> cursor(row_ptr.begin(), row_ptr.end() - 1);
    std::vector<int> column_index(static_cast<size_t>(reduced_nnz));
    std::vector<HPRLP_FLOAT> csr_value(static_cast<size_t>(reduced_nnz));
    for (int col = 0; col < reduced_n; ++col) {
        const int begin = vbeg[col];
        const int end = col + 1 < reduced_n ? vbeg[col + 1] : reduced_nnz;
        for (int entry = begin; entry < end; ++entry) {
            const int row = row_index[entry];
            const int destination = cursor[row]++;
            column_index[destination] = col;
            csr_value[destination] = static_cast<HPRLP_FLOAT>(csc_value[entry]);
        }
    }

    std::vector<HPRLP_FLOAT> row_lower(static_cast<size_t>(reduced_m));
    std::vector<HPRLP_FLOAT> row_upper(static_cast<size_t>(reduced_m));
    const HPRLP_FLOAT infinity = std::numeric_limits<HPRLP_FLOAT>::infinity();
    for (int row = 0; row < reduced_m; ++row) {
        if (sense[row] == GRB_LESS_EQUAL) {
            row_lower[row] = -infinity;
            row_upper[row] = static_cast<HPRLP_FLOAT>(rhs[row]);
        } else if (sense[row] == GRB_GREATER_EQUAL) {
            row_lower[row] = static_cast<HPRLP_FLOAT>(rhs[row]);
            row_upper[row] = infinity;
        } else if (sense[row] == GRB_EQUAL) {
            row_lower[row] = static_cast<HPRLP_FLOAT>(rhs[row]);
            row_upper[row] = static_cast<HPRLP_FLOAT>(rhs[row]);
        } else {
            throw std::runtime_error("unsupported constraint sense in Gurobi presolved model");
        }
    }

    std::vector<HPRLP_FLOAT> variable_lower(static_cast<size_t>(reduced_n));
    std::vector<HPRLP_FLOAT> variable_upper(static_cast<size_t>(reduced_n));
    std::vector<HPRLP_FLOAT> cost(static_cast<size_t>(reduced_n));
    for (int col = 0; col < reduced_n; ++col) {
        variable_lower[col] = hprlp_bound(lower_raw[col]);
        variable_upper[col] = hprlp_bound(upper_raw[col]);
        cost[col] = static_cast<HPRLP_FLOAT>(objective[col]);
    }

    CSRMatrix csr;
    csr.nrows = reduced_m;
    csr.ncols = reduced_n;
    csr.nnz = reduced_nnz;
    csr.row_ptr = row_ptr.data();
    csr.col_idx = column_index.data();
    csr.values = csr_value.data();
    build_model_from_arrays(&csr, row_lower.data(), row_upper.data(),
                            variable_lower.data(), variable_upper.data(), cost.data(),
                            static_cast<HPRLP_FLOAT>(objective_constant), output);
    if (!output->A || output->m != reduced_m || output->n != reduced_n) {
        throw std::runtime_error("failed to convert Gurobi presolved model to HPRLP format");
    }
}

#endif

} // namespace

bool run_embedded_gurobi_presolve(const LP_info_cpu *model,
                                  const HPRLP_parameters *param,
                                  LP_info_cpu *reduced_model,
                                  void **presolver_handle_out,
                                  HPRLP_FLOAT *presolve_time_out,
                                  bool *timed_out_out) {
    if (!model || !model->A || !reduced_model || !presolver_handle_out) {
        return false;
    }

    initialize_empty_model(reduced_model);
    *presolver_handle_out = nullptr;
    if (presolve_time_out) {
        *presolve_time_out = 0.0;
    }
    if (timed_out_out) {
        *timed_out_out = false;
    }
    (void)param;

#if !HPRLP_HAS_GUROBI
    std::cerr << "[error] Gurobi support was not found when HPRLP was built; "
                 "set GUROBI_HOME or install third_party/gurobi/linux64"
              << std::endl;
    return false;
#else
    GurobiResources resources;
    try {
        check_gurobi(GRBemptyenv(&resources.env), resources.env, "GRBemptyenv");
        check_gurobi(GRBsetintparam(resources.env, GRB_INT_PAR_OUTPUTFLAG, 1),
                      resources.env, "enable Gurobi logging");
        check_gurobi(GRBstartenv(resources.env), resources.env, "start Gurobi environment");
        build_gurobi_model(model, &resources);

        GRBenv *model_env = GRBgetenv(resources.original);
        check_gurobi(GRBupdatemodel(resources.original), model_env, "update Gurobi model");
        check_gurobi(GRBsetintparam(model_env, GRB_INT_PAR_PRESOLVE, -1),
                      model_env, "set Gurobi Presolve=-1");
        // Keep presolve unlimited; the public --time-limit belongs to the
        // subsequent HPRLP solve rather than to model reduction.
        const double presolve_time_limit = GRB_INFINITY;
        check_gurobi(GRBsetdblparam(model_env, GRB_DBL_PAR_TIMELIMIT,
                                    presolve_time_limit),
                      model_env, "set Gurobi presolve time limit");

        std::cout << "Gurobi presolve configuration: Presolve=-1, TimeLimit=Infinity, OutputFlag=1" << std::endl;
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const int presolve_error = GRBpresolvemodel(resources.original, &resources.presolved);
        const HPRLP_FLOAT elapsed = std::chrono::duration_cast<std::chrono::duration<HPRLP_FLOAT> >(
            std::chrono::steady_clock::now() - start).count();
        if (presolve_time_out) {
            *presolve_time_out = elapsed;
        }
        check_gurobi(presolve_error, model_env, "GRBpresolvemodel");
        const bool finite_time_limit = presolve_time_limit < GRB_INFINITY;
        if (finite_time_limit && static_cast<double>(elapsed) >= presolve_time_limit) {
            if (timed_out_out) {
                *timed_out_out = true;
            }
            std::cout << "Gurobi presolve reached TimeLimit=" << presolve_time_limit
                      << " seconds after " << elapsed
                      << " seconds; discarding the presolved model and falling back "
                         "to the original LP"
                      << std::endl;
            return false;
        }
        if (!resources.presolved) {
            throw std::runtime_error(
                "Gurobi did not return a reduced model (the input may be infeasible or unbounded)");
        }

        int reduced_m = 0;
        int reduced_n = 0;
        int reduced_nnz = 0;
        GRBenv *presolved_env = GRBgetenv(resources.presolved);
        check_gurobi(GRBgetintattr(resources.presolved, GRB_INT_ATTR_NUMCONSTRS, &reduced_m),
                      presolved_env, "get Gurobi presolved row count");
        check_gurobi(GRBgetintattr(resources.presolved, GRB_INT_ATTR_NUMVARS, &reduced_n),
                      presolved_env, "get Gurobi presolved column count");
        check_gurobi(GRBgetintattr(resources.presolved, GRB_INT_ATTR_NUMNZS, &reduced_nnz),
                      presolved_env, "get Gurobi presolved nonzero count");

        copy_presolved_model(resources.presolved, reduced_m, reduced_n, reduced_nnz, reduced_model);

        GurobiPresolverHandle *handle = new (std::nothrow) GurobiPresolverHandle;
        if (!handle) {
            throw std::runtime_error("failed to allocate Gurobi presolver handle");
        }
        handle->original_m = model->m;
        handle->original_n = model->n;
        handle->reduced_m = reduced_m;
        handle->reduced_n = reduced_n;
        handle->reduced_nnz = reduced_nnz;
        *presolver_handle_out = handle;
        std::cout << "Gurobi presolve core time: " << elapsed << " seconds" << std::endl;
        std::cout << "Gurobi reduced problem: (" << model->m << ", " << model->n
                  << ", " << model->A->numElements << ") -> (" << reduced_m << ", "
                  << reduced_n << ", " << reduced_nnz << ")" << std::endl;
        if (reduced_m == 0 || reduced_n == 0 || reduced_nnz == 0) {
            std::cout << "Gurobi presolve produced an empty/solved reduced model" << std::endl;
        }
        return true;
    } catch (const std::exception &error) {
        std::cerr << "[error] Gurobi presolve failed: " << error.what() << std::endl;
        free_lp_info_cpu(reduced_model);
        initialize_empty_model(reduced_model);
        return false;
    }
#endif
}

void free_embedded_gurobi_presolver(void *presolver_handle) {
    delete static_cast<GurobiPresolverHandle *>(presolver_handle);
}
