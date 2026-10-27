#include "adapters/gpu.hpp"

#include "gpu/memory/compressible_memory.h"
#include "io/mps_reader.h"
#include "gpu/preprocessing/preprocess.h"
#include "support/utils.h"

#include "gpu_presolver/folding/folding.hpp"
#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include "gpu_presolver/presolve/duplicate_rule_workspace.hpp"
#include "gpu_presolver/presolve/rules/rule_bound_strengthening.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_columns.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace gf = gpu_presolver::folding;
namespace gp = gpu_presolver::presolve;

struct OriginalKktMetrics {
    HPRLP_FLOAT primal_obj = 0.0;
    HPRLP_FLOAT dual_obj = 0.0;
    HPRLP_FLOAT primal_feas = 0.0;
    HPRLP_FLOAT dual_feas = 0.0;
    HPRLP_FLOAT gap = 0.0;
};

struct GpuPresolverHandle {
    gp::LPInfoGpu original_lp;
    gf::FoldingPipelineSummary folding_summary;
    double presolve_seconds = 0.0;
    double presolve_core_seconds = 0.0;
    double preprocessing_seconds = 0.0;
    double presolve_setup_seconds = 0.0;
    double presolve_workspace_release_seconds = 0.0;
    int reduced_m = 0;
    int reduced_n = 0;
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

void initialize_empty_device_model(LP_info_gpu *model) {
    if (!model) {
        return;
    }
    // Reset optional solver metadata along with the matrix/vector fields.
    *model = LP_info_gpu{};
}

void throw_if_cuda_error(cudaError_t status, const char *context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

// Reserve the two initial recovery-map sizes before the run_presolve timer.
// Keep the async pool's freed blocks across the setup synchronization so the
// first recovery allocations can reuse them; restore the pool policy later.
class PreparedRecoveryAsyncPool {
public:
    PreparedRecoveryAsyncPool(std::int32_t rows, std::int32_t cols) {
        if (rows <= 0 && cols <= 0) return;
        int device = 0;
        throw_if_cuda_error(cudaGetDevice(&device), "recovery pool device");
        throw_if_cuda_error(cudaDeviceGetDefaultMemPool(&pool_, device), "recovery pool query");
        throw_if_cuda_error(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                                   &old_threshold_), "recovery pool old threshold");
        std::uint64_t retain = std::numeric_limits<std::uint64_t>::max();
        throw_if_cuda_error(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                                   &retain), "recovery pool retain");
        try {
            if (rows > 0) {
                throw_if_cuda_error(cudaMallocAsync(&row_temp_, sizeof(std::int32_t) *
                                                     static_cast<std::size_t>(rows), nullptr),
                                    "prewarm recovery rows");
            }
            if (cols > 0) {
                throw_if_cuda_error(cudaMallocAsync(&col_temp_, sizeof(std::int32_t) *
                                                     static_cast<std::size_t>(cols), nullptr),
                                    "prewarm recovery cols");
            }
            if (row_temp_) {
                throw_if_cuda_error(cudaFreeAsync(row_temp_, nullptr), "release prewarm rows");
                row_temp_ = nullptr;
            }
            if (col_temp_) {
                throw_if_cuda_error(cudaFreeAsync(col_temp_, nullptr), "release prewarm cols");
                col_temp_ = nullptr;
            }
        } catch (...) {
            if (row_temp_) cudaFreeAsync(row_temp_, nullptr);
            if (col_temp_) cudaFreeAsync(col_temp_, nullptr);
            cudaDeviceSynchronize();
            cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &old_threshold_);
            pool_ = nullptr;
            throw;
        }
    }
    ~PreparedRecoveryAsyncPool() {
        if (pool_) {
            cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &old_threshold_);
            cudaMemPoolTrimTo(pool_, 0);
        }
    }
    PreparedRecoveryAsyncPool(const PreparedRecoveryAsyncPool &) = delete;
    PreparedRecoveryAsyncPool &operator=(const PreparedRecoveryAsyncPool &) = delete;

private:
    cudaMemPool_t pool_ = nullptr;
    std::uint64_t old_threshold_ = 0;
    void *row_temp_ = nullptr;
    void *col_temp_ = nullptr;
};

void run_timed_gpu_presolve(GpuPresolverHandle *handle,
                            const gp::PresolveParams &params) {
    std::cout << "GPU-Preprocessor configuration: folding=" << (params.enable_folding ? "true" : "false")
              << ", presolve=" << (params.enable_presolve ? "true" : "false")
              << ", new_rules=" << (params.enable_structure_specific_rules ? "true" : "false")
              << std::endl;
    auto &pipeline = handle->folding_summary;
    pipeline.presolve_enabled = params.enable_presolve;
    pipeline.original_rows = pipeline.folded_rows = handle->original_lp.A.rows;
    pipeline.original_cols = pipeline.folded_cols = handle->original_lp.A.cols;

    const bool preprocessing_enabled = params.enable_folding || params.enable_presolve;
    std::chrono::steady_clock::time_point preprocessing_start{};
    handle->preprocessing_seconds = 0.0;
    if (preprocessing_enabled) {
        // Complete the model upload before timing the preprocessing pipeline.
        throw_if_cuda_error(cudaDeviceSynchronize(), "before preprocessing timer");
        preprocessing_start = std::chrono::steady_clock::now();
    }

    // Retain the folded model for recovery, but keep folding outside the
    // presolve interval (including when folding finds no aggregation).
    if (params.enable_folding) {
        pipeline.folding = gf::run_folding(handle->original_lp,
                                           params.folding_tolerance, params.verbose);
        pipeline.folding_applied = pipeline.folding.applied;
        if (pipeline.folding_applied) {
            pipeline.folded_rows = pipeline.folding.folded_rows;
            pipeline.folded_cols = pipeline.folding.folded_cols;
        }
    }
    const gp::LPInfoGpu &input = pipeline.folding_applied
        ? pipeline.folding.folded_lp : handle->original_lp;
    handle->presolve_seconds = 0.0;
    handle->presolve_core_seconds = 0.0;
    // Prepare scratch before the library's core timer and report workspace
    // setup/release separately from the exported presolve time.
    if (params.enable_presolve) {
        const auto setup_start = std::chrono::steady_clock::now();
        auto workspace = std::make_unique<gp::PreparedCleanupWorkspace>(
            static_cast<std::size_t>(input.A.rows),
            static_cast<std::size_t>(input.A.cols));
        if (workspace->enabled() && input.A.cols > 0) {
            const std::size_t reduction_bytes = gp::cleanup_reduction_temp_bytes(
                input.c, workspace->reduction_result(),
                input.A.cols);
            workspace->reserve_reduction_scratch(reduction_bytes);
            workspace->prepare_selection(input.c, input.A.cols);
        }
        auto recovery_pool = std::make_unique<PreparedRecoveryAsyncPool>(
            input.A.rows, input.A.cols);
        auto bound_strengthening_workspace =
            std::make_unique<gp::PreparedBoundStrengtheningWorkspace>(
                input.A.rows, input.A.cols,
                params.enable_presolve &&
                    params.enable_bound_strengthening);
        const bool parallel_column_preverify = gpu_presolver::runtime::options().parallel_column_preverify;
        const std::size_t parallel_rows_bytes = params.enable_parallel_rows
            ? gp::parallel_rows_scratch_bytes(input.A.rows) : 0;
        const std::size_t parallel_columns_bytes = params.enable_parallel_columns
            ? gp::parallel_columns_scratch_bytes(
                  input.A.rows, input.A.cols,
                  parallel_column_preverify) : 0;
        auto duplicate_rule_workspace =
            std::make_unique<gp::PreparedDuplicateRuleWorkspace>(
                std::max(parallel_rows_bytes, parallel_columns_bytes));
        throw_if_cuda_error(cudaDeviceSynchronize(), "before presolve timer");
        handle->presolve_setup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_start).count();
        pipeline.presolve = gf::run_optional_gpu_presolve(input, params, true);
        throw_if_cuda_error(cudaDeviceSynchronize(), "after presolve timer");
        // Presolve is the native core interval; the complete pipeline wall
        // interval is measured independently as preprocessing time.
        handle->presolve_seconds = pipeline.presolve.elapsed_seconds;
        handle->presolve_core_seconds = pipeline.presolve.elapsed_seconds;
        const auto release_start = std::chrono::steady_clock::now();
        duplicate_rule_workspace.reset();
        bound_strengthening_workspace.reset();
        workspace.reset();
        recovery_pool.reset();
        handle->presolve_workspace_release_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - release_start).count();
    } else {
        pipeline.presolve = gf::run_optional_gpu_presolve(input, params, true);
    }
    if (preprocessing_enabled) {
        // Include folding, presolve workspace setup/release and presolve itself.
        throw_if_cuda_error(cudaDeviceSynchronize(), "after preprocessing timer");
        handle->preprocessing_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - preprocessing_start).count();
    }
}

template <typename T>
T *copy_vector_to_device(const T *host, size_t count, const char *context) {
    if (count == 0) {
        return nullptr;
    }
    T *device = nullptr;
    throw_if_cuda_error(cudaMalloc(reinterpret_cast<void **>(&device), sizeof(T) * count), context);
    throw_if_cuda_error(cudaMemcpy(device, host, sizeof(T) * count, cudaMemcpyHostToDevice), context);
    return device;
}

std::int32_t *copy_int_vector_to_i32_device(const int *host, size_t count, const char *context) {
    if (count == 0) {
        return nullptr;
    }
    std::vector<std::int32_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<std::int32_t>(host[i]);
    }
    return copy_vector_to_device(values.data(), values.size(), context);
}

void free_device_csr(gp::DeviceCsrMatrix *matrix) {
    if (!matrix) {
        return;
    }
    cudaFree(matrix->rowPtr);
    cudaFree(matrix->colVal);
    cudaFree(matrix->nzVal);
    *matrix = gp::DeviceCsrMatrix{};
}

void free_device_lp(gp::LPInfoGpu *lp) {
    if (!lp) {
        return;
    }
    free_device_csr(&lp->A);
    free_device_csr(&lp->AT);
    cudaFree(lp->c);
    cudaFree(lp->AL);
    cudaFree(lp->AU);
    cudaFree(lp->l);
    cudaFree(lp->u);
    *lp = gp::LPInfoGpu{};
}

void free_gpu_presolver_handle_resources(GpuPresolverHandle *handle) {
    if (!handle) {
        return;
    }
    // Recovery records own their device storage through RAII.
    gp::free_gpu_presolve_reduced_lp(handle->folding_summary.presolve);
    gf::free_folded_lp(handle->folding_summary.folding);
    handle->folding_summary = gf::FoldingPipelineSummary{};
    free_device_lp(&handle->original_lp);
    handle->reduced_m = 0;
    handle->reduced_n = 0;
}

gp::DeviceCsrMatrix upload_csr(const sparseMatrix *matrix, const char *name) {
    gp::DeviceCsrMatrix out;
    out.rows = static_cast<std::int32_t>(matrix->row);
    out.cols = static_cast<std::int32_t>(matrix->col);
    out.nnz = static_cast<std::int32_t>(matrix->numElements);
    out.rowPtr = copy_int_vector_to_i32_device(
        matrix->rowPtr, static_cast<size_t>(matrix->row) + 1, name);
    out.colVal = copy_int_vector_to_i32_device(
        matrix->colIndex, static_cast<size_t>(matrix->numElements), name);
    out.nzVal = copy_vector_to_device(
        matrix->value, static_cast<size_t>(matrix->numElements), name);
    return out;
}

void upload_hprlp_model(const LP_info_cpu *model, gp::LPInfoGpu *out) {
    out->A = upload_csr(model->A, "cudaMalloc/copy GPU presolver A");

    sparseMatrix at_host;
    CSR_transpose_host(*(model->A), &at_host);
    out->AT = upload_csr(&at_host, "cudaMalloc/copy GPU presolver AT");
    std::free(at_host.value);
    std::free(at_host.colIndex);
    std::free(at_host.rowPtr);

    out->c = copy_vector_to_device(model->c, static_cast<size_t>(model->n), "cudaMalloc/copy GPU presolver c");
    out->AL = copy_vector_to_device(model->AL, static_cast<size_t>(model->m), "cudaMalloc/copy GPU presolver AL");
    out->AU = copy_vector_to_device(model->AU, static_cast<size_t>(model->m), "cudaMalloc/copy GPU presolver AU");
    out->l = copy_vector_to_device(model->l, static_cast<size_t>(model->n), "cudaMalloc/copy GPU presolver l");
    out->u = copy_vector_to_device(model->u, static_cast<size_t>(model->n), "cudaMalloc/copy GPU presolver u");
    out->obj_constant = model->obj_constant;
}

template <typename T>
std::vector<T> copy_device_vector_to_host(const T *device, size_t count, const char *context) {
    std::vector<T> values(count);
    if (count > 0) {
        throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * count, cudaMemcpyDeviceToHost),
                            context);
    }
    return values;
}

std::vector<int> copy_device_i32_vector_to_int_host(const std::int32_t *device,
                                                    size_t count,
                                                    const char *context) {
    const std::vector<std::int32_t> raw = copy_device_vector_to_host(device, count, context);
    std::vector<int> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<int>(raw[i]);
    }
    return values;
}

bool copy_reduced_lp_to_hprlp(const gp::LPInfoGpu &lp,
                              double obj_constant,
                              LP_info_cpu *reduced_model) {
    if (lp.A.rows <= 0 || lp.A.cols <= 0 || lp.A.nnz <= 0) {
        std::cerr << "[warn] GPU-Preprocessor returned an empty reduced LP; preprocessing failed" << std::endl;
        return false;
    }

    const std::vector<int> row_ptr = copy_device_i32_vector_to_int_host(
        lp.A.rowPtr, static_cast<size_t>(lp.A.rows) + 1, "cudaMemcpy reduced rowPtr");
    const std::vector<int> col_idx = copy_device_i32_vector_to_int_host(
        lp.A.colVal, static_cast<size_t>(lp.A.nnz), "cudaMemcpy reduced colVal");
    const std::vector<HPRLP_FLOAT> values = copy_device_vector_to_host(
        lp.A.nzVal, static_cast<size_t>(lp.A.nnz), "cudaMemcpy reduced values");
    const std::vector<HPRLP_FLOAT> al = copy_device_vector_to_host(
        lp.AL, static_cast<size_t>(lp.A.rows), "cudaMemcpy reduced AL");
    const std::vector<HPRLP_FLOAT> au = copy_device_vector_to_host(
        lp.AU, static_cast<size_t>(lp.A.rows), "cudaMemcpy reduced AU");
    const std::vector<HPRLP_FLOAT> lower = copy_device_vector_to_host(
        lp.l, static_cast<size_t>(lp.A.cols), "cudaMemcpy reduced l");
    const std::vector<HPRLP_FLOAT> upper = copy_device_vector_to_host(
        lp.u, static_cast<size_t>(lp.A.cols), "cudaMemcpy reduced u");
    const std::vector<HPRLP_FLOAT> c = copy_device_vector_to_host(
        lp.c, static_cast<size_t>(lp.A.cols), "cudaMemcpy reduced c");

    const CSRMatrix csr{
        static_cast<int>(lp.A.rows),
        static_cast<int>(lp.A.cols),
        static_cast<int>(lp.A.nnz),
        const_cast<int*>(row_ptr.data()),
        const_cast<int*>(col_idx.data()),
        const_cast<HPRLP_FLOAT*>(values.data()),
    };

    initialize_empty_model(reduced_model);
    build_model_from_arrays(&csr,
                            al.data(),
                            au.data(),
                            lower.data(),
                            upper.data(),
                            c.data(),
                            obj_constant,
                            reduced_model);
    return reduced_model->A && reduced_model->m > 0 && reduced_model->n > 0;
}

int *copy_device_i32_to_int_device(const std::int32_t *device, size_t count, const char *context) {
    if (count == 0) {
        return nullptr;
    }
    int *out = nullptr;
    throw_if_cuda_error(
        hprlp_device_malloc_compressible(&out, sizeof(int) * count),
        context);
    throw_if_cuda_error(cudaMemcpy(out, device, sizeof(int) * count, cudaMemcpyDeviceToDevice), context);
    return out;
}

HPRLP_FLOAT *copy_device_double_to_hprlp_device(const double *device, size_t count, const char *context) {
    if (count == 0) {
        return nullptr;
    }
    HPRLP_FLOAT *out = nullptr;
    throw_if_cuda_error(
        hprlp_device_malloc_compressible(
            &out, sizeof(HPRLP_FLOAT) * count),
        context);
    throw_if_cuda_error(cudaMemcpy(out, device, sizeof(HPRLP_FLOAT) * count, cudaMemcpyDeviceToDevice), context);
    return out;
}

sparseMatrix *copy_device_csr_to_hprlp_device(const gp::DeviceCsrMatrix &matrix,
                                              const char *context) {
    sparseMatrix *out = new sparseMatrix{};
    out->row = static_cast<int>(matrix.rows);
    out->col = static_cast<int>(matrix.cols);
    out->numElements = static_cast<int>(matrix.nnz);
    try {
        out->rowPtr = copy_device_i32_to_int_device(
            matrix.rowPtr, static_cast<size_t>(matrix.rows) + 1, context);
        out->colIndex = copy_device_i32_to_int_device(
            matrix.colVal, static_cast<size_t>(matrix.nnz), context);
        out->value = copy_device_double_to_hprlp_device(
            matrix.nzVal, static_cast<size_t>(matrix.nnz), context);
    } catch (...) {
        hprlp_device_free(out->rowPtr);
        hprlp_device_free(out->colIndex);
        hprlp_device_free(out->value);
        delete out;
        throw;
    }
    return out;
}

bool copy_reduced_lp_to_hprlp_device(const gp::LPInfoGpu &lp,
                                     double obj_constant,
                                     LP_info_gpu *reduced_model) {
    if (lp.A.rows <= 0 || lp.A.cols <= 0 || lp.A.nnz <= 0) {
        std::cerr << "[warn] GPU-Preprocessor returned an empty reduced LP; preprocessing failed" << std::endl;
        return false;
    }

    initialize_empty_device_model(reduced_model);
    try {
        reduced_model->m = static_cast<int>(lp.A.rows);
        reduced_model->n = static_cast<int>(lp.A.cols);
        reduced_model->obj_constant = static_cast<HPRLP_FLOAT>(obj_constant);
        reduced_model->A = copy_device_csr_to_hprlp_device(lp.A, "cudaMemcpy D2D reduced A");
        if (!build_stable_device_transpose(
                reduced_model->A, &reduced_model->AT)) {
            throw std::runtime_error(
                "failed to build stable GPU transpose of reduced A");
        }
        reduced_model->AL = copy_device_double_to_hprlp_device(
            lp.AL, static_cast<size_t>(lp.A.rows), "cudaMemcpy D2D reduced AL");
        reduced_model->AU = copy_device_double_to_hprlp_device(
            lp.AU, static_cast<size_t>(lp.A.rows), "cudaMemcpy D2D reduced AU");
        reduced_model->l = copy_device_double_to_hprlp_device(
            lp.l, static_cast<size_t>(lp.A.cols), "cudaMemcpy D2D reduced l");
        reduced_model->u = copy_device_double_to_hprlp_device(
            lp.u, static_cast<size_t>(lp.A.cols), "cudaMemcpy D2D reduced u");
        reduced_model->c = copy_device_double_to_hprlp_device(
            lp.c, static_cast<size_t>(lp.A.cols), "cudaMemcpy D2D reduced c");
    } catch (...) {
        free_lp_info(reduced_model);
        initialize_empty_device_model(reduced_model);
        throw;
    }
    return reduced_model->A && reduced_model->AT && reduced_model->m > 0 && reduced_model->n > 0;
}

void release_result_vectors(HPRLP_results *result) {
    if (!result) {
        return;
    }
    std::free(result->x);
    std::free(result->y);
    std::free(result->z);
    result->x = nullptr;
    result->y = nullptr;
    result->z = nullptr;
}

void csr_matvec(const sparseMatrix *A, const HPRLP_FLOAT *x, HPRLP_FLOAT *out) {
    for (int row = 0; row < A->row; ++row) {
        HPRLP_FLOAT sum = 0.0;
        for (int idx = A->rowPtr[row]; idx < A->rowPtr[row + 1]; ++idx) {
            sum += A->value[idx] * x[A->colIndex[idx]];
        }
        out[row] = sum;
    }
}

void csr_transpose_matvec(const sparseMatrix *A, const HPRLP_FLOAT *y, HPRLP_FLOAT *out) {
    std::fill(out, out + A->col, 0.0);
    for (int row = 0; row < A->row; ++row) {
        for (int idx = A->rowPtr[row]; idx < A->rowPtr[row + 1]; ++idx) {
            out[A->colIndex[idx]] += A->value[idx] * y[row];
        }
    }
}

void project_row_duals(const LP_info_cpu *model, std::vector<HPRLP_FLOAT> *y_proj) {
    for (int i = 0; i < model->m; ++i) {
        const bool lower_inf = std::isinf(model->AL[i]) && model->AL[i] < 0.0;
        const bool upper_inf = std::isinf(model->AU[i]) && model->AU[i] > 0.0;
        if (lower_inf && upper_inf) {
            (*y_proj)[i] = 0.0;
        } else if (upper_inf) {
            (*y_proj)[i] = std::max((*y_proj)[i], static_cast<HPRLP_FLOAT>(0.0));
        } else if (lower_inf) {
            (*y_proj)[i] = std::min((*y_proj)[i], static_cast<HPRLP_FLOAT>(0.0));
        }
    }
}

void project_bound_duals(const LP_info_cpu *model, std::vector<HPRLP_FLOAT> *z_proj) {
    for (int j = 0; j < model->n; ++j) {
        const bool lower_inf = std::isinf(model->l[j]) && model->l[j] < 0.0;
        const bool upper_inf = std::isinf(model->u[j]) && model->u[j] > 0.0;
        if (lower_inf && upper_inf) {
            (*z_proj)[j] = 0.0;
        } else if (upper_inf) {
            (*z_proj)[j] = std::max((*z_proj)[j], static_cast<HPRLP_FLOAT>(0.0));
        } else if (lower_inf) {
            (*z_proj)[j] = std::min((*z_proj)[j], static_cast<HPRLP_FLOAT>(0.0));
        }
    }
}

HPRLP_FLOAT squared_norm_of_conceptual_rhs(const HPRLP_FLOAT *lower,
                                           const HPRLP_FLOAT *upper,
                                           int len) {
    HPRLP_FLOAT sum_sq = 0.0;
    for (int i = 0; i < len; ++i) {
        const HPRLP_FLOAT lower_val = std::isfinite(lower[i]) ? std::abs(lower[i]) : 0.0;
        const HPRLP_FLOAT upper_val = std::isfinite(upper[i]) ? std::abs(upper[i]) : 0.0;
        const HPRLP_FLOAT rhs_val = std::max(lower_val, upper_val);
        sum_sq += rhs_val * rhs_val;
    }
    return sum_sq;
}

OriginalKktMetrics compute_original_kkt_metrics(const LP_info_cpu *model,
                                                const HPRLP_FLOAT *x,
                                                const HPRLP_FLOAT *y,
                                                const HPRLP_FLOAT *z) {
    OriginalKktMetrics metrics;
    std::vector<HPRLP_FLOAT> y_proj(y, y + model->m);
    std::vector<HPRLP_FLOAT> z_proj(z, z + model->n);
    project_row_duals(model, &y_proj);
    project_bound_duals(model, &z_proj);

    std::vector<HPRLP_FLOAT> Ax(model->m, 0.0);
    std::vector<HPRLP_FLOAT> ATy(model->n, 0.0);
    csr_matvec(model->A, x, Ax.data());
    csr_transpose_matvec(model->A, y_proj.data(), ATy.data());

    HPRLP_FLOAT norm_c_sq = 0.0;
    for (int j = 0; j < model->n; ++j) {
        norm_c_sq += model->c[j] * model->c[j];
    }
    const HPRLP_FLOAT norm_b = 1.0 + std::sqrt(
        squared_norm_of_conceptual_rhs(model->AL, model->AU, model->m));
    const HPRLP_FLOAT norm_c = 1.0 + std::sqrt(norm_c_sq);

    HPRLP_FLOAT err_Ax_sq = 0.0;
    for (int i = 0; i < model->m; ++i) {
        HPRLP_FLOAT violation = 0.0;
        if (std::isfinite(model->AL[i]) && Ax[i] < model->AL[i]) {
            violation = std::max(violation, model->AL[i] - Ax[i]);
        }
        if (std::isfinite(model->AU[i]) && Ax[i] > model->AU[i]) {
            violation = std::max(violation, Ax[i] - model->AU[i]);
        }
        err_Ax_sq += violation * violation;
    }

    HPRLP_FLOAT err_x_sq = 0.0;
    for (int j = 0; j < model->n; ++j) {
        HPRLP_FLOAT violation = 0.0;
        if (std::isfinite(model->l[j]) && x[j] < model->l[j]) {
            violation = std::max(violation, model->l[j] - x[j]);
        }
        if (std::isfinite(model->u[j]) && x[j] > model->u[j]) {
            violation = std::max(violation, x[j] - model->u[j]);
        }
        err_x_sq += violation * violation;
    }
    metrics.primal_feas = std::max(std::sqrt(err_Ax_sq), std::sqrt(err_x_sq)) / norm_b;

    HPRLP_FLOAT dual_residual_sq = 0.0;
    for (int j = 0; j < model->n; ++j) {
        const HPRLP_FLOAT dual_residual = model->c[j] - ATy[j] - z_proj[j];
        dual_residual_sq += dual_residual * dual_residual;
    }
    metrics.dual_feas = std::sqrt(dual_residual_sq) / norm_c;

    HPRLP_FLOAT p_lin = 0.0;
    for (int j = 0; j < model->n; ++j) {
        p_lin += model->c[j] * x[j];
    }

    HPRLP_FLOAT d_lin = 0.0;
    for (int i = 0; i < model->m; ++i) {
        const HPRLP_FLOAT support = y_proj[i] >= 0.0 ?
            (std::isfinite(model->AL[i]) ? model->AL[i] : 0.0) :
            (std::isfinite(model->AU[i]) ? model->AU[i] : 0.0);
        d_lin += y_proj[i] * support;
    }
    for (int j = 0; j < model->n; ++j) {
        const HPRLP_FLOAT support = z_proj[j] >= 0.0 ?
            (std::isfinite(model->l[j]) ? model->l[j] : 0.0) :
            (std::isfinite(model->u[j]) ? model->u[j] : 0.0);
        d_lin += z_proj[j] * support;
    }

    metrics.gap = std::abs(d_lin - p_lin) / (1.0 + std::abs(d_lin) + std::abs(p_lin));
    metrics.primal_obj = p_lin + model->obj_constant;
    metrics.dual_obj = d_lin + model->obj_constant;
    return metrics;
}

void print_postsolve_kkt_validation(const OriginalKktMetrics &metrics,
                                    const IntegrationParameters *param) {
    const HPRLP_FLOAT stop_tol = param ? param->stop_tol : 1e-4;
    const HPRLP_FLOAT original_kkt_error = std::max(metrics.primal_feas,
                                                    std::max(metrics.dual_feas, metrics.gap));
    if (original_kkt_error <= stop_tol) {
        std::cout << "Postsolve original KKT check passed" << std::endl;
        return;
    }

    std::cout << "Warning: postsolve original KKT check failed"
              << " (reduced-LP solver status is retained; inspect the residuals below)" << std::endl;
    const std::streamsize saved_precision = std::cout.precision();
    const std::ios::fmtflags saved_flags = std::cout.flags();
    std::cout << std::scientific << std::setprecision(2)
              << "Stop Tolerance: " << stop_tol << std::endl
              << std::fixed << std::setprecision(8)
              << "Primal Objective: " << metrics.primal_obj << std::endl
              << "Dual Objective: " << metrics.dual_obj << std::endl
              << std::scientific << std::setprecision(2)
              << "Primal Residual: " << metrics.primal_feas << std::endl
              << "Dual Residual: " << metrics.dual_feas << std::endl
              << "Relative Gap: " << metrics.gap << std::endl;
    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
}

} // namespace

bool ijoc_run_gpu_presolve(const LP_info_cpu *model,
                               const IntegrationParameters *param,
                               LP_info_cpu *reduced_model,
                               void **presolver_handle_out,
                               HPRLP_FLOAT *presolve_time_out,
                               HPRLP_FLOAT *folding_time_out,
                               HPRLP_FLOAT *presolve_core_time_out,
                               HPRLP_FLOAT *preprocessing_time_out) {
    if (!model || !model->A || !reduced_model || !presolver_handle_out) {
        return false;
    }

    initialize_empty_model(reduced_model);
    *presolver_handle_out = nullptr;
    if (presolve_time_out) {
        *presolve_time_out = 0.0;
    }
    if (presolve_core_time_out) {
        *presolve_core_time_out = 0.0;
    }
    if (preprocessing_time_out) {
        *preprocessing_time_out = 0.0;
    }
    if (folding_time_out) {
        *folding_time_out = 0.0;
    }

    std::unique_ptr<GpuPresolverHandle> handle(new (std::nothrow) GpuPresolverHandle);
    if (!handle) {
        std::cerr << "[warn] Failed to allocate GPU-Preprocessor handle; preprocessing failed" << std::endl;
        return false;
    }

    try {
        if (param) {
            throw_if_cuda_error(cudaSetDevice(param->device_number), "cudaSetDevice for GPU-Preprocessor");
        }
        upload_hprlp_model(model, &handle->original_lp);

        gp::PresolveParams presolve_params;
        presolve_params.verbose = false;
        presolve_params.enable_folding = param ? param->enable_gpu_folding : true;
        presolve_params.enable_presolve = param ? param->enable_gpu_presolve : true;
        presolve_params.enable_structure_specific_rules = param ? param->enable_gpu_new_rules : true;
        if (param && std::isfinite(param->time_limit) && param->time_limit > 0.0) {
            presolve_params.max_time = param->time_limit;
        }

        std::cout << "Doing preprocessing (GPU-Preprocessor"
                  << (presolve_params.enable_folding ? " folding" : "")
                  << ")..." << std::endl;
        run_timed_gpu_presolve(handle.get(), presolve_params);
        gp::GpuPresolveSummary &summary = handle->folding_summary.presolve;
        if (presolve_time_out) {
            *presolve_time_out = static_cast<HPRLP_FLOAT>(handle->presolve_seconds);
        }
        if (presolve_core_time_out) {
            *presolve_core_time_out = static_cast<HPRLP_FLOAT>(handle->presolve_core_seconds);
        }
        if (preprocessing_time_out) {
            *preprocessing_time_out = static_cast<HPRLP_FLOAT>(handle->preprocessing_seconds);
        }
        if (folding_time_out) {
            *folding_time_out = static_cast<HPRLP_FLOAT>(
                handle->folding_summary.folding.profile.total_seconds);
        }
        std::cout << "GPU-Preprocessor folding time: "
                  << handle->folding_summary.folding.profile.total_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor folding: ("
                  << handle->folding_summary.original_rows << ", "
                  << handle->folding_summary.original_cols << ") -> ("
                  << handle->folding_summary.folded_rows << ", "
                  << handle->folding_summary.folded_cols << ")" << std::endl;
        std::cout << std::endl;
        std::cout << "GPU-Preprocessor presolve time: "
                  << std::setprecision(8) << handle->presolve_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor preprocessing time: "
                  << handle->preprocessing_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor presolve workspace setup time: "
                  << std::setprecision(8) << handle->presolve_setup_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor presolve workspace release time: "
                  << handle->presolve_workspace_release_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor presolve: ("
                  << handle->folding_summary.folded_rows << ", "
                  << handle->folding_summary.folded_cols << ") -> ("
                  << summary.reduced_rows << ", "
                  << summary.reduced_cols << ")" << std::endl;
        std::cout << std::endl;

        if (summary.has_infeasible || summary.has_unbounded) {
            std::cerr << "[warn] GPU-Preprocessor reported "
                      << (summary.has_infeasible ? "infeasible" : "unbounded")
                      << "; preprocessing failed" << std::endl;
            free_gpu_presolver_handle_resources(handle.get());
            return false;
        }

        std::cout << "GPU-Preprocessor reduced model:" << std::endl;
        if (!copy_reduced_lp_to_hprlp(summary.reduced_lp,
                                      summary.record.obj_constant_new,
                                      reduced_model)) {
            free_gpu_presolver_handle_resources(handle.get());
            return false;
        }

        handle->reduced_m = reduced_model->m;
        handle->reduced_n = reduced_model->n;
        gp::free_gpu_presolve_reduced_lp(summary);

        GpuPresolverHandle *released_handle = handle.release();
        *presolver_handle_out = released_handle;
        return true;
    } catch (const std::exception &err) {
        std::cerr << "[warn] GPU-Preprocessor failed: " << err.what()
                  << "; preprocessing failed" << std::endl;
        free_lp_info_cpu(reduced_model);
        initialize_empty_model(reduced_model);
        free_gpu_presolver_handle_resources(handle.get());
        // Clear a pending allocation error after releasing presolver storage
        // so subsequent CUDA operations do not inherit it.
        const cudaError_t recovery_status = cudaDeviceSynchronize();
        if (recovery_status != cudaSuccess &&
            recovery_status != cudaErrorMemoryAllocation) {
            std::cerr << "[warn] GPU-Preprocessor CUDA recovery: "
                      << cudaGetErrorString(recovery_status) << std::endl;
        }
        (void)cudaGetLastError();
        return false;
    }
}

bool ijoc_run_gpu_presolve_device(const LP_info_cpu *model,
                                      const IntegrationParameters *param,
                                      LP_info_gpu *reduced_model,
                                      void **presolver_handle_out,
                                      HPRLP_FLOAT *presolve_time_out,
                                      HPRLP_FLOAT *folding_time_out,
                                      std::chrono::steady_clock::time_point
                                          *first_device_ready_out,
                                      HPRLP_FLOAT *presolve_core_time_out,
                                      HPRLP_FLOAT *preprocessing_time_out) {
    if (!model || !model->A || !reduced_model || !presolver_handle_out) {
        return false;
    }

    initialize_empty_device_model(reduced_model);
    *presolver_handle_out = nullptr;
    if (presolve_time_out) {
        *presolve_time_out = 0.0;
    }
    if (presolve_core_time_out) {
        *presolve_core_time_out = 0.0;
    }
    if (folding_time_out) {
        *folding_time_out = 0.0;
    }
    if (first_device_ready_out) {
        *first_device_ready_out = {};
    }
    if (preprocessing_time_out) {
        *preprocessing_time_out = 0.0;
    }

    std::unique_ptr<GpuPresolverHandle> handle(new (std::nothrow) GpuPresolverHandle);
    if (!handle) {
        std::cerr << "[warn] Failed to allocate GPU-Preprocessor handle; preprocessing failed" << std::endl;
        return false;
    }

    try {
        if (param) {
            throw_if_cuda_error(cudaSetDevice(param->device_number), "cudaSetDevice for GPU-Preprocessor");
        }
        upload_hprlp_model(model, &handle->original_lp);
        // Expose when the original model and its transpose are fully resident
        // on the selected GPU, before preprocessing begins.
        if (first_device_ready_out) {
            *first_device_ready_out = std::chrono::steady_clock::now();
        }

        gp::PresolveParams presolve_params;
        presolve_params.verbose = false;
        presolve_params.enable_folding = param ? param->enable_gpu_folding : true;
        presolve_params.enable_presolve = param ? param->enable_gpu_presolve : true;
        presolve_params.enable_structure_specific_rules = param ? param->enable_gpu_new_rules : true;
        if (param && std::isfinite(param->time_limit) && param->time_limit > 0.0) {
            presolve_params.max_time = param->time_limit;
        }

        std::cout << "Doing preprocessing (GPU-Preprocessor"
                  << (presolve_params.enable_folding ? " folding" : "")
                  << ")..." << std::endl;
        run_timed_gpu_presolve(handle.get(), presolve_params);
        gp::GpuPresolveSummary &summary = handle->folding_summary.presolve;
        if (presolve_time_out) {
            *presolve_time_out = static_cast<HPRLP_FLOAT>(handle->presolve_seconds);
        }
        if (presolve_core_time_out) {
            *presolve_core_time_out = static_cast<HPRLP_FLOAT>(handle->presolve_core_seconds);
        }
        if (preprocessing_time_out) {
            *preprocessing_time_out = static_cast<HPRLP_FLOAT>(handle->preprocessing_seconds);
        }
        if (folding_time_out) {
            *folding_time_out = static_cast<HPRLP_FLOAT>(
                handle->folding_summary.folding.profile.total_seconds);
        }
        std::cout << "GPU-Preprocessor folding time: "
                  << handle->folding_summary.folding.profile.total_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor folding: ("
                  << handle->folding_summary.original_rows << ", "
                  << handle->folding_summary.original_cols << ") -> ("
                  << handle->folding_summary.folded_rows << ", "
                  << handle->folding_summary.folded_cols << ")" << std::endl;
        std::cout << std::endl;
        std::cout << "GPU-Preprocessor presolve time: "
                  << std::setprecision(8) << handle->presolve_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor preprocessing time: "
                  << handle->preprocessing_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor presolve workspace setup time: "
                  << std::setprecision(8) << handle->presolve_setup_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor presolve workspace release time: "
                  << handle->presolve_workspace_release_seconds << " seconds" << std::endl;
        std::cout << "GPU-Preprocessor presolve: ("
                  << handle->folding_summary.folded_rows << ", "
                  << handle->folding_summary.folded_cols << ") -> ("
                  << summary.reduced_rows << ", "
                  << summary.reduced_cols << ")" << std::endl;
        std::cout << std::endl;

        if (summary.has_infeasible || summary.has_unbounded) {
            std::cerr << "[warn] GPU-Preprocessor reported "
                      << (summary.has_infeasible ? "infeasible" : "unbounded")
                      << "; preprocessing failed" << std::endl;
            free_gpu_presolver_handle_resources(handle.get());
            return false;
        }

        if (!copy_reduced_lp_to_hprlp_device(summary.reduced_lp,
                                             summary.record.obj_constant_new,
                                             reduced_model)) {
            free_gpu_presolver_handle_resources(handle.get());
            return false;
        }

        const bool dictionary_ready =
            prepare_device_packed_dictionary_metadata(reduced_model);
        std::cout << "GPU-resident packed-dictionary metadata: "
                  << (dictionary_ready ? "active" : "unavailable")
                  << std::endl;
        const bool operator_metadata_ready =
            prepare_device_operator_metadata(reduced_model);
        std::cout << "GPU-resident unit/state operator metadata: "
                  << (operator_metadata_ready ? "active" : "unavailable")
                  << std::endl;

        handle->reduced_m = reduced_model->m;
        handle->reduced_n = reduced_model->n;
        std::cout << "GPU-Preprocessor reduced model:" << std::endl;
        std::cout << "problem information: nRow = " << summary.reduced_rows
                  << ", nCol = " << summary.reduced_cols
                  << ", nnz A = " << summary.reduced_nnz << std::endl;
        std::cout << std::endl;
        gp::free_gpu_presolve_reduced_lp(summary);

        *presolver_handle_out = handle.release();
        return true;
    } catch (const std::exception &err) {
        std::cerr << "[warn] GPU-Preprocessor failed: " << err.what()
                  << "; preprocessing failed" << std::endl;
        free_lp_info(reduced_model);
        initialize_empty_device_model(reduced_model);
        free_gpu_presolver_handle_resources(handle.get());
        // Clear a pending allocation error after releasing presolver storage
        // so subsequent CUDA operations do not inherit it.
        const cudaError_t recovery_status = cudaDeviceSynchronize();
        if (recovery_status != cudaSuccess &&
            recovery_status != cudaErrorMemoryAllocation) {
            std::cerr << "[warn] GPU-Preprocessor CUDA recovery: "
                      << cudaGetErrorString(recovery_status) << std::endl;
        }
        (void)cudaGetLastError();
        return false;
    }
}

bool ijoc_apply_gpu_postsolve(HPRLP_results *result,
                                  void *presolver_handle,
                                  int original_m,
                                  int original_n) {
    if (!result || !presolver_handle || !result->x || !result->y || !result->z) {
        return false;
    }

    GpuPresolverHandle *handle = static_cast<GpuPresolverHandle*>(presolver_handle);
    if (handle->folding_summary.original_rows != original_m ||
        handle->folding_summary.original_cols != original_n ||
        handle->reduced_m <= 0 || handle->reduced_n <= 0) {
        return false;
    }

    double *x_red = nullptr;
    double *y_red = nullptr;
    double *z_red = nullptr;
    try {
        x_red = copy_vector_to_device(result->x, static_cast<size_t>(handle->reduced_n),
                                      "cudaMalloc/copy reduced x");
        y_red = copy_vector_to_device(result->y, static_cast<size_t>(handle->reduced_m),
                                      "cudaMalloc/copy reduced y");
        z_red = copy_vector_to_device(result->z, static_cast<size_t>(handle->reduced_n),
                                      "cudaMalloc/copy reduced z");
        gf::UnfoldedSolutionHost unfolded =
            gf::postsolve_and_unfold_to_host(
                x_red, y_red, z_red, handle->folding_summary, &handle->original_lp);
        cudaFree(x_red);
        cudaFree(y_red);
        cudaFree(z_red);
        x_red = nullptr;
        y_red = nullptr;
        z_red = nullptr;

        if (unfolded.x.size() != static_cast<size_t>(original_n) ||
            unfolded.y.size() != static_cast<size_t>(original_m) ||
            unfolded.z.size() != static_cast<size_t>(original_n)) {
            return false;
        }

        HPRLP_FLOAT *x_full = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(original_n) * sizeof(HPRLP_FLOAT)));
        HPRLP_FLOAT *y_full = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(original_m) * sizeof(HPRLP_FLOAT)));
        HPRLP_FLOAT *z_full = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(original_n) * sizeof(HPRLP_FLOAT)));
        if ((!x_full && original_n > 0) || (!y_full && original_m > 0) || (!z_full && original_n > 0)) {
            std::free(x_full);
            std::free(y_full);
            std::free(z_full);
            return false;
        }

        std::memcpy(x_full, unfolded.x.data(), static_cast<size_t>(original_n) * sizeof(HPRLP_FLOAT));
        std::memcpy(y_full, unfolded.y.data(), static_cast<size_t>(original_m) * sizeof(HPRLP_FLOAT));
        std::memcpy(z_full, unfolded.z.data(), static_cast<size_t>(original_n) * sizeof(HPRLP_FLOAT));

        release_result_vectors(result);
        result->x = x_full;
        result->y = y_full;
        result->z = z_full;
        return true;
    } catch (const std::exception &err) {
        cudaFree(x_red);
        cudaFree(y_red);
        cudaFree(z_red);
        std::cerr << "[warn] GPU-Preprocessor postsolve failed: " << err.what() << std::endl;
        return false;
    }
}

bool ijoc_gpu_postsolve_and_validate(HPRLP_results *result,
                                             const LP_info_cpu *original_model,
                                             void *presolver_handle,
                                             const IntegrationParameters *param) {
    if (!result || !original_model) {
        return false;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "GPU-PREPROCESSOR POSTSOLVE" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const bool reduced_solution_optimal =
        std::strcmp(result->status, "OPTIMAL") == 0;

    if (!ijoc_apply_gpu_postsolve(result, presolver_handle, original_model->m, original_model->n)) {
        return false;
    }

    if (!reduced_solution_optimal) {
        std::cout << "Skipping postsolve original KKT check since the reduced solution is not optimal"
                  << std::endl;
        return true;
    }

    const OriginalKktMetrics metrics = compute_original_kkt_metrics(
        original_model, result->x, result->y, result->z);
    print_postsolve_kkt_validation(metrics, param);
    return true;
}

void ijoc_free_gpu(void *presolver_handle) {
    if (!presolver_handle) {
        return;
    }
    GpuPresolverHandle *handle = static_cast<GpuPresolverHandle*>(presolver_handle);
    free_gpu_presolver_handle_resources(handle);
    delete handle;
}
