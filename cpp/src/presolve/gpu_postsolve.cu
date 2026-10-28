#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"

#include <cuda_runtime.h>

#include "postsolve/postsolve_bound_strengthening.cuh"
#include "postsolve/postsolve_common.cuh"
#include "postsolve/postsolve_doubleton_equations.cuh"
#include "postsolve/postsolve_fixed_variables.cuh"
#include "postsolve/postsolve_parallel_columns.cuh"
#include "postsolve/postsolve_parallel_rows.cuh"
#include "postsolve/postsolve_piecewise_linear_reformulation_absolute_value.hpp"
#include "postsolve/postsolve_piecewise_linear_reformulation_variable_pair.hpp"
#include "postsolve/postsolve_projected_variables.cuh"
#include "postsolve/postsolve_singleton_columns.cuh"
#include "postsolve/postsolve_singleton_rows.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {

using namespace postsolve;
namespace {

__global__ void _kernel_scatter_reduced_to_original(double* dst,
                                                    const double* src,
                                                    const std::int32_t* red2org,
                                                    std::int32_t n_red) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n_red) {
    dst[red2org[i]] = src[i];
  }
}

__global__ void _kernel_mark_retrieved(std::uint8_t* mask,
                                       const std::int32_t* red2org,
                                       std::int32_t n_red) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n_red) {
    mask[red2org[i]] = std::uint8_t{1};
  }
}

__global__ void _kernel_project_column_duals_from_original(double* x_org,
                                                           double* z_org,
                                                           const std::uint8_t* protected_cols,
                                                           const NnzOffset* AT_rowPtr,
                                                           const std::int32_t* AT_colVal,
                                                           const double* AT_nzVal,
                                                           const double* y_org,
                                                           const double* c,
                                                           const double* l,
                                                           const double* u,
                                                           std::int32_t n,
                                                           double tol) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n || (protected_cols != nullptr && protected_cols[col] != std::uint8_t{0})) {
    return;
  }
  _postsolve_project_single_column_dual_from_original(
      x_org, z_org, AT_rowPtr, AT_colVal, AT_nzVal, y_org, c, l, u, col, tol);
}

__global__ void _kernel_restore_fixed_values(double* x_org,
                                             const std::int32_t* fixed_idx,
                                             const double* fixed_val,
                                             std::int32_t fixed_count) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < fixed_count) {
    x_org[fixed_idx[i]] = fixed_val[i];
  }
}

__global__ void _kernel_replay_postsolve_tape(double* x_org,
                                              double* y_org,
                                              double* z_org,
                                              std::uint8_t* z_retrieved,
                                              const std::int32_t* reduction_types,
                                              const NnzOffset* idx_starts,
                                              const NnzOffset* val_starts,
                                              const std::int32_t* tape_idx,
                                              const double* tape_val,
                                              const NnzOffset* A_rowPtr,
                                              const std::int32_t* A_colVal,
                                              const double* A_nzVal,
                                              const NnzOffset* AT_rowPtr,
                                              const std::int32_t* AT_colVal,
                                              const double* AT_nzVal,
                                              const double* c,
                                              const double* AL,
                                              const double* AU,
                                              const double* l,
                                              const double* u,
                                              NnzOffset record_begin,
                                              NnzOffset record_end,
                                              std::int32_t has_original_model,
                                              std::int32_t replay_primal,
                                              std::int32_t replay_dual,
                                              std::int32_t parallel_row_doc_recovery,
                                              double tol) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (NnzOffset k = record_end - 1; k >= record_begin; --k) {
    const std::int32_t reduction_type = reduction_types[k];
    const NnzOffset idx0 = idx_starts[k];
    const NnzOffset idx1 = idx_starts[k + 1] - 1;
    const NnzOffset val0 = val_starts[k];
    const NnzOffset val1 = val_starts[k + 1] - 1;
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelColumn)) {
      replay_parallel_column(
          x_org, z_org, z_retrieved, tape_idx, tape_val, idx0, val0, replay_primal, replay_dual, tol);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation)) {
      replay_doubleton_equation(
          x_org, y_org, z_org, z_retrieved, tape_idx, tape_val, AT_rowPtr, AT_colVal, AT_nzVal, c,
          idx0, idx1, val0, val1, has_original_model, replay_primal, replay_dual, tol);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::SubCol)) {
      replay_singleton_column(
          x_org, y_org, z_org, z_retrieved, tape_idx, tape_val, idx0, val0, replay_primal,
          replay_dual);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedCol)) {
      replay_fixed_column(
          x_org, y_org, z_org, z_retrieved, tape_idx, tape_val, idx0, idx1, val0, replay_primal,
          replay_dual);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedColInf)) {
      replay_infinite_fixed_column(
          x_org, z_org, z_retrieved, tape_idx, tape_val, idx0, val0, replay_primal, replay_dual);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FmeCol)) {
      replay_projected_column(
          x_org, y_org, z_org, z_retrieved, tape_idx, tape_val, idx0, val0, replay_primal,
          replay_dual);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)) {
      replay_equation_to_inequality(
          y_org, tape_idx, tape_val, idx0, val0);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelRow) &&
               idx1 >= idx0 + 1 && val1 >= val0 + 4) {
      replay_parallel_row(
          x_org, y_org, tape_idx, tape_val, A_rowPtr, A_colVal, A_nzVal, idx0, val0,
          has_original_model, parallel_row_doc_recovery, tol);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DeletedRow) &&
               has_original_model != 0 && idx1 >= idx0 + 1 && val1 >= val0 + 2) {
      replay_deleted_singleton_row(
          x_org, y_org, z_org, z_retrieved, tape_idx, tape_val, AT_rowPtr, AT_colVal, AT_nzVal, c,
          l, u, idx0, val0, tol);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow) &&
               has_original_model != 0 && idx1 >= idx0 + 1) {
      replay_row_bound_change(
          x_org, y_org, z_org, z_retrieved, tape_idx, tape_val, A_rowPtr, A_colVal, A_nzVal, idx0,
          val0, tol);
    }
  }
}

void scatter_reduced_to_original(double* dst,
                                 const double* src,
                                 const std::vector<std::int32_t>& red2org,
                                 std::int32_t original_count,
                                 const char* context) {
  if (original_count > 0) {
    throw_if_cuda_error(cudaMemset(dst, 0, sizeof(double) * static_cast<std::size_t>(original_count)),
                        context);
  }
  if (red2org.empty() || src == nullptr) {
    return;
  }
  std::int32_t* red2org_d = copy_vector_to_device(red2org, context);
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_scatter_reduced_to_original<<<blocks, GPU_PRESOLVE_THREADS>>>(dst, src, red2org_d, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_reduced_to_original");
  cudaFree(red2org_d);
}

void restore_fixed_values(double* x_org,
                          const std::vector<std::int32_t>& fixed_idx,
                          const std::vector<double>& fixed_val) {
  if (fixed_idx.empty()) {
    return;
  }
  std::int32_t* fixed_idx_d = copy_vector_to_device(fixed_idx, "cudaMalloc/copy fixed_idx");
  double* fixed_val_d = copy_vector_to_device(fixed_val, "cudaMalloc/copy fixed_val");
  const std::int32_t fixed_count = static_cast<std::int32_t>(fixed_idx.size());
  const int blocks = (fixed_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_restore_fixed_values<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, fixed_idx_d, fixed_val_d, fixed_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_restore_fixed_values");
  cudaFree(fixed_idx_d);
  cudaFree(fixed_val_d);
}

struct DevicePostsolveTape {
  std::int32_t* types = nullptr;
  NnzOffset* index_starts = nullptr;
  NnzOffset* value_starts = nullptr;
  std::int32_t* indices = nullptr;
  double* vals = nullptr;
  NnzOffset count = 0;
  bool owns_storage = false;

  ~DevicePostsolveTape() {
    if (owns_storage) {
      cudaFree(types);
      cudaFree(index_starts);
      cudaFree(value_starts);
      cudaFree(indices);
      cudaFree(vals);
    }
  }
};

void upload_postsolve_tape(const PostsolveTape& tape, DevicePostsolveTape& device_tape) {
  device_tape.count = static_cast<NnzOffset>(tape.types.size());
  if (device_tape.count == 0) {
    return;
  }
  device_tape.owns_storage = true;
  device_tape.types = copy_vector_to_device(tape.types, "cudaMalloc/copy postsolve tape types");
  device_tape.index_starts =
      copy_vector_to_device(tape.index_starts, "cudaMalloc/copy postsolve tape index_starts");
  device_tape.value_starts =
      copy_vector_to_device(tape.value_starts, "cudaMalloc/copy postsolve tape value_starts");
  device_tape.indices = copy_vector_to_device(tape.indices, "cudaMalloc/copy postsolve tape indices");
  device_tape.vals = copy_vector_to_device(tape.vals, "cudaMalloc/copy postsolve tape vals");
}

void borrow_postsolve_tape(const PostsolveTapeGpu& tape,
                           DevicePostsolveTape& device_tape) {
  device_tape.types = tape.types;
  device_tape.index_starts = tape.index_starts;
  device_tape.value_starts = tape.value_starts;
  device_tape.indices = tape.indices;
  device_tape.vals = tape.vals;
  device_tape.count = tape.record_count;
  device_tape.owns_storage = false;
}

void replay_postsolve_tape_range(double* x_org,
                                 double* y_org,
                                 double* z_org,
                                 std::uint8_t* z_retrieved,
                                 const DevicePostsolveTape& tape,
                                 const LPInfoGpu* original_model,
                                 NnzOffset record_begin,
                                 NnzOffset record_end,
                                 bool replay_primal,
                                 bool replay_dual,
                                 double tol) {
  if (record_begin < 0 || record_end < record_begin ||
      record_end > tape.count) {
    throw std::runtime_error("invalid postsolve tape replay range");
  }
  if (record_begin == record_end) {
    return;
  }
  _kernel_replay_postsolve_tape<<<1, 1>>>(x_org,
                                          y_org,
                                          z_org,
                                          z_retrieved,
                                          tape.types,
                                          tape.index_starts,
                                          tape.value_starts,
                                          tape.indices,
                                          tape.vals,
                                          original_model != nullptr ? original_model->A.rowPtr : nullptr,
                                          original_model != nullptr ? original_model->A.colVal : nullptr,
                                          original_model != nullptr ? original_model->A.nzVal : nullptr,
                                          original_model != nullptr ? original_model->AT.rowPtr : nullptr,
                                          original_model != nullptr ? original_model->AT.colVal : nullptr,
                                          original_model != nullptr ? original_model->AT.nzVal : nullptr,
                                          original_model != nullptr ? original_model->c : nullptr,
                                          original_model != nullptr ? original_model->AL : nullptr,
                                          original_model != nullptr ? original_model->AU : nullptr,
                                          original_model != nullptr ? original_model->l : nullptr,
                                          original_model != nullptr ? original_model->u : nullptr,
                                          record_begin,
                                          record_end,
                                          original_model != nullptr ? 1 : 0,
                                          replay_primal ? 1 : 0,
                                          replay_dual ? 1 : 0,
                                          1,
                                          tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_replay_postsolve_tape");
}

void mark_retrieved_values(std::uint8_t* z_retrieved, const std::vector<std::int32_t>& red2org) {
  if (z_retrieved == nullptr || red2org.empty()) {
    return;
  }
  std::int32_t* red2org_d = copy_vector_to_device(red2org, "cudaMalloc/copy z_retrieved red2org");
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_mark_retrieved<<<blocks, GPU_PRESOLVE_THREADS>>>(z_retrieved, red2org_d, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_retrieved");
  cudaFree(red2org_d);
}

std::vector<std::uint8_t> build_exact_replay_column_mask(const PostsolveTape& tape, std::int32_t n) {
  std::vector<std::uint8_t> protected_cols(static_cast<std::size_t>(n), std::uint8_t{0});
  for (std::size_t k = 0; k < tape.types.size(); ++k) {
    const NnzOffset idx0 = tape.index_starts[k];
    const auto type = static_cast<PostsolveReductionType>(tape.types[k]);
    if (type == PostsolveReductionType::SubCol) {
      const std::int32_t col = tape.indices[idx0];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::DoubletonEquation) {
      const std::int32_t elim_col = tape.indices[idx0];
      const std::int32_t keep_col = tape.indices[idx0 + 1];
      if (0 <= elim_col && elim_col < n) {
        protected_cols[static_cast<std::size_t>(elim_col)] = std::uint8_t{1};
      }
      if (0 <= keep_col && keep_col < n) {
        protected_cols[static_cast<std::size_t>(keep_col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::FixedColInf) {
      const std::int32_t col = tape.indices[idx0 + 1];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::ParallelColumn) {
      const std::int32_t col = tape.indices[idx0];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    }
  }
  return protected_cols;
}

__global__ void mark_protected_device(int *mask, const int *types, const NnzOffset *starts,
                                      const int *indices, int count, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= count)
    return;
  int t = types[k];
  NnzOffset p = starts[k];
  int a = -1, b = -1;
  if (t == int(PostsolveReductionType::SubCol) || t == int(PostsolveReductionType::ParallelColumn))
    a = indices[p];
  if (t == int(PostsolveReductionType::DoubletonEquation)) {
    a = indices[p];
    b = indices[p + 1];
  }
  if (t == int(PostsolveReductionType::FixedColInf))
    a = indices[p + 1];
  if (a >= 0 && a < n)
    atomicOr(mask + a, 1);
  if (b >= 0 && b < n)
    atomicOr(mask + b, 1);
}
__global__ void pack_protected_device(unsigned char *out, const int *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = in[i] != 0;
}

void project_column_duals_from_original(double *x_org, double *y_org, double *z_org,
                                        const PresolveRecordGpu &record,
                                        const LPInfoGpu *original_model, double tol) {
  if (x_org == nullptr || y_org == nullptr || z_org == nullptr || original_model == nullptr ||
      original_model->AT.rows <= 0 || original_model->AT.rowPtr == nullptr ||
      original_model->c == nullptr || original_model->l == nullptr || original_model->u == nullptr) {
    return;
  }
  const std::int32_t n = original_model->AT.rows;
  std::uint8_t *protected_cols_d = nullptr;
  if (record.device) {
    int *mask = nullptr;
    throw_if_cuda_error(cudaMalloc(&mask, sizeof(int) * std::size_t(n)), "allocate protected mask");
    throw_if_cuda_error(cudaMemset(mask, 0, sizeof(int) * std::size_t(n)), "clear protected mask");
    throw_if_cuda_error(cudaMalloc(&protected_cols_d, std::size_t(n)), "allocate protected bytes");
    for (const auto &segment : record.device->segments) {
      mark_protected_device<<<(segment.record_count + 255) / 256, 256>>>(
          mask, segment.types, segment.index_starts, segment.indices, segment.record_count, n);
      throw_if_cuda_error(cudaGetLastError(), "mark protected columns");
    }
    pack_protected_device<<<(n + 255) / 256, 256>>>(protected_cols_d, mask, n);
    throw_if_cuda_error(cudaGetLastError(), "pack protected columns");
    recovery_device_free(mask);
  } else {
    auto protected_cols = build_exact_replay_column_mask(record.tape, n);
    protected_cols_d = copy_vector_to_device(protected_cols, "copy protected columns");
  }
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_project_column_duals_from_original<<<blocks, GPU_PRESOLVE_THREADS>>>(
      x_org,
      z_org,
      protected_cols_d,
      original_model->AT.rowPtr,
      original_model->AT.colVal,
      original_model->AT.nzVal,
      y_org,
      original_model->c,
      original_model->l,
      original_model->u,
      n,
      tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_project_column_duals_from_original");
  cudaFree(protected_cols_d);
}

void validate_postsolve_tape(const PostsolveTape& tape) {
  const std::size_t count = tape.types.size();
  if (count > static_cast<std::size_t>(
                  std::numeric_limits<NnzOffset>::max())) {
    throw std::runtime_error("postsolve tape exceeds replay offset range");
  }
  if (tape.index_starts.size() != count + 1 ||
      tape.value_starts.size() != count + 1 ||
      tape.dual_modes.size() != count) {
    throw std::runtime_error("malformed postsolve tape offsets");
  }
  if (tape.index_starts.front() != 0 || tape.value_starts.front() != 0) {
    throw std::runtime_error("postsolve tape offsets must start at zero");
  }
  for (std::size_t k = 0; k < count; ++k) {
    if (tape.index_starts[k] < 0 ||
        tape.index_starts[k] > tape.index_starts[k + 1] ||
        tape.value_starts[k] < 0 ||
        tape.value_starts[k] > tape.value_starts[k + 1]) {
      throw std::runtime_error("postsolve tape offsets are not monotone");
    }
    if (tape.types[k] <
            static_cast<std::int32_t>(PostsolveReductionType::FixedCol) ||
        tape.types[k] >
            static_cast<std::int32_t>(PostsolveReductionType::FmeCol)) {
      throw std::runtime_error("postsolve tape contains an unknown reduction");
    }
    if (tape.dual_modes[k] >
        static_cast<std::uint8_t>(PostsolveDualMode::Minimal)) {
      throw std::runtime_error("postsolve tape contains an unknown dual mode");
    }
  }
  if (static_cast<std::size_t>(tape.index_starts.back()) !=
          tape.indices.size() ||
      static_cast<std::size_t>(tape.value_starts.back()) != tape.vals.size()) {
    throw std::runtime_error("postsolve tape terminal offsets do not match data");
  }
}

void validate_primal_recovery_timeline(const PresolveRecordGpu& record) {
  if (!record.device)
    validate_postsolve_tape(record.tape);
  if (record.piecewise_linear_reformulation_absolute_value_recoveries.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      record.piecewise_linear_reformulation_variable_pair_recoveries.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error(
        "primal recovery payload exceeds 32-bit checkpoint range");
  }
  std::vector<std::uint8_t> seen_absolute_value(
      record.piecewise_linear_reformulation_absolute_value_recoveries.size(), std::uint8_t{0});
  std::vector<std::uint8_t> seen_variable_pair(
      record.piecewise_linear_reformulation_variable_pair_recoveries.size(), std::uint8_t{0});
  NnzOffset previous_tape_position = 0;
  const NnzOffset tape_count =
      record.device ? record.device->records : static_cast<NnzOffset>(record.tape.types.size());
  for (const PrimalRecoveryCheckpoint& checkpoint :
       record.primal_recovery_timeline) {
    if (checkpoint.tape_position < previous_tape_position ||
        checkpoint.tape_position > tape_count) {
      throw std::runtime_error(
          "primal recovery checkpoints are not chronological");
    }
    previous_tape_position = checkpoint.tape_position;
    if (checkpoint.kind == PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue) {
      if (checkpoint.payload_index < 0 ||
          checkpoint.payload_index >=
              static_cast<std::int32_t>(seen_absolute_value.size()) ||
          seen_absolute_value[static_cast<std::size_t>(
              checkpoint.payload_index)] != std::uint8_t{0}) {
        throw std::runtime_error(
            "invalid or duplicate piecewise_linear_reformulation_absolute_value recovery checkpoint");
      }
      seen_absolute_value[static_cast<std::size_t>(checkpoint.payload_index)] =
          std::uint8_t{1};
      validate_piecewise_linear_reformulation_absolute_value_recovery(
          record.piecewise_linear_reformulation_absolute_value_recoveries[static_cast<std::size_t>(
              checkpoint.payload_index)],
          record.n0);
    } else if (checkpoint.kind ==
               PrimalRecoveryKind::PiecewiseLinearReformulationVariablePair) {
      if (checkpoint.payload_index < 0 ||
          checkpoint.payload_index >=
              static_cast<std::int32_t>(seen_variable_pair.size()) ||
          seen_variable_pair[static_cast<std::size_t>(
              checkpoint.payload_index)] != std::uint8_t{0}) {
        throw std::runtime_error(
            "invalid or duplicate piecewise_linear_reformulation_variable_pair recovery checkpoint");
      }
      seen_variable_pair[static_cast<std::size_t>(checkpoint.payload_index)] =
          std::uint8_t{1};
      validate_piecewise_linear_reformulation_variable_pair_recovery(
          record.piecewise_linear_reformulation_variable_pair_recoveries[static_cast<std::size_t>(
              checkpoint.payload_index)],
          record.n0);
    } else {
      throw std::runtime_error("unknown primal recovery checkpoint kind");
    }
  }
  for (const std::uint8_t seen : seen_absolute_value) {
    if (seen == std::uint8_t{0}) {
      throw std::runtime_error(
          "piecewise_linear_reformulation_absolute_value recovery payload is missing a checkpoint");
    }
  }
  for (const std::uint8_t seen : seen_variable_pair) {
    if (seen == std::uint8_t{0}) {
      throw std::runtime_error(
          "piecewise_linear_reformulation_variable_pair recovery payload is missing a checkpoint");
    }
  }
}

}  // namespace

void postsolve_restore_piecewise_linear_reformulation_absolute_value_gpu(
    double* x_org,
    const std::vector<PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep>& recoveries,
    const double* original_l) {
  for (auto it = recoveries.rbegin(); it != recoveries.rend(); ++it) {
    restore_piecewise_linear_reformulation_absolute_value_step(x_org, *it, original_l);
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_restore_piecewise_linear_reformulation_absolute_value_gpu synchronize");
}

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const double* original_l) {
  return postsolve_gpu(x_red, y_red, z_red, record, nullptr, original_l);
}

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const LPInfoGpu* original_model,
                                 const double* original_l) {
  validate_device_recovery(record);
  validate_primal_recovery_timeline(record);
  GpuPostsolveResult result;
  result.n0 = record.n0;
  result.m0 = record.m0;
  if (original_l == nullptr && original_model != nullptr) {
    original_l = original_model->l;
  }
  if (record.n0 > 0) {
    throw_if_cuda_error(cudaMalloc(&result.x_org, sizeof(double) * static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve x_org");
    throw_if_cuda_error(cudaMalloc(&result.z_org, sizeof(double) * static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve z_org");
  }
  if (record.m0 > 0) {
    throw_if_cuda_error(cudaMalloc(&result.y_org, sizeof(double) * static_cast<std::size_t>(record.m0)),
                        "cudaMalloc postsolve y_org");
  }

  std::uint8_t* z_retrieved = nullptr;
  if (record.n0 > 0) {
    throw_if_cuda_error(cudaMalloc(&z_retrieved, static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve z_retrieved");
    throw_if_cuda_error(cudaMemset(z_retrieved, 0, static_cast<std::size_t>(record.n0)),
                        "cudaMemset postsolve z_retrieved");
  }

  if (record.device) {
    const auto &d = *record.device;
    auto scatter = [&](double *dst, const double *src, const RecoveryArray<int> &map, int count) {
      if (count)
        throw_if_cuda_error(cudaMemset(dst, 0, sizeof(double) * std::size_t(count)),
                            "clear original vector");
      if (src && map.count)
        _kernel_scatter_reduced_to_original<<<(map.count + 255) / 256, 256>>>(dst, src, map.data,
                                                                              map.count);
    };
    scatter(result.x_org, x_red, d.cols, record.n0);
    scatter(result.y_org, y_red, d.rows, record.m0);
    scatter(result.z_org, z_red, d.cols, record.n0);
    if (d.cols.count)
      _kernel_mark_retrieved<<<(d.cols.count + 255) / 256, 256>>>(z_retrieved, d.cols.data,
                                                                  d.cols.count);
    for (const auto &f : d.fixed)
      if (f.indices.count)
        _kernel_restore_fixed_values<<<(f.indices.count + 255) / 256, 256>>>(
            result.x_org, f.indices.data, f.values.data, f.indices.count);
    throw_if_cuda_error(cudaGetLastError(), "device recovery scatter/fixed");
  } else {
    scatter_reduced_to_original(result.x_org, x_red, record.col_red2org, record.n0,
                                "cudaMemset/scatter x_org");
    scatter_reduced_to_original(result.y_org, y_red, record.row_red2org, record.m0,
                                "cudaMemset/scatter y_org");
    scatter_reduced_to_original(result.z_org, z_red, record.col_red2org, record.n0,
                                "cudaMemset/scatter z_org");
    mark_retrieved_values(z_retrieved, record.col_red2org);
    restore_fixed_values(result.x_org, record.fixed_idx, record.fixed_val);
  }
  DevicePostsolveTape device_tape;
  if (record.device) {
    device_tape.count = record.device->records;
  } else if (record.tape_gpu.record_count > 0) {
    if (record.tape_gpu.record_count !=
            static_cast<NnzOffset>(record.tape.types.size()) ||
        record.tape_gpu.index_count !=
            static_cast<NnzOffset>(record.tape.indices.size()) ||
        record.tape_gpu.value_count !=
            static_cast<NnzOffset>(record.tape.vals.size())) {
      throw std::runtime_error(
          "device postsolve tape does not mirror the canonical CPU tape");
    }
    borrow_postsolve_tape(record.tape_gpu, device_tape);
  } else {
    upload_postsolve_tape(record.tape, device_tape);
  }
  auto replay_range = [&](NnzOffset begin, NnzOffset end, bool primal, bool dual) {
    if (!record.device) {
      replay_postsolve_tape_range(result.x_org, result.y_org, result.z_org, z_retrieved,
                                  device_tape, original_model, begin, end, primal, dual, 1e-7);
      return;
    }
    NnzOffset stop = record.device->records;
    for (auto it = record.device->segments.rbegin(); it != record.device->segments.rend(); ++it) {
      NnzOffset start = stop - it->record_count;
      NnzOffset lo = std::max(begin, start), hi = std::min(end, stop);
      if (lo < hi) {
        DevicePostsolveTape view;
        borrow_postsolve_tape(*it, view);
        replay_postsolve_tape_range(result.x_org, result.y_org, result.z_org, z_retrieved, view,
                                    original_model, lo - start, hi - start, primal, dual, 1e-7);
      }
      stop = start;
    }
  };
  NnzOffset range_end = device_tape.count;
  for (auto checkpoint_it = record.primal_recovery_timeline.rbegin();
       checkpoint_it != record.primal_recovery_timeline.rend();
       ++checkpoint_it) {
    replay_range(checkpoint_it->tape_position, range_end, true, false);
    if (checkpoint_it->kind == PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue) {
      if (record.device)
        restore_device_structural(
            result.x_org, record.device->piecewise_linear_reformulation_absolute_value[checkpoint_it->payload_index], original_l);
      else
        restore_piecewise_linear_reformulation_absolute_value_step(result.x_org,
                                       record.piecewise_linear_reformulation_absolute_value_recoveries[static_cast<std::size_t>(
                                           checkpoint_it->payload_index)],
                                       original_l);
    } else {
      if (record.device)
        restore_device_piecewise_linear_reformulation_variable_pair(result.x_org,
                                 record.device->piecewise_linear_reformulation_variable_pair[checkpoint_it->payload_index]);
      else
        restore_piecewise_linear_reformulation_variable_pair(result.x_org,
                                    record.piecewise_linear_reformulation_variable_pair_recoveries[static_cast<std::size_t>(
                                        checkpoint_it->payload_index)]);
    }
    range_end = checkpoint_it->tape_position;
  }
  replay_range(0, range_end, true, false);
  replay_range(0, device_tape.count, false, true);
  project_column_duals_from_original(result.x_org, result.y_org, result.z_org, record,
                                     original_model, 1.0e-7);
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_gpu synchronize");
  cudaFree(z_retrieved);
  return result;
}

}  // namespace gpu_presolver::presolve
