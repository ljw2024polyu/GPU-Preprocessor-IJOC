// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/presolve/device_recovery.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_preprocessor::presolve::postsolve {

constexpr int GPU_PRESOLVE_THREADS = 256;

inline void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
T* copy_vector_to_device(const std::vector<T>& values, const char* context) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  throw_if_cuda_error(cudaMalloc(&device, sizeof(T) * values.size()), context);
  throw_if_cuda_error(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
                      context);
  return device;
}

__device__ __forceinline__ bool _postsolve_is_finite(double x) {
  return x > -INFINITY && x < INFINITY;
}

__device__ __forceinline__ bool _postsolve_at_lower(double x, double l, double tol) {
  return _postsolve_is_finite(l) && x <= l + tol;
}

__device__ __forceinline__ bool _postsolve_at_upper(double x, double u, double tol) {
  return _postsolve_is_finite(u) && x >= u - tol;
}

__device__ __forceinline__ double _clamp_with_inf(double x, double lo, double hi) {
  double y = x;
  if (_postsolve_is_finite(lo)) {
    y = fmax(y, lo);
  }
  if (_postsolve_is_finite(hi)) {
    y = fmin(y, hi);
  }
  return y;
}

__device__ __forceinline__ double _postsolve_compute_aty_col(const NnzOffset* AT_rowPtr,
                                             const std::int32_t* AT_colVal,
                                             const double* AT_nzVal,
                                             const double* y_org,
                                             std::int32_t col) {
  double aty = 0.0;
  for (NnzOffset p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
    aty += AT_nzVal[p] * y_org[AT_colVal[p]];
  }
  return aty;
}

__device__ __forceinline__ double _postsolve_project_single_column_dual_from_original(double* x_org,
                                                                      double* z_org,
                                                                      const NnzOffset* AT_rowPtr,
                                                                      const std::int32_t* AT_colVal,
                                                                      const double* AT_nzVal,
                                                                      const double* y_org,
                                                                      const double* c,
                                                                      const double* l,
                                                                      const double* u,
                                                                      std::int32_t col,
                                                                      double tol) {
  const double r = c[col] - _postsolve_compute_aty_col(AT_rowPtr, AT_colVal, AT_nzVal, y_org, col);
  const double xj = x_org[col];
  const double lj = l[col];
  const double uj = u[col];
  const bool at_lower = _postsolve_at_lower(xj, lj, tol);
  const bool at_upper = _postsolve_at_upper(xj, uj, tol);
  double zj = 0.0;
  if (at_lower && at_upper) {
    zj = r;
  } else if (!at_lower && !at_upper) {
    zj = 0.0;
  } else if (at_lower) {
    zj = fmax(r, 0.0);
  } else {
    zj = fmin(r, 0.0);
  }
  const double dual_eps = fmin(tol, 1.0e-12);
  if (zj > dual_eps && !isfinite(lj)) {
    zj = 0.0;
  } else if (zj < -dual_eps && !isfinite(uj)) {
    zj = 0.0;
  }
  z_org[col] = zj;
  return zj;
}

inline void validate_recovery_col(std::int32_t col,
                           std::int32_t n0,
                           const char* context) {
  if (col < 0 || col >= n0) {
    throw std::runtime_error(std::string(context) +
                             " contains an invalid original column");
  }
}

__device__ __forceinline__ void replay_row_dual_shift(
    double* y_org,
    const std::int32_t* tape_idx,
    const double* tape_val,
    NnzOffset idx0,
    NnzOffset val0) {
  if (y_org != nullptr) {
    const std::int32_t row = tape_idx[idx0];
    y_org[row] += tape_val[val0];
  }
}

}  // namespace gpu_preprocessor::presolve::postsolve
