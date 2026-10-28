#pragma once

#include "postsolve_common.cuh"

namespace gpu_presolver::presolve::postsolve {

// Projected-column records shared by zero-cost product and two-row variable elimination.

__device__ __forceinline__ void replay_projected_column(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    NnzOffset idx0,
    NnzOffset val0,
    std::int32_t replay_primal,
    std::int32_t replay_dual) {
  const std::int32_t K = tape_idx[idx0];
  const std::int32_t elim_col = tape_idx[idx0 + 1];
  const std::int32_t L = static_cast<std::int32_t>(llround(tape_val[val0]));
  NnzOffset col_cursor = idx0 + 3 + K;
  NnzOffset vpos = val0 + 1;
  double lo = -INFINITY;
  double hi = INFINITY;
  bool has_lo = false;
  bool has_hi = false;
  if (replay_primal != 0) {
    for (std::int32_t s = 0; s < L; ++s) {
      const double orient = tape_val[vpos];
      const NnzOffset len = static_cast<NnzOffset>(llround(tape_val[vpos + 1]));
      double cand = tape_val[vpos + 2];
      for (std::int32_t t = 0; t < len; ++t) {
        const std::int32_t row_col = tape_idx[col_cursor++];
        const double coeff = tape_val[vpos + 3 + t];
        cand += coeff * x_org[row_col];
      }
      if (orient > 0.0) {
        lo = fmax(lo, cand);
        has_lo = true;
      } else {
        hi = fmin(hi, cand);
        has_hi = true;
      }
      vpos += 3 + len;
    }
    x_org[elim_col] = has_lo ? lo : (has_hi ? hi : 0.0);
  }
  if (replay_dual != 0 && z_org != nullptr) {
    z_org[elim_col] = 0.0;
    if (z_retrieved != nullptr) {
      z_retrieved[elim_col] = std::uint8_t{1};
    }
  }
  if (replay_dual != 0 && y_org != nullptr) {
    for (std::int32_t t = 0; t < K; ++t) {
      y_org[tape_idx[idx0 + 2 + t]] = 0.0;
    }
  }
}

}  // namespace gpu_presolver::presolve::postsolve
