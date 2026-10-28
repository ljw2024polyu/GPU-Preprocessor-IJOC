#pragma once

#include "postsolve_common.cuh"

namespace gpu_presolver::presolve::postsolve {

__device__ __forceinline__ void replay_row_bound_change(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    const NnzOffset* A_rowPtr,
    const std::int32_t* A_colVal,
    const double* A_nzVal,
    NnzOffset idx0,
    NnzOffset val0,
    double tol) {
  const std::int32_t col = tape_idx[idx0];
  const std::int32_t row = tape_idx[idx0 + 1];
  if (z_retrieved != nullptr && z_retrieved[col] == std::uint8_t{0}) {
    return;
  }
  const double old_l = tape_val[val0];
  const double old_u = tape_val[val0 + 1];
  const double new_l = tape_val[val0 + 2];
  const double new_u = tape_val[val0 + 3];
  const bool lower_changed = new_l > old_l + tol;
  const bool upper_changed = new_u < old_u - tol;
  if (lower_changed == upper_changed) {
    return;
  }
  const double implied_bound = lower_changed ? new_l : new_u;
  const double original_other_bound = lower_changed ? old_u : old_l;
  const bool original_other_is_lower = !lower_changed;
  const double xj = x_org[col];
  const double zj = z_org[col];
  if (_postsolve_is_finite(original_other_bound) && fabs(xj - original_other_bound) <= tol) {
    const bool sign_ok = original_other_is_lower ? (zj >= -tol) : (zj <= tol);
    if (sign_ok) {
      return;
    }
  }
  if (fabs(xj - implied_bound) > tol) {
    return;
  }
  double aij = 0.0;
  for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    if (A_colVal[p] == col) {
      aij = A_nzVal[p];
      break;
    }
  }
  if (fabs(aij) <= tol) {
    return;
  }
  const double delta_y = zj / aij;
  if (y_org != nullptr) {
    y_org[row] += delta_y;
  }
  for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    const std::int32_t kcol = A_colVal[p];
    if (kcol == col || (z_retrieved != nullptr && z_retrieved[kcol] == std::uint8_t{0})) {
      continue;
    }
    z_org[kcol] -= (A_nzVal[p] / aij) * zj;
  }
  z_org[col] = 0.0;
  if (z_retrieved != nullptr) {
    z_retrieved[col] = std::uint8_t{1};
  }
}

}  // namespace gpu_presolver::presolve::postsolve
