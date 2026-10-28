#pragma once

#include "postsolve_common.cuh"

namespace gpu_presolver::presolve::postsolve {

enum : std::uint8_t {
  PARALLEL_ROW_STATE_ZERO = 0,
  PARALLEL_ROW_STATE_NONNEG = 1,
  PARALLEL_ROW_STATE_NONPOS = 2,
  PARALLEL_ROW_STATE_FREE = 3,
};

__device__ __forceinline__ void _parallel_row_scaled_interval(double lower,
                                              double upper,
                                              double ratio,
                                              double* scaled_l,
                                              double* scaled_u) {
  const double raw_l = lower * ratio;
  const double raw_u = upper * ratio;
  if (ratio >= 0.0) {
    *scaled_l = raw_l;
    *scaled_u = raw_u;
  } else {
    *scaled_l = raw_u;
    *scaled_u = raw_l;
  }
}

__device__ __forceinline__ bool _parallel_row_interval_contains(double outer_lo,
                                                double outer_hi,
                                                double inner_lo,
                                                double inner_hi,
                                                double tol) {
  return outer_lo <= inner_lo + tol && inner_hi <= outer_hi + tol;
}

__device__ __forceinline__ double _parallel_row_activity(std::int32_t row,
                                         const NnzOffset* A_rowPtr,
                                         const std::int32_t* A_colVal,
                                         const double* A_nzVal,
                                         const double* x_org) {
  double activity = 0.0;
  for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    activity += A_nzVal[p] * x_org[A_colVal[p]];
  }
  return activity;
}

__device__ __forceinline__ std::uint8_t _parallel_row_dual_state(double activity,
                                                 double lower,
                                                 double upper,
                                                 double tol) {
  const bool at_lower = _postsolve_is_finite(lower) && activity <= lower + tol;
  const bool at_upper = _postsolve_is_finite(upper) && activity >= upper - tol;
  if (at_lower && at_upper) {
    return PARALLEL_ROW_STATE_FREE;
  }
  if (at_lower) {
    return PARALLEL_ROW_STATE_NONNEG;
  }
  if (at_upper) {
    return PARALLEL_ROW_STATE_NONPOS;
  }
  return PARALLEL_ROW_STATE_ZERO;
}

__device__ __forceinline__ void _parallel_row_state_interval(std::uint8_t state, double* lo, double* hi) {
  if (state == PARALLEL_ROW_STATE_FREE) {
    *lo = -INFINITY;
    *hi = INFINITY;
  } else if (state == PARALLEL_ROW_STATE_NONNEG) {
    *lo = 0.0;
    *hi = INFINITY;
  } else if (state == PARALLEL_ROW_STATE_NONPOS) {
    *lo = -INFINITY;
    *hi = 0.0;
  } else {
    *lo = 0.0;
    *hi = 0.0;
  }
}

__device__ __forceinline__ void _parallel_row_deleted_dual_interval(double ybar_p,
                                                    double ratio,
                                                    std::uint8_t kept_state,
                                                    double* lo,
                                                    double* hi) {
  const double pivot = ratio * ybar_p;
  if (kept_state == PARALLEL_ROW_STATE_FREE) {
    *lo = -INFINITY;
    *hi = INFINITY;
  } else if (kept_state == PARALLEL_ROW_STATE_ZERO) {
    *lo = pivot;
    *hi = pivot;
  } else if (kept_state == PARALLEL_ROW_STATE_NONNEG) {
    if (ratio > 0.0) {
      *lo = -INFINITY;
      *hi = pivot;
    } else {
      *lo = pivot;
      *hi = INFINITY;
    }
  } else {
    if (ratio > 0.0) {
      *lo = pivot;
      *hi = INFINITY;
    } else {
      *lo = -INFINITY;
      *hi = pivot;
    }
  }
}

__device__ __forceinline__ bool _parallel_row_doc_dual_split(double ybar_p,
                                             double ratio,
                                             double kept_old_AL,
                                             double kept_old_AU,
                                             double deleted_scaled_AL,
                                             double deleted_scaled_AU,
                                             double tol,
                                             double* yp,
                                             double* yd) {
  if (fabs(ybar_p) <= tol) {
    *yp = 0.0;
    *yd = 0.0;
    return true;
  }

  const double bar_l = fmax(kept_old_AL, deleted_scaled_AL);
  const double bar_u = fmin(kept_old_AU, deleted_scaled_AU);
  if (!(bar_l <= bar_u + tol)) {
    *yp = ybar_p;
    *yd = 0.0;
    return false;
  }

  const bool lower_tightened = bar_l > kept_old_AL + tol;
  const bool upper_tightened = bar_u < kept_old_AU - tol;
  if (lower_tightened && upper_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
    return false;
  }
  if (!lower_tightened && !upper_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
    return true;
  }

  if (ybar_p > 0.0) {
    if (lower_tightened) {
      *yp = 0.0;
      *yd = ratio * ybar_p;
    } else {
      *yp = ybar_p;
      *yd = 0.0;
    }
    return true;
  }

  if (lower_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
  } else {
    *yp = 0.0;
    *yd = ratio * ybar_p;
  }
  return true;
}

__device__ __forceinline__ void replay_parallel_row(
    double* x_org,
    double* y_org,
    const std::int32_t* tape_idx,
    const double* tape_val,
    const NnzOffset* A_rowPtr,
    const std::int32_t* A_colVal,
    const double* A_nzVal,
    NnzOffset idx0,
    NnzOffset val0,
    std::int32_t has_original_model,
    std::int32_t parallel_row_doc_recovery,
    double tol) {
  const std::int32_t kept_row = tape_idx[idx0];
  const std::int32_t deleted_row = tape_idx[idx0 + 1];
  const double ratio = tape_val[val0];
  const double kept_old_AL = tape_val[val0 + 1];
  const double kept_old_AU = tape_val[val0 + 2];
  const double deleted_old_AL = tape_val[val0 + 3];
  const double deleted_old_AU = tape_val[val0 + 4];
  if (fabs(ratio) <= tol || y_org == nullptr) {
    return;
  }
  double deleted_scaled_AL = 0.0;
  double deleted_scaled_AU = 0.0;
  _parallel_row_scaled_interval(deleted_old_AL, deleted_old_AU, ratio,
                                &deleted_scaled_AL, &deleted_scaled_AU);
  if (_parallel_row_interval_contains(deleted_scaled_AL,
                                      deleted_scaled_AU,
                                      kept_old_AL,
                                      kept_old_AU,
                                      tol)) {
    y_org[deleted_row] = 0.0;
    return;
  }
  if (has_original_model != 0) {
    const double alpha_p = _parallel_row_activity(kept_row, A_rowPtr, A_colVal, A_nzVal, x_org);
    const double alpha_d = _parallel_row_activity(deleted_row, A_rowPtr, A_colVal, A_nzVal, x_org);
    const std::uint8_t kept_state =
        _parallel_row_dual_state(alpha_p, kept_old_AL, kept_old_AU, tol);
    const std::uint8_t deleted_state =
        _parallel_row_dual_state(alpha_d, deleted_old_AL, deleted_old_AU, tol);
    double yd_lo = 0.0;
    double yd_hi = 0.0;
    double yp_lo = 0.0;
    double yp_hi = 0.0;
    _parallel_row_state_interval(deleted_state, &yd_lo, &yd_hi);
    _parallel_row_deleted_dual_interval(y_org[kept_row], ratio, kept_state, &yp_lo, &yp_hi);
    const double sd_lo = fmax(yd_lo, yp_lo);
    const double sd_hi = fmin(yd_hi, yp_hi);
    if (sd_lo <= sd_hi + tol) {
      const double yd_new = _clamp_with_inf(0.0, sd_lo, sd_hi);
      const double yp_new = y_org[kept_row] - yd_new / ratio;
      y_org[kept_row] = yp_new;
      y_org[deleted_row] = yd_new;
      return;
    }

    if (parallel_row_doc_recovery != 0) {
      double yp_doc = y_org[kept_row];
      double yd_doc = 0.0;
      if (_parallel_row_doc_dual_split(y_org[kept_row],
                                       ratio,
                                       kept_old_AL,
                                       kept_old_AU,
                                       deleted_scaled_AL,
                                       deleted_scaled_AU,
                                       tol,
                                       &yp_doc,
                                       &yd_doc)) {
        y_org[kept_row] = yp_doc;
        y_org[deleted_row] = yd_doc;
      }
    }
  }
}

}  // namespace gpu_presolver::presolve::postsolve
