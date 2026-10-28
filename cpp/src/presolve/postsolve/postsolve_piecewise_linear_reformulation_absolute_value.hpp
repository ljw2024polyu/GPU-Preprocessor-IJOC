#pragma once

#include "gpu_presolver/presolve/device_recovery.hpp"

namespace gpu_presolver::presolve::postsolve {

void restore_piecewise_linear_reformulation_absolute_value_step(
    double* x_org,
    const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& recovery,
    const double* original_l);

void restore_device_structural(double *x, const DevicePiecewiseLinearReformulationAbsoluteValueStep &s, const double *lower);

void validate_piecewise_linear_reformulation_absolute_value_recovery(
    const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& recovery,
    std::int32_t n0);

}  // namespace gpu_presolver::presolve::postsolve
