// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/presolve/device_recovery.hpp"

namespace gpu_preprocessor::presolve::postsolve {

void restore_piecewise_linear_reformulation_absolute_value_step(
    double* x_org,
    const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& recovery,
    const double* original_l);

void restore_device_structural(double *x, const DevicePiecewiseLinearReformulationAbsoluteValueStep &s, const double *lower);

void validate_piecewise_linear_reformulation_absolute_value_recovery(
    const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& recovery,
    std::int32_t n0);

}  // namespace gpu_preprocessor::presolve::postsolve
