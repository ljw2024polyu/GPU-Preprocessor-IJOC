// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/presolve/presolve_structs.hpp"

#include <vector>

namespace gpu_preprocessor::presolve {

// Recovery helper for the absolute-value branch of piecewise-linear
// reformulation. Use postsolve_gpu for any record that may contain ordinary
// tape entries or interleaved primal-recovery checkpoints.
void postsolve_restore_piecewise_linear_reformulation_absolute_value_gpu(
    double* x_org,
    const std::vector<PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep>& recoveries,
    const double* original_l = nullptr);

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const double* original_l = nullptr);

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const LPInfoGpu* original_model,
                                 const double* original_l = nullptr);

}  // namespace gpu_preprocessor::presolve
