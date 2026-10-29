// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::presolve {

void apply_rule_doubleton_equations(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                                    const PresolveStatsGpu &stats,
                                    const PresolveParams &pparams);

} // namespace gpu_preprocessor::presolve
