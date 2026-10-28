#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_piecewise_linear_reformulation_absolute_value(PresolvePlanGpu& plan,
                                           const LPInfoGpu& lp,
                                           const PresolveStatsGpu& stats,
                                           const PresolveParams& pparams);

bool piecewise_linear_reformulation_absolute_value_prefix_screen_passes(const LPInfoGpu& lp);

}  // namespace gpu_presolver::presolve
