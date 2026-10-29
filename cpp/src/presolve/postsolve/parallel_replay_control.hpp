// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace gpu_presolver::presolve::postsolve::parallel {

// Internal controls for paired replay checks; no public parameter or tape change.
struct Options {
  bool fixed = true;
  bool subcol = true;
  bool components = true;
  bool collect_statistics = false;
  std::size_t scratch_limit_bytes = std::numeric_limits<std::size_t>::max();
};
struct Statistics {
  std::size_t fixed_ranges = 0;
  std::size_t subcol_ranges = 0;
  std::size_t component_ranges = 0;
  std::size_t components = 0;
  std::size_t parallel_records = 0;
  std::size_t fallback_ranges = 0;
};
inline Options& options() { static thread_local Options value; return value; }
inline Statistics& statistics() { static thread_local Statistics value; return value; }
inline bool enabled(bool value, const char* name) {
  const char* setting = std::getenv(name);
  return value && (!setting || (std::strcmp(setting, "0") && std::strcmp(setting, "false")));
}

} // namespace gpu_presolver::presolve::postsolve::parallel
