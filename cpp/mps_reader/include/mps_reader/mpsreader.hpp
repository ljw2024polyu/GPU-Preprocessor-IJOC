// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "mps_reader/lp_model.hpp"

#include <iosfwd>
#include <string>

namespace mps_reader {

struct MpsModel {
  std::string name;
  LpModel lp;
};

MpsModel read_mps(std::istream& input);

}  // namespace mps_reader
