# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

# The proprietary runtime and license are installed separately.
set(GUROBI_ROOT "$ENV{GUROBI_HOME}" CACHE PATH "Gurobi 12.0.2 platform directory (e.g. linux64)")
find_path(GUROBI_INCLUDE_DIR gurobi_c.h HINTS "${GUROBI_ROOT}" ENV GUROBI_HOME
  PATH_SUFFIXES include NO_DEFAULT_PATH)
find_library(GUROBI_C_LIBRARY NAMES gurobi120 HINTS "${GUROBI_ROOT}" ENV GUROBI_HOME
  PATH_SUFFIXES lib NO_DEFAULT_PATH)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gurobi REQUIRED_VARS GUROBI_INCLUDE_DIR GUROBI_C_LIBRARY
  REASON_FAILURE_MESSAGE "Install Gurobi 12.0.2 and set GUROBI_ROOT or GUROBI_HOME to its platform directory, or build with HPRLP_GUROBI=OFF.")
if(Gurobi_FOUND AND NOT TARGET Gurobi::C)
  add_library(Gurobi::C SHARED IMPORTED)
  set_target_properties(Gurobi::C PROPERTIES IMPORTED_LOCATION "${GUROBI_C_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIR}")
endif()
mark_as_advanced(GUROBI_INCLUDE_DIR GUROBI_C_LIBRARY)
