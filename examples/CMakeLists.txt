# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)
project(GPUPreprocessorExample LANGUAGES CXX CUDA)

find_package(GPUPreprocessor CONFIG REQUIRED)
add_executable(gpu_preprocessor_example presolve_example.cpp)
target_link_libraries(gpu_preprocessor_example PRIVATE GPUPreprocessor::core)
