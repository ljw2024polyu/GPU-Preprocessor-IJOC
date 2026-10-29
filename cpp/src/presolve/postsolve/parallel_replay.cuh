// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "parallel_replay_control.hpp"
#include "gpu_presolver/presolve/presolve_structs.hpp"
// Isolate radix-sort kernels from other CUB instantiations in the application.
#ifndef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE gpu_preprocessor_postsolve_cub
#endif
#include <cub/cub.cuh>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve::postsolve::parallel {

constexpr NnzOffset min_records = 1024;
using Key = std::uint64_t;
enum class Mode { Serial, Fixed, Subcol, Components };
struct Tape {
  const int* types;
  const NnzOffset* is;
  const NnzOffset* vs;
  const int* ids;
  const double* values;
};
struct State {
  double *x, *y, *z;
  std::uint8_t* retrieved;
  LPInfoGpu model;
  int m, n;
  bool original, primal, dual;
  int parallel_row_doc;
  double tol;
};
inline void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("parallel postsolve: ") + cudaGetErrorString(status));
}
inline int blocks(NnzOffset count) {
  return int((std::min(count, NnzOffset(256 * 1024)) + 255) / 256);
}
template<class T> struct Buffer {
  T* data = nullptr;
  std::size_t capacity = 0;
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { if (data) cudaFree(data); }
};

__global__ void certify_fixed(int* invalid, Tape t, NnzOffset begin, NnzOffset end, int m, int n) {
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += std::uint64_t(blockDim.x) * gridDim.x) {
    const auto ip = t.is[k], ni = t.is[k + 1] - ip, nv = t.vs[k + 1] - t.vs[k];
    if (t.types[k] != int(PostsolveReductionType::FixedVariable) || ni < 1 || nv != ni + 1) {
      atomicExch(invalid, 1); continue;
    }
    const int col = t.ids[ip];
    if (col < 0 || col >= n || (k > std::uint64_t(begin) && t.ids[t.is[k - 1]] >= col))
      atomicExch(invalid, 1);
    for (NnzOffset p = 1; p < ni; ++p)
      if (t.ids[ip + p] < 0 || t.ids[ip + p] >= m) atomicExch(invalid, 1);
  }
}
__global__ void certify_subcol_targets(int* invalid, int* targets, Tape t,
    NnzOffset begin, NnzOffset end, State s) {
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += std::uint64_t(blockDim.x) * gridDim.x) {
    const auto ip = t.is[k], vp = t.vs[k];
    const auto ni = t.is[k + 1] - ip, nv = t.vs[k + 1] - vp;
    if (t.types[k] != int(PostsolveReductionType::SingletonColumnSubstitution) || ni < 3 || nv < 6) {
      atomicExch(invalid, 1); continue;
    }
    const int col = t.ids[ip], row = t.ids[ip + 1], count = t.ids[ip + 2];
    if (col < 0 || col >= s.n || row < 0 || row >= s.m || count < 0 || ni - 3 != count ||
        nv - 6 != count || !isfinite(t.values[vp]) || t.values[vp] == 0.0 ||
        !isfinite(t.values[vp + 1]) || !isfinite(t.values[vp + 4]) ||
        (s.y && !isfinite(s.y[row]))) {
      atomicExch(invalid, 1); continue;
    }
    if (k > std::uint64_t(begin) && (t.types[k - 1] != t.types[k] ||
        t.is[k] - t.is[k - 1] < 3 || t.ids[t.is[k - 1] + 1] >= row)) atomicExch(invalid, 1);
    if (atomicExch(targets + col, 1)) atomicExch(invalid, 1);
  }
}
__global__ void certify_subcol_support(int* invalid, const int* targets, Tape t,
    NnzOffset begin, NnzOffset end, State s) {
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += std::uint64_t(blockDim.x) * gridDim.x) {
    const auto ip = t.is[k], vp = t.vs[k];
    if (t.types[k] != int(PostsolveReductionType::SingletonColumnSubstitution) ||
        t.is[k + 1] - ip < 3 || t.vs[k + 1] - vp < 6) continue;
    const int count = t.ids[ip + 2];
    if (count < 0 || t.is[k + 1] - ip - 3 != count || t.vs[k + 1] - vp - 6 != count) continue;
    for (int p = 0; p < count; ++p) {
      const int col = t.ids[ip + 3 + p];
      if (col < 0 || col >= s.n || targets[col] || !isfinite(s.x[col]) ||
          !isfinite(t.values[vp + 6 + p])) atomicExch(invalid, 1);
    }
  }
}

// Each mutable column resource includes x, z and its retrieval flag. This
// conservative grouping also covers primal handlers that inspect reduced costs.
__device__ inline int root(int* parent, int a) {
  for (;;) { const int b = atomicAdd(parent + a, 0); if (a == b) return a; a = b; }
}
__device__ inline void unite(int* parent, int a, int b) {
  for (;;) {
    a = root(parent, a); b = root(parent, b);
    if (a == b) return;
    const int hi = max(a, b), lo = min(a, b);
    const int previous = atomicMin(parent + hi, lo);
    if (previous == hi) return;
    a = previous; b = lo;
  }
}
struct Access {
  int *invalid, *columns, *rows, *parent;
  int m, n, record;
  bool connect;
  __device__ bool require(bool value) const { if (!value) atomicExch(invalid, 1); return value; }
  __device__ void use(int id, bool row, bool write) const {
    if (!require(id >= 0 && id < (row ? m : n))) return;
    int* owners = row ? rows : columns;
    if (!connect) { if (write) atomicCAS(owners + id, -1, record); }
    else { const int owner = owners[id]; if (owner >= 0) unite(parent, record, owner); }
  }
  __device__ void col(int id, bool write = false) const { use(id, false, write); }
  __device__ void row(int id, bool write = false) const { use(id, true, write); }
  __device__ void original_column(int col, const State& s) const {
    if (!require(col >= 0 && col < n && s.model.AT.rowPtr && s.model.AT.colVal)) return;
    for (auto p = s.model.AT.rowPtr[col]; p < s.model.AT.rowPtr[col + 1]; ++p)
      row(s.model.AT.colVal[p]);
  }
  __device__ void original_row(int id, const State& s) const {
    if (!require(id >= 0 && id < m && s.model.A.rowPtr && s.model.A.colVal)) return;
    for (auto p = s.model.A.rowPtr[id]; p < s.model.A.rowPtr[id + 1]; ++p)
      col(s.model.A.colVal[p]);
  }
};
__device__ inline void accesses(Access a, int type, const int* ids, NnzOffset ni,
    const double* v, NnzOffset nv, const State& s) {
  using T = PostsolveReductionType;
  switch (static_cast<T>(type)) {
  case T::FixedVariable:
    if (!a.require(ni >= 1 && nv == ni + 1)) return;
    a.col(ids[0], true);
    if (s.dual) for (NnzOffset p = 1; p < ni; ++p) a.row(ids[p]);
    break;
  case T::SingletonColumnSubstitution:
    if (!a.require(ni >= 3 && nv >= 6 && ids[2] >= 0 && ni - 3 == ids[2] && nv - 6 == ids[2])) return;
    a.col(ids[0], true);
    if (ids[1] >= 0 && s.dual) a.row(ids[1], true);
    for (int p = 0; p < ids[2]; ++p) a.col(ids[3 + p]);
    break;
  case T::DoubletonEquation:
    if (!a.require(ni >= 3 && nv >= 10)) return;
    a.col(ids[0], true); a.col(ids[1], true);
    if (s.dual) {
      if (ids[2] >= 0) a.row(ids[2], true);
      if (s.original) {
        if (ni >= 4 && nv >= 12) {
          if (!a.require(ids[3] >= 0 && ni - 4 == ids[3] && nv - 12 == ids[3])) return;
          for (int p = 0; p < ids[3]; ++p) a.row(ids[4 + p]);
        } else a.original_column(ids[0], s);
      }
    }
    break;
  case T::ParallelColumn:
    if (!a.require(ni == 2 && nv >= 5)) return;
    a.col(ids[0], true); a.col(ids[1], true); break;
  case T::ParallelRow:
    if (!a.require(ni == 2 && nv >= 5)) return;
    if (s.dual) {
      a.row(ids[0], true); a.row(ids[1], true);
      if (s.original) { a.original_row(ids[0], s); a.original_row(ids[1], s); }
    }
    break;
  case T::DeletedRow:
    if (!s.dual || !s.original || ni < 2 || nv < 3) break;
    a.col(ids[1], true); a.row(ids[0], true); a.original_column(ids[1], s); break;
  case T::RowDualShift:
    if (!a.require(ni >= 1 && nv >= 1)) return;
    if (s.dual) a.row(ids[0], true);
    break;
  case T::InfiniteBoundDualFix: {
    if (!a.require(ni >= 3 && nv >= 1 && ids[2] >= 0 && ids[2] <= ni - 3)) return;
    a.col(ids[1], true);
    NnzOffset ip = 3, vp = 1;
    for (int r = 0; r < ids[2]; ++r) {
      if (!a.require(ip < ni && vp < nv)) return;
      const int length = ids[ip];
      if (!a.require(length >= 0 && length < ni - ip && length < nv - vp)) return;
      if (s.primal) for (int p = 0; p < length; ++p) a.col(ids[ip + 1 + p]);
      ip += NnzOffset(length) + 1; vp += NnzOffset(length) + 1;
    }
    a.require(ip == ni && vp == nv); break;
  }
  case T::FmeCol: {
    if (!a.require(ni >= 4 && ids[0] >= 0 && ids[0] <= ni - 4)) return;
    const int rows = ids[0], terms = ids[2 + rows], expressions = ids[3 + rows];
    if (!a.require(terms >= 0 && expressions >= 0 && expressions <= (ni - rows - 4) / 2)) return;
    a.col(ids[1], true);
    if (s.dual) for (int p = 0; p < rows; ++p) a.row(ids[2 + p], true);
    const NnzOffset support = NnzOffset(rows) + 4 + NnzOffset(2) * expressions;
    if (!a.require(ni - support == terms)) return;
    NnzOffset used = 0, values = 0;
    for (int p = 0; p < expressions; ++p) {
      const int length = ids[rows + 5 + NnzOffset(2) * p];
      if (!a.require(length >= 0 && length <= terms - used && values < nv && length <= nv - values - 1)) return;
      used += length; values += NnzOffset(length) + 1;
    }
    if (!a.require(used == terms && values == nv)) return;
    if (s.primal) for (int p = 0; p < terms; ++p) a.col(ids[support + p]);
    break;
  }
  case T::RowLowerBoundChange: case T::RowUpperBoundChange: case T::VariableBoundChange:
    break;
  default:
    // Keep row-implied bound recovery and unknown layouts on the existing interpreter.
    a.require(false);
  }
}
__global__ void inspect(int* invalid, int* owners, int* parent, Tape t,
    NnzOffset begin, int count, State s, bool connect) {
  for (auto p = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       p < std::uint64_t(count); p += std::uint64_t(blockDim.x) * gridDim.x) {
    if (!connect) parent[p] = int(p);
    const auto k = begin + NnzOffset(p), ip = t.is[k], vp = t.vs[k];
    Access a{invalid, owners, owners + s.n, parent, s.m, s.n, int(p), connect};
    accesses(a, t.types[k], t.ids + ip, t.is[k + 1] - ip, t.values + vp, t.vs[k + 1] - vp, s);
  }
}
__global__ void make_keys(Key* keys, const int* parent, int* control, int count, bool collect_statistics) {
  for (auto p = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       p < std::uint64_t(count); p += std::uint64_t(blockDim.x) * gridDim.x) {
    int r = int(p);
    while (parent[r] != r) r = parent[r];
    keys[p] = (Key(r) << 32) | Key(p);
    if (collect_statistics && r == int(p)) atomicAdd(control + 1, 1);
  }
}
struct Plan { Mode mode = Mode::Serial; const int* control = nullptr; const Key* ordered = nullptr; };
struct Context {
  Options config = options();
  Buffer<int> control, targets, owners, parent;
  Buffer<Key> keys, ordered;
  Buffer<std::uint8_t> temporary;
  std::size_t allocated = 0;
  bool fixed = enabled(config.fixed, "GPUPRESOLVER_PARALLEL_FIXED_COLS");
  bool subcol = enabled(config.subcol, "GPUPRESOLVER_PARALLEL_SUBCOL");
  bool components = enabled(config.components, "GPUPRESOLVER_PARALLEL_COMPONENTS");

  template<class T> bool ensure(Buffer<T>& buffer, std::size_t count) {
    if (count <= buffer.capacity) return true;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return false;
    const auto bytes = count * sizeof(T), old_bytes = buffer.capacity * sizeof(T);
    if (bytes > config.scratch_limit_bytes || allocated - old_bytes > config.scratch_limit_bytes - bytes) return false;
    T* next = nullptr;
    const auto status = cudaMalloc(&next, bytes);
    if (status == cudaErrorMemoryAllocation) {
      if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) cudaGetLastError();
      return false;
    }
    check(status);
    if (buffer.data) check(cudaFree(buffer.data));
    buffer.data = next; buffer.capacity = count; allocated = allocated - old_bytes + bytes;
    return true;
  }
  Plan prepare(Tape tape, State state, NnzOffset begin, NnzOffset end) {
    const auto length = end - begin;
    if (length < min_records || length > INT_MAX || state.m < 0 || state.n <= 0 || (!fixed && !subcol && !components)) return {};
    int last = -1;
    check(cudaMemcpy(&last, tape.types + end - 1, sizeof(int), cudaMemcpyDeviceToHost));
    Mode mode = Mode::Components;
    if (last == int(PostsolveReductionType::FixedVariable)) { if (!fixed) return {}; mode = Mode::Fixed; }
    else if (last == int(PostsolveReductionType::SingletonColumnSubstitution)) { if (!subcol) return {}; mode = Mode::Subcol; }
    else if (!components || last == int(PostsolveReductionType::RowImpliedBoundChange)) return {};
    if (!ensure(control, 2)) return {};
    if (mode == Mode::Subcol && !ensure(targets, state.n)) return {};
    if (mode == Mode::Components && (!ensure(owners, std::size_t(state.m) + state.n) ||
        !ensure(parent, length) || !ensure(keys, length) || !ensure(ordered, length))) return {};
    std::size_t bytes = 0;
    if (mode == Mode::Components) {
      check(CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys(nullptr, bytes, keys.data, ordered.data, int(length)));
      if (!ensure(temporary, bytes)) return {};
    }
    check(cudaMemsetAsync(control.data, 0, 2 * sizeof(int)));
    if (mode == Mode::Fixed) {
      certify_fixed<<<blocks(length), 256>>>(control.data, tape, begin, end, state.m, state.n);
    } else if (mode == Mode::Subcol) {
      check(cudaMemsetAsync(targets.data, 0, std::size_t(state.n) * sizeof(int)));
      certify_subcol_targets<<<blocks(length), 256>>>(control.data, targets.data, tape, begin, end, state);
      certify_subcol_support<<<blocks(length), 256>>>(control.data, targets.data, tape, begin, end, state);
    } else {
      check(cudaMemsetAsync(owners.data, 0xff, (std::size_t(state.m) + state.n) * sizeof(int)));
      inspect<<<blocks(length), 256>>>(control.data, owners.data, parent.data, tape, begin, int(length), state, false);
      inspect<<<blocks(length), 256>>>(control.data, owners.data, parent.data, tape, begin, int(length), state, true);
      make_keys<<<blocks(length), 256>>>(keys.data, parent.data, control.data, int(length), config.collect_statistics);
      check(cudaGetLastError());
      check(CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys(temporary.data, bytes, keys.data, ordered.data, int(length)));
    }
    check(cudaGetLastError());
    return {mode, control.data, ordered.data};
  }
  void record(Plan plan, NnzOffset length) {
    if (!config.collect_statistics || length < min_records) return;
    auto& stats = statistics();
    int result[2] = {1, 0};
    if (plan.control) check(cudaMemcpy(result, plan.control, sizeof(result), cudaMemcpyDeviceToHost));
    if (result[0]) { ++stats.fallback_ranges; return; }
    stats.parallel_records += std::size_t(length);
    if (plan.mode == Mode::Fixed) ++stats.fixed_ranges;
    else if (plan.mode == Mode::Subcol) ++stats.subcol_ranges;
    else { ++stats.component_ranges; stats.components += result[1]; }
  }
};

} // namespace gpu_presolver::presolve::postsolve::parallel
