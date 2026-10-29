// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "../src/presolve/postsolve/parallel_replay_control.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace gpu_presolver;
using namespace gpu_presolver::presolve;
namespace parallel = gpu_presolver::presolve::postsolve::parallel;
namespace {
void check(cudaError_t error) { if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> T* upload(const std::vector<T>& v) {
  if (v.empty()) return nullptr;
  T* p=nullptr; check(cudaMalloc(&p,v.size()*sizeof(T)));
  check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
template<class T> std::vector<T> download(const T* p, std::size_t n) {
  std::vector<T> v(n); if(n) check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
}
void bytes_equal(const std::vector<double>& a,const std::vector<double>& b) {
  assert(a.size()==b.size()); assert(a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);
}
struct Fixture {
  PresolveRecordGpu record;
  std::vector<double> x,y,z;
  double *xd=nullptr,*yd=nullptr,*zd=nullptr;
  explicit Fixture(int n,int m=0) : x(n),y(m),z(n) {
    record.n0=record.n1=n; record.m0=record.m1=m;
    record.col_red2org.resize(n); record.row_red2org.resize(m);
    std::iota(record.col_red2org.begin(),record.col_red2org.end(),0);
    std::iota(record.row_red2org.begin(),record.row_red2org.end(),0);
    for(int i=0;i<n;++i) { x[i]=1.0+(i%11)*0.125; z[i]=(i%5-2)*0.25; }
    for(int i=0;i<m;++i) y[i]=(i%7-3)*0.5;
  }
  ~Fixture() { cudaFree(xd); cudaFree(yd); cudaFree(zd); }
  void inputs(bool null_duals=false) { xd=upload(x); if(!null_duals) { yd=upload(y); zd=upload(z); } }
  void add(PostsolveReductionType type,std::vector<int> ids,std::vector<double> values) {
    auto& t=record.tape; t.types.push_back(int(type)); t.dual_modes.push_back(std::uint8_t(PostsolveDualMode::Exact));
    t.indices.insert(t.indices.end(),ids.begin(),ids.end()); t.vals.insert(t.vals.end(),values.begin(),values.end());
    t.index_starts.push_back(NnzOffset(t.indices.size())); t.value_starts.push_back(NnzOffset(t.vals.size()));
  }
  void resident(int split=0) {
    record.device=std::make_shared<DeviceRecoveryState>(); auto& d=*record.device;
    d.rows.data=upload(record.row_red2org); d.rows.count=record.m1;
    d.cols.data=upload(record.col_red2org); d.cols.count=record.n1;
    auto& t=record.tape;
    const int total=int(t.types.size()); if(!split) split=total;
    for(int begin=0;begin<total;begin+=split) {
      const int end=std::min(total,begin+split);
      const auto ib=t.index_starts[begin], ie=t.index_starts[end], vb=t.value_starts[begin], ve=t.value_starts[end];
      auto seg=allocate_recovery_segment(end-begin,ie-ib,ve-vb);
      auto copy=[](auto* dst,const auto* src,std::size_t n) { if(n) check(cudaMemcpy(dst,src,n*sizeof(*src),cudaMemcpyHostToDevice)); };
      copy(seg.types,t.types.data()+begin,end-begin); copy(seg.dual_modes,t.dual_modes.data()+begin,end-begin);
      std::vector<NnzOffset> is,vs;
      for(int k=begin;k<=end;++k) { is.push_back(t.index_starts[k]-ib); vs.push_back(t.value_starts[k]-vb); }
      copy(seg.index_starts,is.data(),is.size()); copy(seg.value_starts,vs.data(),vs.size());
      copy(seg.indices,t.indices.data()+ib,ie-ib); copy(seg.vals,t.vals.data()+vb,ve-vb);
      d.records+=seg.record_count; d.indices+=seg.index_count; d.values+=seg.value_count;
      d.segments.push_back(std::move(seg));
    }
  }
};
struct Point { std::vector<double> x,y,z; };
Point recover(Fixture& f, bool optimized, const LPInfoGpu* model=nullptr) {
  parallel::options().fixed=parallel::options().subcol=parallel::options().components=optimized;
  auto p=postsolve_gpu(f.xd,f.yd,f.zd,f.record,model);
  Point result{download(p.x_org,p.n0),download(p.y_org,p.m0),download(p.z_org,p.n0)};
  cudaFree(p.x_org); cudaFree(p.y_org); cudaFree(p.z_org); return result;
}
std::filesystem::path dump_dir;
bool assert_paths=true;
int checked=0;
void paired(const std::string& name,Fixture& f,int path=0,const LPInfoGpu* model=nullptr) {
  parallel::options().collect_statistics=true; parallel::statistics()={};
  const auto before=recover(f,false,model);
  parallel::statistics()={}; const auto after=recover(f,true,model);
  bytes_equal(before.x,after.x); bytes_equal(before.y,after.y); bytes_equal(before.z,after.z);
  const auto stats=parallel::statistics();
  if(assert_paths && path==1) assert(stats.fixed_ranges>0);
  if(assert_paths && path==2) assert(stats.subcol_ranges>0);
  if(assert_paths && path==3) assert(stats.component_ranges>0);
  if(assert_paths && path==-1) assert(stats.parallel_records==0);
  if(f.xd) bytes_equal(f.x,download(f.xd,f.x.size()));
  if(f.yd) bytes_equal(f.y,download(f.yd,f.y.size()));
  if(f.zd) bytes_equal(f.z,download(f.zd,f.z.size()));
  if(!dump_dir.empty()) {
    std::ofstream file(dump_dir/(name+".bin"),std::ios::binary);
    for(const auto* v:{&after.x,&after.y,&after.z}) file.write(reinterpret_cast<const char*>(v->data()),std::streamsize(v->size()*sizeof(double)));
    assert(file.good());
  }
  ++checked;
}
void fixed_cases() {
  for(int variant=0;variant<8;++variant) {
    const int count=variant==1?1023:2048;
    Fixture f(count+2,2);
    for(int i=0;i<count;++i) {
      int col=i;
      if(variant==2) col=count-1-i;
      if(variant==3 && i==count-1) col=i-1;
      const double fixed=i%2?-0.0:0.125;
      f.add(PostsolveReductionType::FixedVariable,{col,0,1,0},{fixed,3.0,1e16,-0.5,-1e16});
    }
    if(variant==4) f.resident();
    if(variant==5) f.resident(1024);
    if(variant==6) parallel::options().scratch_limit_bytes=0;
    f.inputs(variant==7);
    paired("fixed_"+std::to_string(variant),f,(variant==1 || variant==2 || variant==3 || variant==6)?-1:1);
    parallel::options().scratch_limit_bytes=std::numeric_limits<std::size_t>::max();
  }
}
void subcol_cases() {
  for(int variant=0;variant<10;++variant) {
    const int count=2048; Fixture f(count+2,count);
    for(int i=0;i<count;++i) {
      int col=variant==1?count-1-i:i, row=i, support=count;
      if(variant==2 && i==count-1) col=i-1;
      if(variant==3 && i==count-1) row=i-1;
      if(variant==4 && i==0) support=1;
      if(variant==5 && i==count-1) support=0;
      f.add(PostsolveReductionType::SingletonColumnSubstitution,{col,row,3,support,count+1,support},
        {2.0,3.0,-INFINITY,INFINITY,0.5,double(i%2),1e16,0.25,-1e16});
    }
    if(variant==6) f.resident(1024);
    if(variant==7) parallel::options().scratch_limit_bytes=16;
    if(variant==8) {
      PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep step;
      step.splits.push_back({count,count+1,1.0});
      f.record.piecewise_linear_reformulation_absolute_value_recoveries.push_back(step);
      f.record.primal_recovery_timeline.push_back({PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue,0,1024});
    }
    f.inputs(variant==9);
    paired("subcol_"+std::to_string(variant),f,(variant>=2 && variant<=5)||variant==7?-1:2);
    parallel::options().scratch_limit_bytes=std::numeric_limits<std::size_t>::max();
  }
}
void component_cases() {
  for(int variant=0;variant<10;++variant) {
    const int count=2048; Fixture f(2*count+2,count);
    for(int i=0;i<count;++i) {
      const int elim=i, keep=variant==1?(i+1)%count:count+i;
      if(variant==2 && i%2) f.add(PostsolveReductionType::RowDualShift,{i},{0.125});
      else if(variant==3) f.add(PostsolveReductionType::ParallelColumn,{elim,keep},{1.0,0.0,INFINITY,0.0,INFINITY});
      else if(variant==4) f.add(PostsolveReductionType::InfiniteBoundDualFix,{1,elim,1,2,elim,keep},{0.0,1.0,1.0,-1.0});
      else if(variant==5) f.add(PostsolveReductionType::FmeCol,{1,elim,i,1,1,1,1,keep},{-0.25,0.5});
      else f.add(PostsolveReductionType::DoubletonEquation,{elim,keep,i},{2.0,-1.0,3.0,-INFINITY,INFINITY,0.0,INFINITY,0.0,INFINITY,0.5});
    }
    if(variant==6) f.resident(1024);
    if(variant==7) parallel::options().scratch_limit_bytes=8+4*(f.record.m0+f.record.n0)+20*count;
    if(variant==8) {
      // Unsupported row-implied bound records must retain whole-range serial replay.
      f.add(PostsolveReductionType::RowImpliedBoundChange,{0,0},{0,1,0,0.5});
    }
    f.inputs(variant==9);
    paired("components_"+std::to_string(variant),f,(variant==7 || variant==8)?-1:3);
    if(assert_paths && variant==1) assert(parallel::statistics().components==2);
    parallel::options().scratch_limit_bytes=std::numeric_limits<std::size_t>::max();
  }
}
void original_model_cases() {
  const int count=2048;
  // Doubleton records read a shared original row: all their
  // row writes must belong to the same component in the dual pass.
  Fixture f(2*count,count);
  std::vector<NnzOffset> ap(count+1),atp(2*count+1);
  std::vector<int> ac(count),atc(count,0); std::vector<double> av(count,0.25);
  for(int i=1;i<=count;++i) ap[i]=count;
  std::iota(ac.begin(),ac.end(),0);
  for(int i=0;i<2*count;++i) atp[i+1]=std::min(i+1,count);
  LPInfoGpu model;
  model.A={count,2*count,count,upload(ap),upload(ac),upload(av)};
  model.AT={2*count,count,count,upload(atp),upload(atc),upload(av)};
  model.c=upload(std::vector<double>(2*count,1.0));
  model.l=upload(std::vector<double>(2*count,0.0)); model.u=upload(std::vector<double>(2*count,INFINITY));
  for(int i=0;i<count;++i)
    f.add(PostsolveReductionType::DoubletonEquation,{i,count+i,i},{1,-1,0,-INFINITY,INFINITY,0,INFINITY,0,INFINITY,1});
  f.inputs(); paired("original_at",f,3,&model);
  // Deleted-row recovery reads all original column support, even when a longer
  // diagnostic payload is present; it cannot use the private format's shortcut.
  Fixture deleted(2*count,count);
  for(int i=0;i<count;++i) deleted.add(PostsolveReductionType::DeletedRow,{i,i},{0,INFINITY,1,0,0,0,0,0,0});
  deleted.inputs(); paired("deleted_original_at",deleted,3,&model);
  Fixture rows(2*count,count);
  for(int i=0;i<count;++i) rows.add(PostsolveReductionType::ParallelRow,{i,(i+1)%count},{1,0,INFINITY,-INFINITY,1});
  rows.inputs(); paired("parallel_rows_original_a",rows,3,&model);
  cudaFree(model.A.rowPtr); cudaFree(model.A.colVal); cudaFree(model.A.nzVal);
  cudaFree(model.AT.rowPtr); cudaFree(model.AT.colVal); cudaFree(model.AT.nzVal);
  cudaFree(model.c); cudaFree(model.l); cudaFree(model.u);
}

}
int main(int argc,char** argv) {
  if(argc==3 && std::string(argv[1])=="--dump-output") {
    dump_dir=argv[2]; std::filesystem::create_directories(dump_dir); assert_paths=false;
  }
  fixed_cases(); subcol_cases(); component_cases(); original_model_cases();
  std::cout<<"parallel_postsolve paired_cases="<<checked<<" bitwise_equal=all\n";
  return 0;
}
