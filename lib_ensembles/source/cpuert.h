#pragma once
#include "cpudte.h"

namespace lib_ensembles {
template <typename T>
class CpuErt : public CpuDte<T> {
 public:
  CpuErt();

 private:
  virtual T GetDistribution(
      col_array<col_array<T>>& dists, col_array<int>& counts, int attribute,
	  lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& current_node,
      std::mt19937& att_rng, const col_array<T>& data,
      const col_array<T>& targets, col_array<int>& indices,
      AlgorithmsLib::AlgorithmType type, int nr_samples,
      int nr_targets) override;
};
}