#pragma once
#include "cpudte.h"

namespace lib_ensembles {
template <typename T>
class CpuRf : public CpuDte<T> {
 public:
  CpuRf();

 private:
  virtual T GetDistribution(
      col_array<col_array<T>>& dists, col_array<int>& counts, int attribute,
      lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& current_node,
      std::mt19937& att_rng, const col_array<T>& data,
      const col_array<T>& targets, col_array<int>& indices,
      AlgorithmsLib::AlgorithmType type, int nr_samples,
      int nr_targets) override;

  T Distribution(col_array<col_array<T>>& dists, int att,
                 col_array<int>& sortedIndices, int l, int r, int nr_targets,
                 const col_array<T>& data, const col_array<T>& targets,
                 int nr_samples);
  T DistributionRegression(col_array<col_array<T>>& dists, int att,
                           col_array<int>& sortedIndices, int l, int r,
                           col_array<int>& count, int nr_targets,
                           const col_array<T>& data,
                           const col_array<T>& targets, int nr_samples);

  int Partition(int attribute, col_array<int>& index, int l, int r,
                const col_array<T>& data, int nr_samples);

  void SortOnAttribute(
      int attribute,
      lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& node,
      col_array<int>& indices, const col_array<T>& data, int nr_samples);
  void QuickSort(int attribute, col_array<int>& index, int left, int right,
                 const col_array<T>& data, int nr_samples);
};
}