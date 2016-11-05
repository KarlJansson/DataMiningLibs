#include "precomp.h"

#include "cpuert.h"

namespace lib_ensembles {
template <typename T>
CpuErt<T>::CpuErt()
    : CpuDte<T>() {}

template <typename T>
T CpuErt<T>::GetDistribution(
    col_array<col_array<T>>& dists, col_array<int>& counts, int attribute,
    lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& current_node,
    std::mt19937& att_rng, const col_array<T>& data,
    const col_array<T>& targets, col_array<int>& indices,
    AlgorithmsLib::AlgorithmType type, int nr_samples, int nr_targets) {
  T split_point;
  int inst_id;
  std::uniform_int_distribution<> instRand(0,
                                           current_node.node_index_count - 1);

  T split = 0;
  for (int i = 0; i < 10; ++i) {
    inst_id = indices[current_node.node_index_start + instRand(att_rng)];
    split += data[nr_samples * attribute + inst_id];
  }
  split_point = split / 10;

  // Calculate distribution
  dists = col_array<col_array<T>>(2, col_array<T>(nr_targets, 0));
  counts = col_array<int>(2, 0);
  for (int i = 0; i < current_node.node_index_count; ++i) {
    inst_id = indices[current_node.node_index_start + i];
    switch (type) {
      case AlgorithmsLib::kClassification:
        ++dists[data[nr_samples * attribute + inst_id] < split_point ? 0 : 1]
               [int(targets[inst_id])];
        break;
      case AlgorithmsLib::kRegression:
        dists[data[nr_samples * attribute + inst_id] < split_point ? 0 : 1]
             [0] += targets[inst_id];
        break;
    }
    ++counts[data[nr_samples * attribute + inst_id] < split_point ? 0 : 1];
  }

  return split_point;
}

template CpuErt<float>::CpuErt();
template CpuErt<double>::CpuErt();
}