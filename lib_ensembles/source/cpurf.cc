#include "precomp.h"

#include "cpurf.h"

namespace lib_ensembles {
template <typename T>
CpuRf<T>::CpuRf() {}

template <typename T>
T CpuRf<T>::GetDistribution(
    col_array<col_array<T>>& dists, col_array<int>& counts, int attribute,
    lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& current_node,
    std::mt19937& att_rng, const col_array<T>& data,
    const col_array<T>& targets, col_array<int>& indices,
    AlgorithmsLib::AlgorithmType type, int nr_samples, int nr_targets) {
  T split_point = 0.0;
  SortOnAttribute(attribute, current_node, indices, data, nr_samples);
  switch (type) {
    case AlgorithmsLib::kClassification:
      split_point = Distribution(
          dists, attribute, indices, current_node.node_index_start,
          current_node.node_index_start + current_node.node_index_count,
          nr_targets, data, targets, nr_samples);
      break;
    case AlgorithmsLib::kRegression:
      split_point = DistributionRegression(
          dists, attribute, indices, current_node.node_index_start,
          current_node.node_index_start + current_node.node_index_count, counts,
          nr_targets, data, targets, nr_samples);
      break;
  }
  return split_point;
}

template <typename T>
T CpuRf<T>::DistributionRegression(col_array<col_array<T>>& dists, int att,
                                   col_array<int>& sortedIndices, int l, int r,
                                   col_array<int>& count, int nr_targets,
                                   const col_array<T>& data,
                                   const col_array<T>& targets,
                                   int nr_samples) {
  T splitPoint = -flt_max;
  col_array<col_array<T>> dist;
  int i;

  col_array<col_array<T>> currDist(2, col_array<T>(nr_targets, 0));
  dist = col_array<col_array<T>>(2, col_array<T>(nr_targets, 0));
  col_array<int> currCounts(2, 0), counts(2, 0);

  // begin with moving all instances into second subset
  for (int ii = l; ii < r; ii++) {
    int inst = sortedIndices[ii];
    currDist[1][0] += targets[inst];
    dist[1][0] += targets[inst];
    ++counts[1];
    ++currCounts[1];
  }

  T currVal = -flt_max;  // current value of splitting criterion
  T bestVal = -flt_max;  // best value of splitting criterion
  int bestI = 0;  // the value of "i" BEFORE which the splitpoint is placed

  for (i = l + 1; i < r; i++) {  // --- try all split points
    int inst = sortedIndices[i];
    int prevInst = sortedIndices[i - 1];

    currDist[0][0] += targets[prevInst];
    currDist[1][0] -= targets[prevInst];
    currCounts[0] += 1;
    currCounts[1] -= 1;

    // do not allow splitting between two instances with the same value
    if (data[att * nr_samples + inst] > data[att * nr_samples + prevInst]) {
      currVal = CpuDte<T>::RegressionResponse(currDist, currCounts);

      if (currVal > bestVal) {
        bestVal = currVal;
        bestI = i;
      }
    }
  }

  if (bestI > 0) {
    int instJustBeforeSplit = sortedIndices[bestI - 1];
    int instJustAfterSplit = sortedIndices[bestI];
    splitPoint = (data[att * nr_samples + instJustAfterSplit] +
                  data[att * nr_samples + instJustBeforeSplit]) /
                 2.0f;

    for (int ii = l; ii < bestI; ii++) {
      int inst = sortedIndices[ii];
      dist[0][0] += targets[inst];
      dist[1][0] -= targets[inst];
      counts[0] += 1;
      counts[1] -= 1;
    }
  }

  // return distribution after split and best split point
  dists = dist;
  count = counts;
  return splitPoint;
}

template <typename T>
T CpuRf<T>::Distribution(col_array<col_array<T>>& dists, int att,
                         col_array<int>& sortedIndices, int l, int r,
                         int nr_targets, const col_array<T>& data,
                         const col_array<T>& targets, int nr_samples) {
  T splitPoint = -flt_max;
  col_array<col_array<T>> dist;
  int i;

  col_array<col_array<T>> currDist(2, col_array<T>(nr_targets, 0));
  dist = col_array<col_array<T>>(2, col_array<T>(nr_targets, 0));

  // begin with moving all instances into second subset
  for (int ii = l; ii < r; ii++) {
    int inst = sortedIndices[ii];
    currDist[1][int(targets[inst])] += 1;
    dist[1][int(targets[inst])] += 1;
  }

  T currVal = -flt_max;  // current value of splitting criterion
  T bestVal = -flt_max;  // best value of splitting criterion
  int bestI = 0;  // the value of "i" BEFORE which the splitpoint is placed

  // Entropy over collumns
  T prior = 0;
  T sumForColumn, total = 0;
  for (int ii = 0; ii < dist[0].size(); ii++) {
    sumForColumn = 0;
    for (i = 0; i < dist.size(); i++) {
      sumForColumn += dist[i][ii];
    }
    prior -= CpuDte<T>::lnFunc(sumForColumn);
    total += sumForColumn;
  }
  prior = (prior + CpuDte<T>::lnFunc(total));

  for (i = l + 1; i < r; i++) {  // --- try all split points
    int inst = sortedIndices[i];
    int prevInst = sortedIndices[i - 1];

    currDist[0][int(targets[prevInst])] += 1;
    currDist[1][int(targets[prevInst])] -= 1;

    // do not allow splitting between two instances with the same value
    if (data[att * nr_samples + inst] > data[att * nr_samples + prevInst]) {
      currVal = 0;
      T sumForBranch;
      for (int branchNum = 0; branchNum < currDist.size(); branchNum++) {
        sumForBranch = 0;
        for (int classNum = 0; classNum < currDist[0].size(); classNum++) {
          currVal = currVal + CpuDte<T>::lnFunc(currDist[branchNum][classNum]);
          sumForBranch += currDist[branchNum][classNum];
        }
        currVal = currVal - CpuDte<T>::lnFunc(sumForBranch);
      }
      currVal = -currVal;

      if (prior - currVal > bestVal) {
        bestVal = prior - currVal;
        bestI = i;
      }
    }
  }

  if (bestI > 0) {
    int instJustBeforeSplit = sortedIndices[bestI - 1];
    int instJustAfterSplit = sortedIndices[bestI];
    splitPoint = (data[att * nr_samples + instJustAfterSplit] +
                  data[att * nr_samples + instJustBeforeSplit]) /
                 2.0f;

    for (int ii = l; ii < bestI; ii++) {
      int inst = sortedIndices[ii];
      dist[0][int(targets[inst])] += 1;
      dist[1][int(targets[inst])] -= 1;
    }
  }

  // return distribution after split and best split point
  dists = dist;
  return splitPoint;
}

template <typename T>
void CpuRf<T>::SortOnAttribute(
    int attribute, lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& node,
    col_array<int>& indices, const col_array<T>& data, int nr_samples) {
  QuickSort(attribute, indices, node.node_index_start,
            node.node_index_start + node.node_index_count - 1, data,
            nr_samples);
}

template <typename T>
void CpuRf<T>::QuickSort(int attribute, col_array<int>& index, int left,
                         int right, const col_array<T>& data, int nr_samples) {
  if (left < right) {
    int middle = Partition(attribute, index, left, right, data, nr_samples);
    QuickSort(attribute, index, left, middle, data, nr_samples);
    QuickSort(attribute, index, middle + 1, right, data, nr_samples);
  }
}

template <typename T>
int CpuRf<T>::Partition(int attribute, col_array<int>& index, int l, int r,
                        const col_array<T>& data, int nr_samples) {
  T pivot = data[attribute * nr_samples + index[(l + r) / 2]];
  int tmp;

  while (l < r) {
    while ((data[attribute * nr_samples + index[l]] < pivot) && (l < r)) {
      l++;
    }
    while ((data[attribute * nr_samples + index[r]] > pivot) && (l < r)) {
      r--;
    }
    if (l < r) {
      tmp = index[l];
      index[l] = index[r];
      index[r] = tmp;

      l++;
      r--;
    }
  }
  if ((l == r) && (data[attribute * nr_samples + index[r]] > pivot)) {
    r--;
  }

  return r;
}

template CpuRf<float>::CpuRf();
template CpuRf<double>::CpuRf();
}
