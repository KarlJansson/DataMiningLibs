#pragma once
#include "dte_algorithm_shared.h"
#include "lib_algorithms.h"

namespace lib_ensembles {
template <typename T>
class CpuDte : public lib_algorithms::MlAlgorithm<T> {
 public:
  CpuDte();
  virtual ~CpuDte() {}

  sp<lib_models::MlModel> Fit(
      sp<lib_data::MlDataFrame<T>> data,
      sp<lib_algorithms::MlAlgorithmParams> params) override;
  sp<lib_data::MlResultData<T>> Predict(
      sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
      sp<lib_algorithms::MlAlgorithmParams> params) override;

 protected:
  T RegressionResponse(col_array<col_array<T>>& means, col_array<int>& count);
  T ClassificationResponse(col_array<col_array<T>>& dist, bool& priorDone,
                           T& prior);

  virtual T GetDistribution(
      col_array<col_array<T>>& dists, col_array<int>& counts, int attribute,
      lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& current_node,
      std::mt19937& att_rng, const col_array<T>& data,
      const col_array<T>& targets, col_array<int>& indices,
      AlgorithmsLib::AlgorithmType type, int nr_samples, int nr_targets) = 0;

  void calculateOOB();
  void oobThread();

  void calculateFeatureImportance();
  void featureImportanceThread();

  T lnFunc(T num);
  void Seed(int seed, bool bagging, int nr_samples, col_array<int>& indices,
            int total_samples);
};
}
