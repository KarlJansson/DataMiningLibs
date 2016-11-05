#include "precomp.h"

#include "algorithms_interface.h"
#include "cpu_algorithm.h"
#include "lib_core.h"
#include "lib_ensembles.h"
#include "lib_gpu.h"
#include "ml_algorithm_params.h"

namespace lib_algorithms {
template <typename T>
inline CpuAlgorithm<T>::CpuAlgorithm(sp<MlAlgorithm<T>> cpu_alg)
    : algorithm_(cpu_alg) {}
template <typename T>
CpuAlgorithm<T>::~CpuAlgorithm() {}

template <typename T>
sp<lib_models::MlModel> CpuAlgorithm<T>::Fit(sp<lib_data::MlDataFrame<T>> data,
                                             sp<MlAlgorithmParams> params) {
  auto tree_counter = params->Get<sp<int>>(AlgorithmsLib::kTreeCounter);
  if (*tree_counter == 0)
    *tree_counter = params->Get<int>(AlgorithmsLib::kNrTrees);

  return algorithm_->Fit(data, params);
}
template <typename T>
sp<lib_data::MlResultData<T>> CpuAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<MlAlgorithmParams> params) {
  auto tree_counter = params->Get<sp<int>>(AlgorithmsLib::kTreeCounter);
  if (*tree_counter == 0)
    *tree_counter = params->Get<int>(AlgorithmsLib::kNrTrees);

  return algorithm_->Predict(data, model, params);
}

template CpuAlgorithm<float>::CpuAlgorithm(sp<MlAlgorithm<float>> gpu_alg);
template CpuAlgorithm<double>::CpuAlgorithm(sp<MlAlgorithm<double>> gpu_alg);
}