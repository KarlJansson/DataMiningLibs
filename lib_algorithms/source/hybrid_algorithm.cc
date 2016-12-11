#include "precomp.h"

#include "hybrid_algorithm.h"

#include "algorithms_interface.h"
#include "lib_core.h"
#include "lib_ensembles.h"
#include "lib_gpu.h"
#include "ml_algorithm_params.h"

namespace lib_algorithms {
template <typename T>
HybridAlgorithm<T>::HybridAlgorithm(sp<MlAlgorithm<T>> gpu_alg,
                                    sp<MlAlgorithm<T>> cpu_alg)
    : device_algorithm_(gpu_alg), cpu_algorithm_(cpu_alg) {}

template <typename T>
HybridAlgorithm<T>::~HybridAlgorithm() {}

template <typename T>
sp<lib_models::MlModel> HybridAlgorithm<T>::Fit(
    sp<lib_data::MlDataFrame<T>> data, sp<MlAlgorithmParams> params) {
  auto device = GpuLib::GetInstance().CreateGpuDevice(0);
  auto dev_count = device->GetDeviceCount();
  auto tree_counter = params->Get<sp<int>>(AlgorithmsLib::kTreeCounter);
  if (*tree_counter == 0)
    *tree_counter = params->Get<int>(AlgorithmsLib::kNrTrees);
  auto param_vec =
      AlgorithmsLib::GetInstance().SplitDteParamPack(params, dev_count + 1);
  col_array<sp<lib_models::MlModel>> models(dev_count + 1,
                                            sp<lib_models::MlModel>());

  auto run_func = [&](int i) {
    if (i < dev_count) {
      param_vec[i]->Set(AlgorithmsLib::kDevId, i);
      models[i] = device_algorithm_->Fit(data, param_vec[i]);
    } else
      models[i] = cpu_algorithm_->Fit(data, param_vec[i]);
  };
  CoreLib::GetInstance().TBBParallelFor(0, dev_count + 1, run_func);
  auto model = models.empty() ? nullptr : models[0];
  col_array<sp<lib_models::MlModel>> merge_models;
  for (int i = 1; i < models.size(); ++i) {
    if (!model)
      model = models[i];
    else if (models[i])
      merge_models.emplace_back(models[i]);
  }
  if (!merge_models.empty()) model->Merge(merge_models);
  return model;
}

template <typename T>
sp<lib_data::MlResultData<T>> HybridAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<MlAlgorithmParams> params) {
  auto device = GpuLib::GetInstance().CreateGpuDevice(0);
  auto dev_count = device->GetDeviceCount();
  auto param_vec =
      AlgorithmsLib::GetInstance().SplitDteParamPack(params, dev_count + 1);
  auto model_vec = model->SplitModel(dev_count + 1);
  col_array<sp<lib_data::MlResultData<T>>> results(
      dev_count + 1, sp<lib_data::MlResultData<T>>());
  auto run_func = [&](int i) {
    if (i < dev_count) {
      param_vec[i]->Set(AlgorithmsLib::kDevId, i);
      results[i] = device_algorithm_->Predict(data, model_vec[i], param_vec[i]);
    } else
      results[i] = cpu_algorithm_->Predict(data, model_vec[i], param_vec[i]);
  };
  CoreLib::GetInstance().TBBParallelFor(0, dev_count + 1, run_func);
  for (int i = 1; i < results.size(); ++i) *results[0] += *results[i];
  results[0]->AddTargets(data->GetTargets());
  return results[0];
}

template HybridAlgorithm<float>::HybridAlgorithm(
    sp<MlAlgorithm<float>> gpu_alg, sp<MlAlgorithm<float>> cpu_alg);
template HybridAlgorithm<double>::HybridAlgorithm(
    sp<MlAlgorithm<double>> gpu_alg, sp<MlAlgorithm<double>> cpu_alg);
}