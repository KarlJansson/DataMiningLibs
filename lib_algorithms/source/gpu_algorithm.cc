#include "precomp.h"

#include "algorithms_interface.h"
#include "gpu_algorithm.h"
#include "lib_core.h"
#include "lib_ensembles.h"
#include "lib_gpu.h"
#include "ml_algorithm_params.h"

namespace lib_algorithms {
template <typename T>
inline GpuAlgorithm<T>::GpuAlgorithm(sp<MlAlgorithm<T>> gpu_alg)
    : device_algorithm_(gpu_alg) {}
template <typename T>
GpuAlgorithm<T>::~GpuAlgorithm() {}

template <typename T>
sp<lib_models::MlModel> GpuAlgorithm<T>::Fit(sp<lib_data::MlDataFrame<T>> data,
                                             sp<MlAlgorithmParams> params) {
  auto device = GpuLib::GetInstance().CreateGpuDevice(0);
  auto dev_count = device->GetDeviceCount();
  auto tree_counter = params->Get<sp<int>>(AlgorithmsLib::kTreeCounter);
  if (*tree_counter == 0)
    *tree_counter = params->Get<int>(AlgorithmsLib::kNrTrees);
  auto param_vec =
      AlgorithmsLib::GetInstance().SplitDteParamPack(params, dev_count);
  col_array<sp<lib_models::MlModel>> models(dev_count,
                                            sp<lib_models::MlModel>());
  auto run_func = [&](int i) {
    param_vec[i]->Set(AlgorithmsLib::kDevId, i);
    models[i] = device_algorithm_->Fit(data, param_vec[i]);
  };
  CoreLib::GetInstance().TBBParallelFor(0, dev_count, run_func);
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
sp<lib_data::MlResultData<T>> GpuAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<MlAlgorithmParams> params) {
  auto device = GpuLib::GetInstance().CreateGpuDevice(0);
  auto dev_count = device->GetDeviceCount();
  auto param_vec =
      AlgorithmsLib::GetInstance().SplitDteParamPack(params, dev_count);
  auto model_vec = model->SplitModel(dev_count);
  col_array<sp<lib_data::MlResultData<T>>> results(
      dev_count, sp<lib_data::MlResultData<T>>());
  auto run_func = [&](int i) {
    param_vec[i]->Set(AlgorithmsLib::kDevId, i);
    results[i] = device_algorithm_->Predict(data, model_vec[i], param_vec[i]);
  };
  CoreLib::GetInstance().TBBParallelFor(0, dev_count, run_func);
  for (int i = 1; i < results.size(); ++i) *results[0] += *results[i];
  results[0]->AddTargets(data->GetTargets());
  return results[0];
}

template GpuAlgorithm<float>::GpuAlgorithm(sp<MlAlgorithm<float>> gpu_alg);
template GpuAlgorithm<double>::GpuAlgorithm(sp<MlAlgorithm<double>> gpu_alg);
}