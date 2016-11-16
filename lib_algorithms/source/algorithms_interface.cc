#include "precomp.h"

#include "algorithms_interface.h"
#include "cpu_algorithm.h"
#include "gpu_algorithm.h"
#include "hybrid_algorithm.h"
#include "lib_ensembles.h"
#include "ml_algorithm_params_impl.h"

namespace lib_algorithms {
AlgorithmsInterface& AlgorithmsInterface::GetInstance() {
  static AlgorithmsInterface instance;
  return instance;
}

sp<MlAlgorithmParams> AlgorithmsInterface::CreateAlgorithmParams(int size) {
  return std::make_shared<MlAlgorithmParamsImpl>(size);
}

col_array<sp<lib_algorithms::MlAlgorithmParams>>
AlgorithmsInterface::SplitDteParamPack(
    sp<lib_algorithms::MlAlgorithmParams> params, const int parts) {
  col_array<sp<lib_algorithms::MlAlgorithmParams>> part_vec;
  auto orig_trees = params->Get<int>(AlgorithmsLib::kNrTrees);
  auto rest = orig_trees % parts;
  col_array<int> tree_splits(parts, orig_trees / parts);
  for (int i = 0; i < rest; ++i) ++tree_splits[i % parts];

  for (int i = 0; i < parts; ++i) {
    auto params_new = CreateAlgorithmParams(AlgorithmsLib::kDteEndMarker);
    params_new->Set(AlgorithmsLib::kNrFeatures,
                    params->Get<int>(AlgorithmsLib::kNrFeatures));
    params_new->Set(AlgorithmsLib::kMaxDepth,
                    params->Get<int>(AlgorithmsLib::kMaxDepth));
    params_new->Set(AlgorithmsLib::kMinNodeSize,
                    params->Get<int>(AlgorithmsLib::kMinNodeSize));
    params_new->Set(AlgorithmsLib::kAlgoType,
                    params->Get<AlgorithmType>(AlgorithmsLib::kAlgoType));
    params_new->Set(AlgorithmsLib::kBagging,
                    params->Get<bool>(AlgorithmsLib::kBagging));
    params_new->Set(AlgorithmsLib::kEasyEnsemble,
                    params->Get<bool>(AlgorithmsLib::kEasyEnsemble));
    params_new->Set(AlgorithmsLib::kTreeBatchSize,
                    params->Get<int>(AlgorithmsLib::kTreeBatchSize));
    params_new->Set(AlgorithmsLib::kMaxGpuBlocks,
                    params->Get<int>(AlgorithmsLib::kMaxGpuBlocks));
    params_new->Set(AlgorithmsLib::kTreeCounter,
                    params->Get<sp<int>>(AlgorithmsLib::kTreeCounter));
    params_new->Set(AlgorithmsLib::kTreeCounterMutex,
                    params->Get<sp<mutex>>(AlgorithmsLib::kTreeCounterMutex));
    params_new->Set(AlgorithmsLib::kMaxSamplesPerTree,
                    params->Get<int>(AlgorithmsLib::kMaxSamplesPerTree));
    part_vec.emplace_back(params_new);
    part_vec.back()->Set(AlgorithmsLib::kNrTrees, tree_splits[i]);
  }

  return part_vec;
}

template <typename T>
sp<MlAlgorithm<T>> AlgorithmsInterface::CreateCpuAlgorithm(
    sp<MlAlgorithm<T>> algo) {
  return std::make_shared<CpuAlgorithm<T>>(algo);
}

template <typename T>
sp<MlAlgorithm<T>> AlgorithmsInterface::CreateGpuAlgorithm(
    sp<MlAlgorithm<T>> algo) {
  return std::make_shared<GpuAlgorithm<T>>(algo);
}

template <typename T>
sp<MlAlgorithm<T>> AlgorithmsInterface::CreateHybridAlgorithm(
    sp<MlAlgorithm<T>> gpu_algo, sp<MlAlgorithm<T>> cpu_algo) {
  return std::make_shared<HybridAlgorithm<T>>(gpu_algo, cpu_algo);
}

template DLLExport sp<MlAlgorithm<float>>
AlgorithmsInterface::CreateCpuAlgorithm(sp<MlAlgorithm<float>> algo);
template DLLExport sp<MlAlgorithm<double>>
AlgorithmsInterface::CreateCpuAlgorithm(sp<MlAlgorithm<double>> algo);

template DLLExport sp<MlAlgorithm<float>>
AlgorithmsInterface::CreateGpuAlgorithm(sp<MlAlgorithm<float>> algo);
template DLLExport sp<MlAlgorithm<double>>
AlgorithmsInterface::CreateGpuAlgorithm(sp<MlAlgorithm<double>> algo);

template DLLExport sp<MlAlgorithm<float>>
AlgorithmsInterface::CreateHybridAlgorithm(sp<MlAlgorithm<float>> gpu_algo,
                                           sp<MlAlgorithm<float>> cpu_algo);
template DLLExport sp<MlAlgorithm<double>>
AlgorithmsInterface::CreateHybridAlgorithm(sp<MlAlgorithm<double>> gpu_algo,
                                           sp<MlAlgorithm<double>> cpu_algo);
}