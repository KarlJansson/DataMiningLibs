#include "precomp.h"

#include "cpuert.h"
#include "cpurf.h"
#include "ensembles_interface.h"
#ifdef Cuda_Found
#include "lib_cuda_algorithms.h"
#endif

namespace lib_ensembles {
EnsemblesInterface& EnsemblesInterface::GetInstance() {
  static EnsemblesInterface instance;
  return instance;
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateGpuRf() {
  auto algo = CudaAlgorithmsLib::GetInstance().CreateCudaRf<T>();
  return AlgorithmsLib::GetInstance().CreateGpuAlgorithm<T>(algo);
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateGpuErt() {
  auto algo = CudaAlgorithmsLib::GetInstance().CreateCudaErt<T>();
  return AlgorithmsLib::GetInstance().CreateGpuAlgorithm<T>(algo);
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateHybridRf() {
  auto gpu_algo = CudaAlgorithmsLib::GetInstance().CreateCudaRf<T>();
  return AlgorithmsLib::GetInstance().CreateHybridAlgorithm<T>(
      gpu_algo, CreateCpuRf<T>());
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateHybridErt() {
	auto gpu_algo = CudaAlgorithmsLib::GetInstance().CreateCudaErt<T>();
	return AlgorithmsLib::GetInstance().CreateHybridAlgorithm<T>(
		gpu_algo, CreateCpuErt<T>());
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateCpuRf() {
  return AlgorithmsLib::GetInstance().CreateCpuAlgorithm<T>(
      std::make_shared<CpuRf<T>>());
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> EnsemblesInterface::CreateCpuErt() {
  return AlgorithmsLib::GetInstance().CreateCpuAlgorithm<T>(
      std::make_shared<CpuErt<T>>());
}

sp<lib_algorithms::MlAlgorithmParams> EnsemblesInterface::CreateRfParamPack() {
  auto params = AlgorithmsLib::GetInstance().CreateAlgorithmParams(
      AlgorithmsLib::kDteEndMarker);

  // Set default parameters for random forest
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kNrFeatures, 0);
  params->Set(AlgorithmsLib::kMaxSamplesPerTree, 0);
  params->Set(AlgorithmsLib::kMaxDepth, 0);
  params->Set(AlgorithmsLib::kMinNodeSize, 10);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  params->Set(AlgorithmsLib::kBagging, true);
  params->Set(AlgorithmsLib::kEasyEnsemble, false);
  params->Set(AlgorithmsLib::kTreeBatchSize, 100);
  params->Set(AlgorithmsLib::kMaxGpuBlocks, 10000);
  params->Set(AlgorithmsLib::kTreeCounter, std::make_shared<int>(0));
  params->Set(AlgorithmsLib::kTreeCounterMutex, std::make_shared<mutex>());
  return params;
}

sp<lib_algorithms::MlAlgorithmParams> EnsemblesInterface::CreateErtParamPack() {
  auto params = AlgorithmsLib::GetInstance().CreateAlgorithmParams(
      AlgorithmsLib::kDteEndMarker);

  // Set default parameters for random forest
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kNrFeatures, 0);
  params->Set(AlgorithmsLib::kMaxSamplesPerTree, 0);
  params->Set(AlgorithmsLib::kMaxDepth, 0);
  params->Set(AlgorithmsLib::kMinNodeSize, 10);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  params->Set(AlgorithmsLib::kBagging, false);
  params->Set(AlgorithmsLib::kEasyEnsemble, false);
  params->Set(AlgorithmsLib::kTreeBatchSize, 100);
  params->Set(AlgorithmsLib::kMaxGpuBlocks, 10000);
  params->Set(AlgorithmsLib::kTreeCounter, std::make_shared<int>(0));
  params->Set(AlgorithmsLib::kTreeCounterMutex, std::make_shared<mutex>());
  return params;
}

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateCpuErt();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateCpuErt();

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateCpuRf();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateCpuRf();

#ifdef Cuda_Found
template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateGpuRf();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateGpuRf();

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateGpuErt();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateGpuErt();

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateHybridErt();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateHybridErt();

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
EnsemblesInterface::CreateHybridRf();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
EnsemblesInterface::CreateHybridRf();
#endif
}