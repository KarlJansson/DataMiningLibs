#include "precomp.h"

#include "cuda_algorithms_interface.h"
#include "gpudte_algorithm.h"
#include "gpuert.h"
#include "gpurf.h"

namespace lib_cuda_algorithms {
	CudaAlgorithmsInterface& CudaAlgorithmsInterface::GetInstance() {
  static CudaAlgorithmsInterface instance;
  return instance;
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> CudaAlgorithmsInterface::CreateCudaRf() {
  return std::make_shared<GpuDteAlgorithm<T>>(std::make_shared<GpuRf<T>>());
}

template <typename T>
sp<lib_algorithms::MlAlgorithm<T>> CudaAlgorithmsInterface::CreateCudaErt() {
  return std::make_shared<GpuDteAlgorithm<T>>(std::make_shared<GpuErt<T>>());
}


template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
CudaAlgorithmsInterface::CreateCudaRf();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
CudaAlgorithmsInterface::CreateCudaRf();

template DLLExport sp<lib_algorithms::MlAlgorithm<float>>
CudaAlgorithmsInterface::CreateCudaErt();
template DLLExport sp<lib_algorithms::MlAlgorithm<double>>
CudaAlgorithmsInterface::CreateCudaErt();

}