#pragma once
#include "../../lib_algorithms/include/lib_algorithms.h"

namespace lib_cuda_algorithms {
class DLLExport CudaAlgorithmsInterface {
 public:
  static CudaAlgorithmsInterface& GetInstance();

  /**
  * \brief
  *	Create a GpuRf algorithm handle
  *
  * \return
  * ::sp<MlAlgorithm>
  */
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateCudaRf();
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateCudaErt();

 private:
  CudaAlgorithmsInterface() = default;
  ~CudaAlgorithmsInterface() = default;
};
}