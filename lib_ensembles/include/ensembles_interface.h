#pragma once
#include "lib_algorithms.h"

namespace lib_ensembles {
class DLLExport EnsemblesInterface {
 public:
  static EnsemblesInterface& GetInstance();

  /**
  * \brief
  *	Create a GpuRf algorithm handle
  *
  * \return
  * ::sp<MlAlgorithm>
  */
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateGpuRf();
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateGpuErt();
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateHybridRf();
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateHybridErt();
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateCpuRf();
  template <typename T>
  sp<lib_algorithms::MlAlgorithm<T>> CreateCpuErt();

  sp<lib_algorithms::MlAlgorithmParams> CreateRfParamPack();
  sp<lib_algorithms::MlAlgorithmParams> CreateErtParamPack();

 private:
  EnsemblesInterface() = default;
  ~EnsemblesInterface() = default;
};
}