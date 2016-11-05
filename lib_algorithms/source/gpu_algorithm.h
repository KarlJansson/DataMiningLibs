#pragma once
#include "ml_algorithm.h"

namespace lib_algorithms {
template <typename T>
class GpuAlgorithm : public MlAlgorithm<T> {
 public:
  explicit GpuAlgorithm(sp<MlAlgorithm<T>> gpu_alg);
  virtual ~GpuAlgorithm();

  sp<lib_models::MlModel> Fit(sp<lib_data::MlDataFrame<T>> data,
                              sp<MlAlgorithmParams> params) override;
  sp<lib_data::MlResultData<T>> Predict(sp<lib_data::MlDataFrame<T>> data,
                                        sp<lib_models::MlModel> model,
                                        sp<MlAlgorithmParams> params) override;

 private:
  sp<MlAlgorithm<T>> device_algorithm_;
};
}