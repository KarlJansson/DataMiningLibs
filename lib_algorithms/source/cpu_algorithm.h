#pragma once
#include "ml_algorithm.h"

namespace lib_algorithms {
template <typename T>
class CpuAlgorithm : public MlAlgorithm<T> {
 public:
  explicit CpuAlgorithm(sp<MlAlgorithm<T>> cpu_alg);
  virtual ~CpuAlgorithm();

  sp<lib_models::MlModel> Fit(sp<lib_data::MlDataFrame<T>> data,
                              sp<MlAlgorithmParams> params) override;
  sp<lib_data::MlResultData<T>> Predict(sp<lib_data::MlDataFrame<T>> data,
                                        sp<lib_models::MlModel> model,
                                        sp<MlAlgorithmParams> params) override;

 private:
  sp<MlAlgorithm<T>> algorithm_;
};
}