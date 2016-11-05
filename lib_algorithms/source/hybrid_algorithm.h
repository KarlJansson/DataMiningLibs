#pragma once
#include "ml_algorithm.h"

namespace lib_algorithms {
template <typename T>
class HybridAlgorithm : public MlAlgorithm<T> {
 public:
  explicit HybridAlgorithm(sp<MlAlgorithm<T>> gpu_alg,
                           sp<MlAlgorithm<T>> cpu_alg);
  virtual ~HybridAlgorithm();

  sp<lib_models::MlModel> Fit(sp<lib_data::MlDataFrame<T>> data,
                              sp<MlAlgorithmParams> params) override;
  sp<lib_data::MlResultData<T>> Predict(sp<lib_data::MlDataFrame<T>> data,
                                        sp<lib_models::MlModel> model,
                                        sp<MlAlgorithmParams> params) override;

 private:
  sp<MlAlgorithm<T>> cpu_algorithm_;
  sp<MlAlgorithm<T>> device_algorithm_;
  col_map<string, T> algo_params_;
};
}