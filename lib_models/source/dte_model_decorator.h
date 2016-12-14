#pragma once
#include "ml_model_decorator.h"

namespace lib_models {
template <typename T>
class DteModelDecorator : public lib_models::MlModelDecorator {
 public:
  DteModelDecorator();

  void AggregateModels(col_array<sp<lib_models::MlModel>> models) override;
  col_array<sp<lib_models::MlModel>> SplitModel(sp<lib_models::MlModel> model,
                                                const int parts) override;
  void SaveModel(string save_path, sp<lib_models::MlModel> model) override;
  void LoadModel(string model_path, sp<lib_models::MlModel> model) override;
};
}