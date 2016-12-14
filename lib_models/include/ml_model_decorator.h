#pragma once

namespace lib_models {
class MlModel;
class MlModelDecorator {
 public:
  MlModelDecorator() = default;
  virtual ~MlModelDecorator() = default;

  virtual void AggregateModels(col_array<sp<lib_models::MlModel>> models) = 0;
  virtual col_array<sp<lib_models::MlModel>> SplitModel(
      sp<lib_models::MlModel> model, const int parts) = 0;
  virtual void SaveModel(string save_path, sp<lib_models::MlModel> model) = 0;
  virtual void LoadModel(string model_path, sp<lib_models::MlModel> model) = 0;
};
}