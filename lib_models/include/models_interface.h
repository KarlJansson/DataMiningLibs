#pragma once

namespace lib_models {
class DLLExport ModelsInterface {
 public:
  static ModelsInterface& GetInstance();

  sp<MlModel> CreateModel(sp<MlModelDecorator> decorator);
  template <typename T>
  sp<lib_models::MlModelDecorator> CreateDteModelDecorator();
  enum DteModel {
    kNrTrees,
    kNrTargets,
    kNrFeatures,
    kModelType,

    kNodeArray,
    kProbArray,

    kDteModelEndMarker
  };

 private:
  ModelsInterface() = default;
  ~ModelsInterface() = default;
};
}