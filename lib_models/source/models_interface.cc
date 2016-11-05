#include "precomp.h"

#include "dte_model_decorator.h"
#include "ml_model_impl.h"
#include "models_interface.h"

namespace lib_models {
ModelsInterface& ModelsInterface::GetInstance() {
  static ModelsInterface instance;
  return instance;
}

sp<MlModel> ModelsInterface::CreateModel(sp<MlModelDecorator> decorator) {
  return std::make_shared<MlModelImpl>(decorator);
}

template <typename T>
sp<lib_models::MlModelDecorator> ModelsInterface::CreateDteModelDecorator() {
  return std::make_shared<DteModelDecorator<T>>();
}

template DLLExport sp<lib_models::MlModelDecorator>
ModelsInterface::CreateDteModelDecorator<float>();
template DLLExport sp<lib_models::MlModelDecorator>
ModelsInterface::CreateDteModelDecorator<double>();
}