#include "precomp.h"

#include "ml_model_impl.h"

namespace lib_models {
MlModelImpl::MlModelImpl(sp<MlModelDecorator> decorator)
    : decorator_(decorator) {}

void MlModelImpl::AddData(const int id, const sutil::any_type data) {
  while (id >= data_.size()) data_.emplace_back(sutil::any_type());
  data_[id] = std::move(data);
}

sutil::any_type& MlModelImpl::GetData(const int id) { return data_[id]; }

void MlModelImpl::Aggregate(col_array<sp<lib_models::MlModel>> models) {
  col_array<sp<lib_models::MlModel>> models_new;
  models_new.emplace_back(shared_from_this());
  for (auto& ptr : models) models_new.emplace_back(ptr);
  decorator_->AggregateModels(models_new);
}

col_array<sp<lib_models::MlModel>> MlModelImpl::Split(const int parts) {
  return decorator_->SplitModel(shared_from_this(), parts);
}
}