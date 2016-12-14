#pragma once
#include "ml_model.h"
#include "ml_model_decorator.h"

namespace lib_models {
class MlModelImpl : public MlModel,
                    public std::enable_shared_from_this<MlModel> {
 public:
  MlModelImpl() = default;
  MlModelImpl(sp<MlModelDecorator> decorator);

 private:
  void AddData(const int id, const sutil::any_type data) override;
  sutil::any_type& GetData(const int id) override;
  void Aggregate(col_array<sp<lib_models::MlModel>> models) override;
  col_array<sp<lib_models::MlModel>> Split(const int parts) override;
  void SaModel(string save_path) override;
  void LdModel(string model_path) override;

  col_array<sutil::any_type> data_;
  sp<MlModelDecorator> decorator_;
};
}