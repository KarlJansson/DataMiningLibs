#pragma once
namespace sutil {
struct any_type;
}

namespace lib_models {
class DLLExport MlModel {
 public:
  virtual ~MlModel() = default;

  template <typename T>
  void Add(const int id, const T& data) {
    AddData(id, sutil::any_type(data));
  }

  template <typename T>
  T& Get(const int id) {
    return GetData(id).get_value<T>();
  }

  void Merge(col_array<sp<lib_models::MlModel>> models) { Aggregate(models); }
  col_array<sp<lib_models::MlModel>> SplitModel(int parts) {
    return Split(parts);
  }

 private:
  virtual void AddData(const int id, const sutil::any_type data) = 0;
  virtual sutil::any_type& GetData(const int id) = 0;
  virtual void Aggregate(col_array<sp<lib_models::MlModel>> models) = 0;
  virtual col_array<sp<lib_models::MlModel>> Split(const int parts) = 0;
};
}