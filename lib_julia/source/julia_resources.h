#pragma once
#include "lib_models.h"
#include "lib_parsing.h"
#include "lib_data.h"

namespace lib_julia {
class JuliaResources {
 public:
  static JuliaResources& get();

  template <typename T>
  int StoreDataset(char* dataset);
  template <typename T>
  int StoreResults(sp<lib_data::MlResultData<T>> result);
  int StoreModel(sp<lib_models::MlModel> model);

  void RemoveDataset(int id);
  void RemoveModel(int id);
  void RemoveResult(int id);

  template <typename T>
  void SaveModel(int model_id, string save_path);
  template <typename T>
  int LoadModel(string model_path);

  template <typename T>
  sp<lib_data::MlDataFrame<T>> GetDataset(int id);
  template <typename T>
  sp<lib_data::MlResultData<T>> GetResult(int id);
  sp<lib_models::MlModel> GetModel(int id);

 private:
  col_umap<int, sutil::any_type> dataframes_;
  col_umap<int, sutil::any_type> results_;
  col_umap<int, sp<lib_models::MlModel>> models_;

  std::atomic<int> rec_id_;

  JuliaResources() : rec_id_(0){}
  ~JuliaResources() {}
};
}