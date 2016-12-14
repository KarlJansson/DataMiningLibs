#include "precomp.h"

#include "julia_resources.h"
#include "lib_preprocess.h"

namespace lib_julia {
JuliaResources& JuliaResources::get() {
  static JuliaResources instance;
  return instance;
}

template <typename T>
int JuliaResources::StoreDataset(char* dataset) {
  auto& parse_face = ParsingLib::GetInstance();
  auto& preprocess = PreprocessLib::GetInstance();
  int id = ++rec_id_;
  auto preprocess_doc = preprocess.CreatePreprocessDocument();
  preprocess_doc->LoadDocument(dataset);
  for (int i = 0; i < preprocess_doc->NrFeatures(); ++i)
    if (i == preprocess_doc->TargetCol())
      preprocess_doc->TargetifyAttribute(i);
    else
      preprocess_doc->NumberfyAttribute(i);
  auto buff = preprocess_doc->GetModifiedString();
  auto dframe = parse_face.ParseData<T>(ParsingLib::kCsv, buff.str().c_str());
  dataframes_[id] = sutil::any_type(dframe);
  return id;
}

template <typename T>
int JuliaResources::StoreResults(sp<lib_data::MlResultData<T>> result) {
  int id = ++rec_id_;
  results_[id] = sutil::any_type(result);
  return id;
}

template <typename T>
void JuliaResources::SaveModel(int model_id, string save_path) {
  models_[model_id]->SaveModel(save_path);
}

template <typename T>
int JuliaResources::LoadModel(string model_path) {
  auto& model_face = ModelsLib::GetInstance();
  auto model = model_face.CreateModel(model_face.CreateDteModelDecorator<T>());
  model->LoadModel(model_path);
  return StoreModel(model);
}

template <typename T>
sp<lib_data::MlDataFrame<T>> JuliaResources::GetDataset(int id) {
  return dataframes_[id].get_value<sp<lib_data::MlDataFrame<T>>>();
}

template <typename T>
sp<lib_data::MlResultData<T>> JuliaResources::GetResult(int id) {
  return results_[id].get_value<sp<lib_data::MlResultData<T>>>();
}

int JuliaResources::StoreModel(sp<lib_models::MlModel> model) {
  int id = ++rec_id_;
  models_[id] = model;
  return id;
}

void JuliaResources::RemoveDataset(int id) { dataframes_.erase(id); }

void JuliaResources::RemoveModel(int id) { models_.erase(id); }

void JuliaResources::RemoveResult(int id) { results_.erase(id); }

sp<lib_models::MlModel> JuliaResources::GetModel(int id) { return models_[id]; }

template int JuliaResources::StoreDataset<float>(char* dataset);
template int JuliaResources::StoreDataset<double>(char* dataset);

template sp<lib_data::MlDataFrame<float>> JuliaResources::GetDataset(
    int id);
template sp<lib_data::MlDataFrame<double>>
JuliaResources::GetDataset(int id);

template int JuliaResources::StoreResults(
    sp<lib_data::MlResultData<float>> result);
template int JuliaResources::StoreResults(
    sp<lib_data::MlResultData<double>> result);

template sp<lib_data::MlResultData<float>> JuliaResources::GetResult(
    int id);
template sp<lib_data::MlResultData<double>>
JuliaResources::GetResult(int id);

template void JuliaResources::SaveModel<float>(int model_id, string save_path);
template void JuliaResources::SaveModel<double>(int model_id, string save_path);

template int JuliaResources::LoadModel<float>(string model_path);
template int JuliaResources::LoadModel<double>(string model_path);
}