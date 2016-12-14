#include "precomp.h"

#include "julia_interface.h"
#include "julia_resources.h"
#include "lib_algorithms.h"
#include "lib_data.h"
#include "lib_ensembles.h"
#include "lib_models.h"
#include "lib_parsing.h"

float get_prediction(int result_id, int sample, int target) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetPrediction(sample, target);
}

float get_target(int result_id, int sample) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetTarget(sample);
}

int get_confusion_matrix(int result_id, int x, int y) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  auto& matrix = result->GetConfusionMatrix();
  return matrix[x][y];
}

int get_nr_targets(int result_id) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetNrTargets();
}

int get_nr_samples(int result_id) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetNrSamples();
}

float get_accuracy(int result_id) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetAccuracy();
}

float get_auc(int result_id) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetAuc();
}

float get_mse(int result_id) {
  auto result = lib_julia::JuliaResources::get().GetResult<float>(result_id);
  return result->GetMse();
}

int load_dataset(char* data) {
  return lib_julia::JuliaResources::get().StoreDataset<float>(data);
}

int load_model(char* model_path) {
  return lib_julia::JuliaResources::get().LoadModel<float>(model_path);
}

void save_model(int model_id, char* save_path) {
  lib_julia::JuliaResources::get().SaveModel<float>(model_id, save_path);
}

void remove_dataset(int id) {
  return lib_julia::JuliaResources::get().RemoveDataset(id);
}

void remove_model(int id) {
  return lib_julia::JuliaResources::get().RemoveModel(id);
}

void remove_result(int id) {
  return lib_julia::JuliaResources::get().RemoveResult(id);
}

template <typename T>
int predict(int dataset, int model, bool classification,
            sp<lib_algorithms::MlAlgorithm<T>> algo,
            sp<lib_algorithms::MlAlgorithmParams> params) {
  params->Set(AlgorithmsLib::kAlgoType, classification
                                            ? AlgorithmsLib::kClassification
                                            : AlgorithmsLib::kRegression);
  auto& rec_face = lib_julia::JuliaResources::get();
  auto data = rec_face.GetDataset<T>(dataset);
  auto m = rec_face.GetModel(model);
  auto result = algo->Predict(data, m, params);
  auto id = rec_face.StoreResults(result);
  return id;
}

template <typename T>
int fit(int dataset, int nr_trees, int max_depth, bool classifiction,
        sp<lib_algorithms::MlAlgorithm<T>> algo,
        sp<lib_algorithms::MlAlgorithmParams> params) {
  params->Set(AlgorithmsLib::kNrTrees, nr_trees);
  params->Set(AlgorithmsLib::kMaxDepth, max_depth);
  params->Set(AlgorithmsLib::kAlgoType, classifiction
                                            ? AlgorithmsLib::kClassification
                                            : AlgorithmsLib::kRegression);

  auto& rec_face = lib_julia::JuliaResources::get();
  auto data = rec_face.GetDataset<T>(dataset);
  auto model = algo->Fit(data, params);
  return rec_face.StoreModel(model);
}

int gpurf_fit(int dataset, int nr_trees, int max_depth, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateRfParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateGpuRf<float>();
  return fit<float>(dataset, nr_trees, max_depth, classification, algo, params);
}

int gpurf_predict(int dataset, int model, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateRfParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateGpuRf<float>();
  return predict<float>(dataset, model, classification, algo, params);
}

int gpuert_fit(int dataset, int nr_trees, int max_depth, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateErtParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateGpuErt<float>();
  return fit<float>(dataset, nr_trees, max_depth, classification, algo, params);
}

int gpuert_predict(int dataset, int model, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateErtParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateGpuErt<float>();
  return predict<float>(dataset, model, classification, algo, params);
}

int cpurf_fit(int dataset, int nr_trees, int max_depth, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateRfParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateCpuRf<float>();
  return fit<float>(dataset, nr_trees, max_depth, classification, algo, params);
}

int cpurf_predict(int dataset, int model, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateRfParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateCpuRf<float>();
  return predict<float>(dataset, model, classification, algo, params);
}

int cpuert_fit(int dataset, int nr_trees, int max_depth, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateErtParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateCpuErt<float>();
  return fit<float>(dataset, nr_trees, max_depth, classification, algo, params);
}

int cpuert_predict(int dataset, int model, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateErtParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateCpuErt<float>();
  return predict<float>(dataset, model, classification, algo, params);
}

int hybridrf_fit(int dataset, int nr_trees, int max_depth,
                 bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateRfParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateHybridRf<float>();
  return fit<float>(dataset, nr_trees, max_depth, classification, algo, params);
}

int hybridrf_predict(int dataset, int model, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateRfParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateHybridRf<float>();
  return predict<float>(dataset, model, classification, algo, params);
}

int hybridert_fit(int dataset, int nr_trees, int max_depth,
                  bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateErtParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateHybridErt<float>();
  return fit<float>(dataset, nr_trees, max_depth, classification, algo, params);
}

int hybridert_predict(int dataset, int model, bool classification) {
  auto params = EnsemblesLib::GetInstance().CreateErtParamPack();
  auto algo = EnsemblesLib::GetInstance().CreateHybridErt<float>();
  return predict<float>(dataset, model, classification, algo, params);
}
