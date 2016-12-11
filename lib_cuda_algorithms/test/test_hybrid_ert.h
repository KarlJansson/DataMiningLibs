#pragma once
#include "test_resources.h"

namespace test_hybrid_ert {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> hybrid_ert_model_flt;
sp<lib_models::MlModel> hybrid_ert_model_dbl;
auto hybrid_ert_flt = ensembles_face.CreateHybridErt<float>();
auto hybrid_ert_dbl = ensembles_face.CreateHybridErt<double>();

TEST(lib_ensembles_hybrid_ert, fit_double_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_ert_model_dbl = hybrid_ert_dbl->Fit(
                      lib_cuda_algorithms::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_hybrid_ert, predict_double_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results = hybrid_ert_dbl->Predict(
                      lib_cuda_algorithms::data_predict_raw_dbl,
                      hybrid_ert_model_dbl, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_hybrid_ert, fit_float_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_ert_model_flt = hybrid_ert_flt->Fit(
                      lib_cuda_algorithms::data_fit_raw_flt, params););
}

TEST(lib_ensembles_hybrid_ert, predict_float_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results = hybrid_ert_flt->Predict(
                      lib_cuda_algorithms::data_predict_raw_flt,
                      hybrid_ert_model_flt, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_hybrid_ert, fit_float_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_ert_model_flt = hybrid_ert_flt->Fit(
                      lib_cuda_algorithms::data_csv_flt, params););
}

TEST(lib_ensembles_hybrid_ert, predict_float_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      hybrid_ert_flt->Predict(lib_cuda_algorithms::data_csv_flt,
                                              hybrid_ert_model_flt, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_hybrid_ert, fit_double_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_ert_model_dbl = hybrid_ert_dbl->Fit(
                      lib_cuda_algorithms::data_csv_dbl, params););
}

TEST(lib_ensembles_hybrid_ert, predict_double_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results =
                      hybrid_ert_dbl->Predict(lib_cuda_algorithms::data_csv_dbl,
                                              hybrid_ert_model_dbl, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}