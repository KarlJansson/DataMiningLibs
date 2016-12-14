#pragma once
#include "test_resources.h"

namespace test_cpuert {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> cpuert_model_flt;
sp<lib_models::MlModel> cpuert_model_dbl;
auto cpuert_flt = ensembles_face.CreateCpuErt<float>();
auto cpuert_dbl = ensembles_face.CreateCpuErt<double>();

TEST(lib_ensembles_cpuert, fit_double_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpuert_model_dbl = cpuert_dbl->Fit(
                      lib_ensembles::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_cpuert, predict_double_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results =
                      cpuert_dbl->Predict(lib_ensembles::data_predict_raw_dbl,
                                          cpuert_model_dbl, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_cpuert, fit_float_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpuert_model_flt = cpuert_flt->Fit(
                      lib_ensembles::data_fit_raw_flt, params););
}

TEST(lib_ensembles_cpuert, predict_float_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      cpuert_flt->Predict(lib_ensembles::data_predict_raw_flt,
                                          cpuert_model_flt, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_cpuert, fit_float_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpuert_model_flt =
                      cpuert_flt->Fit(lib_ensembles::data_csv_flt, params););
  cpuert_model_flt->SaveModel("./save_test.model");
}

TEST(lib_ensembles_cpuert, predict_float_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  float acc = 0;
  cpuert_model_flt->LoadModel("./save_test.model");
  ASSERT_NO_THROW(auto results = cpuert_flt->Predict(
                      lib_ensembles::data_csv_flt, cpuert_model_flt, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_cpuert, fit_double_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpuert_model_dbl =
                      cpuert_dbl->Fit(lib_ensembles::data_csv_dbl, params););
}

TEST(lib_ensembles_cpuert, predict_double_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results = cpuert_dbl->Predict(
                      lib_ensembles::data_csv_dbl, cpuert_model_dbl, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}