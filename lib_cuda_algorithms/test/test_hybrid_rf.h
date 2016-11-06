#pragma once
#include "test_resources.h"

namespace test_hybrid_rf {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> hybrid_rf_model_flt;
sp<lib_models::MlModel> hybrid_rf_model_dbl;
auto hybrid_rf_flt = ensembles_face.CreateHybridRf<float>();
auto hybrid_rf_dbl = ensembles_face.CreateHybridRf<double>();

TEST(lib_ensembles_hybrid_rf, fit_double_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_rf_model_dbl = hybrid_rf_dbl->Fit(
                      lib_ensembles::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_hybrid_rf, predict_double_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(
      auto results = hybrid_rf_dbl->Predict(lib_ensembles::data_predict_raw_dbl,
                                            hybrid_rf_model_dbl, params);
      acc = results->GetAccuracy(
          lib_ensembles::data_predict_raw_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_hybrid_rf, fit_float_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_rf_model_flt = hybrid_rf_flt->Fit(
                      lib_cuda_algorithms::data_fit_raw_flt, params););
}

TEST(lib_ensembles_hybrid_rf, predict_float_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(
      auto results =
          hybrid_rf_flt->Predict(lib_cuda_algorithms::data_predict_raw_flt,
                                 hybrid_rf_model_flt, params);
      acc = results->GetAccuracy(
          lib_cuda_algorithms::data_predict_raw_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_hybrid_rf, fit_float_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_rf_model_flt = hybrid_rf_flt->Fit(
                      lib_cuda_algorithms::data_csv_flt, params););
}

TEST(lib_ensembles_hybrid_rf, predict_float_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      hybrid_rf_flt->Predict(lib_cuda_algorithms::data_csv_flt,
                                             hybrid_rf_model_flt, params);
                  acc = results->GetAccuracy(
                      lib_cuda_algorithms::data_csv_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_hybrid_rf, fit_double_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(hybrid_rf_model_dbl =
                      hybrid_rf_dbl->Fit(lib_ensembles::data_csv_dbl, params););
}

TEST(lib_ensembles_hybrid_rf, predict_double_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(
      auto results = hybrid_rf_dbl->Predict(lib_ensembles::data_csv_dbl,
                                            hybrid_rf_model_dbl, params);
      acc = results->GetAccuracy(lib_ensembles::data_csv_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}