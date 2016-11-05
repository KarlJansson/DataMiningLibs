#pragma once
#include "test_resources.h"

namespace test_gpuert {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> gpuert_model_flt;
sp<lib_models::MlModel> gpuert_model_dbl;
auto gpuert_flt = ensembles_face.CreateGpuErt<float>();
auto gpuert_dbl = ensembles_face.CreateGpuErt<double>();

TEST(lib_ensembles_gpuert, fit_double_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_dbl = gpuert_dbl->Fit(
                      lib_cuda_algorithms::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_gpuert, predict_double_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results =
                      gpuert_dbl->Predict(lib_cuda_algorithms::data_predict_raw_dbl,
                                          gpuert_model_dbl, params);
                  acc = results->GetAccuracy(
					  lib_cuda_algorithms::data_predict_raw_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpuert, fit_float_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_flt = gpuert_flt->Fit(
	  lib_cuda_algorithms::data_fit_raw_flt, params););
}

TEST(lib_ensembles_gpuert, predict_float_rawdata) {
  auto params = ensembles_face.CreateErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      gpuert_flt->Predict(lib_cuda_algorithms::data_predict_raw_flt,
                                          gpuert_model_flt, params);
                  acc = results->GetAccuracy(
					  lib_cuda_algorithms::data_predict_raw_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpuert, fit_float_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_flt =
                      gpuert_flt->Fit(lib_cuda_algorithms::data_csv_flt, params););
}

TEST(lib_ensembles_gpuert, predict_float_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  float acc = 0;
  ASSERT_NO_THROW(
      auto results = gpuert_flt->Predict(lib_cuda_algorithms::data_csv_flt,
                                         gpuert_model_flt, params);
      acc = results->GetAccuracy(lib_cuda_algorithms::data_csv_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpuert, fit_double_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(gpuert_model_dbl =
                      gpuert_dbl->Fit(lib_cuda_algorithms::data_csv_dbl, params););
}

TEST(lib_ensembles_gpuert, predict_double_csvdata) {
  auto params = ensembles_face.CreateErtParamPack();
  double acc = 0;
  ASSERT_NO_THROW(
      auto results = gpuert_dbl->Predict(lib_cuda_algorithms::data_csv_dbl,
                                         gpuert_model_dbl, params);
      acc = results->GetAccuracy(lib_cuda_algorithms::data_csv_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}