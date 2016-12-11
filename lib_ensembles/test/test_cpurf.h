#pragma once
#include "test_resources.h"

namespace test_cpurf {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> cpurf_model_flt;
sp<lib_models::MlModel> cpurf_model_dbl;
auto cpurf_flt = ensembles_face.CreateCpuRf<float>();
auto cpurf_dbl = ensembles_face.CreateCpuRf<double>();

TEST(lib_ensembles_cpurf, fit_double_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpurf_model_dbl =
                      cpurf_dbl->Fit(lib_ensembles::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_cpurf, predict_double_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results =
                      cpurf_dbl->Predict(lib_ensembles::data_predict_raw_dbl,
                                         cpurf_model_dbl, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_cpurf, fit_float_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpurf_model_flt =
                      cpurf_flt->Fit(lib_ensembles::data_fit_raw_flt, params););
}

TEST(lib_ensembles_cpurf, predict_float_rawdata) {
  auto params = ensembles_face.CreateRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results =
                      cpurf_flt->Predict(lib_ensembles::data_predict_raw_flt,
                                         cpurf_model_flt, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_cpurf, fit_float_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpurf_model_flt =
                      cpurf_flt->Fit(lib_ensembles::data_csv_flt, params););
}

TEST(lib_ensembles_cpurf, predict_float_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(auto results = cpurf_flt->Predict(lib_ensembles::data_csv_flt,
                                                    cpurf_model_flt, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_cpurf, fit_double_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(cpurf_model_dbl =
                      cpurf_dbl->Fit(lib_ensembles::data_csv_dbl, params););
}

TEST(lib_ensembles_cpurf, predict_double_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(auto results = cpurf_dbl->Predict(lib_ensembles::data_csv_dbl,
                                                    cpurf_model_dbl, params);
                  acc = results->GetAccuracy(););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}
}