#pragma once
#include "test_resources.h"

namespace test_gpurf {
auto &algorithms_face = AlgorithmsLib::GetInstance();
auto &ensembles_face = EnsemblesLib::GetInstance();

sp<lib_models::MlModel> model_flt;
sp<lib_models::MlModel> model_dbl;
auto gpurf_flt = ensembles_face.CreateGpuRf<float>();
auto gpurf_dbl = ensembles_face.CreateGpuRf<double>();

TEST(lib_ensembles_gpurf, fit_float_rawdata) {
	auto params = ensembles_face.CreateRfParamPack();
	params->Set(AlgorithmsLib::kNrTrees, 100);
	params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
	ASSERT_NO_THROW(model_flt =
		gpurf_flt->Fit(lib_cuda_algorithms::data_fit_raw_flt, params););
}

TEST(lib_ensembles_gpurf, predict_float_rawdata) {
	auto params = ensembles_face.CreateRfParamPack();
	float acc = 0;
	ASSERT_NO_THROW(auto results = gpurf_flt->Predict(
		lib_cuda_algorithms::data_predict_raw_flt, model_flt, params);
	acc = results->GetAccuracy(
		lib_cuda_algorithms::data_predict_raw_flt->GetTargets()););
	std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpurf, fit_float_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(model_flt =
                      gpurf_flt->Fit(lib_cuda_algorithms::data_csv_flt, params););
}

TEST(lib_ensembles_gpurf, predict_float_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  float acc = 0;
  ASSERT_NO_THROW(
      auto results =
          gpurf_flt->Predict(lib_cuda_algorithms::data_csv_flt, model_flt, params);
      acc = results->GetAccuracy(lib_cuda_algorithms::data_csv_flt->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}

/*TEST(lib_ensembles_gpurf, fit_double_rawdata) {
	auto params = ensembles_face.CreateRfParamPack();
	params->Set(AlgorithmsLib::kNrTrees, 200);
	params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
	ASSERT_NO_THROW(model_dbl =
		gpurf_dbl->Fit(lib_ensembles::data_fit_raw_dbl, params););
}

TEST(lib_ensembles_gpurf, predict_double_rawdata) {
	auto params = ensembles_face.CreateRfParamPack();
	double acc = 0;
	ASSERT_NO_THROW(auto results = gpurf_dbl->Predict(
		lib_ensembles::data_predict_raw_dbl, model_dbl, params);
	acc = results->GetAccuracy(
		lib_ensembles::data_predict_raw_dbl->GetTargets()););
	std::cout << "[          ] accuracy = " << acc << std::endl;
}

TEST(lib_ensembles_gpurf, fit_double_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  params->Set(AlgorithmsLib::kNrTrees, 100);
  params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
  ASSERT_NO_THROW(model_dbl =
                      gpurf_dbl->Fit(lib_ensembles::data_csv_dbl, params););
}

TEST(lib_ensembles_gpurf, predict_double_csvdata) {
  auto params = ensembles_face.CreateRfParamPack();
  double acc = 0;
  ASSERT_NO_THROW(
      auto results =
          gpurf_dbl->Predict(lib_ensembles::data_csv_dbl, model_dbl, params);
      acc = results->GetAccuracy(lib_ensembles::data_csv_dbl->GetTargets()););
  std::cout << "[          ] accuracy = " << acc << std::endl;
}*/
}