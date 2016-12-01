#include "precomp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iterator>

#include "julia_interface.h"

// run on single file
float** experiment_1(char* data, int kNrTrees, int kMaxDepth, int kAlgoType,
                     bool kBagging) {
  float** myResults;
  try {
    auto& algorithms_face = AlgorithmsLib::GetInstance();
    auto& ensembles_face = EnsemblesLib::GetInstance();

    sp<lib_models::MlModel> model_flt;
    sp<lib_models::MlModel> model_dbl;
    auto gpurf_flt = ensembles_face.CreateGpuRf<float>();
    auto gpurf_dbl = ensembles_face.CreateGpuRf<double>();
    auto& parser_face = ParsingLib::GetInstance();
    auto params = ensembles_face.CreateRfParamPack();

    params->Set(AlgorithmsLib::kNrTrees, kNrTrees);
    params->Set(AlgorithmsLib::kMaxDepth, kMaxDepth);
    params->Set(AlgorithmsLib::kBagging, kBagging);

    if (kAlgoType == 0) {
      params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
    } else if (kAlgoType == 1) {
      params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kRegression);
    }

    auto data_fit_raw_flt = parser_face.ParseStream<float>(
        lib_parsing::ParsingInterface::kCsv, data);

    auto data_predict_raw_flt = parser_face.ParseStream<float>(
        lib_parsing::ParsingInterface::kCsv, data);

    model_flt = gpurf_flt->Fit(data_fit_raw_flt, params);
    auto results = gpurf_flt->Predict(data_predict_raw_flt, model_flt, params);

    col_array<col_array<float>> myPredictions = results->GetPredictions_();
    int rows = myPredictions.size() + 1;
    int col = myPredictions[0].size();

    myResults = new float*[rows];

    myResults[0] = new float[3];
    myResults[0][0] = rows;
    myResults[0][1] = col;
    if (kAlgoType == 0) {
      myResults[0][2] =
          results->GetAccuracy(data_predict_raw_flt->GetTargets());
    }

    for (int i = 1; i < rows; i++) {
      myResults[i] = new float[col];
      for (int j = 0; j < col; j++) {
        myResults[i][j] = myPredictions[i - 1][j];
      }
    }

  } catch (...) {
    myResults = new float*[1];
    myResults[0] = new float[1];
    myResults[0][0] = -1;
  }
  return myResults;
}

/**
run on 2 files
data1: training file (fit)
data2: test file (predict)
*/
float** experiment_2(char* data1, char* data2, int kNrTrees, int kMaxDepth,
                     int kAlgoType, bool kBagging) {
  float** myResults;
  try {
    auto& algorithms_face = AlgorithmsLib::GetInstance();
    auto& ensembles_face = EnsemblesLib::GetInstance();

    sp<lib_models::MlModel> model_flt;
    sp<lib_models::MlModel> model_dbl;
    auto gpurf_flt = ensembles_face.CreateGpuRf<float>();
    auto gpurf_dbl = ensembles_face.CreateGpuRf<double>();
    auto& parser_face = ParsingLib::GetInstance();
    auto params = ensembles_face.CreateRfParamPack();

    params->Set(AlgorithmsLib::kNrTrees, kNrTrees);
    params->Set(AlgorithmsLib::kMaxDepth, kMaxDepth);
    params->Set(AlgorithmsLib::kBagging, kBagging);

    if (kAlgoType == 0) {
      params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);
    } else if (kAlgoType == 1) {
      params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kRegression);
    }

    auto data_fit_raw_flt = parser_face.ParseStream<float>(
        lib_parsing::ParsingInterface::kCsv, data1);

    auto data_predict_raw_flt = parser_face.ParseStream<float>(
        lib_parsing::ParsingInterface::kCsv, data2);

    model_flt = gpurf_flt->Fit(data_fit_raw_flt, params);
    auto results = gpurf_flt->Predict(data_predict_raw_flt, model_flt, params);

    col_array<col_array<float>> myPredictions = results->GetPredictions_();
    int rows = myPredictions.size() + 1;
    int col = myPredictions[0].size();

    myResults = new float*[rows];

    myResults[0] = new float[3];
    myResults[0][0] = rows;
    myResults[0][1] = col;
    if (kAlgoType == 0) {
      myResults[0][2] =
          results->GetAccuracy(data_predict_raw_flt->GetTargets());
    }

    for (int i = 1; i < rows; i++) {
      myResults[i] = new float[col];
      for (int j = 0; j < col; j++) {
        myResults[i][j] = myPredictions[i - 1][j];
      }
    }

  } catch (...) {
    myResults = new float*[1];
    myResults[0] = new float[1];
    myResults[0][0] = -1;
  }
  return myResults;
}

/*
Free the data when results have been transferred to julia. 
*/
void free_memory(float** data, int rows) {
  for (int i = 0; i < rows; ++i) delete[] data[i];
  delete[] data;
}