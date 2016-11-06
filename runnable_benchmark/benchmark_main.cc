#define DLLExport
#define TestExport

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include "lib_ensembles.h"
#include "lib_parsing.h"

template <typename T>
void RunBenchmark(string msg, sp<lib_algorithms::MlAlgorithm<T>> algo,
                  sp<lib_algorithms::MlAlgorithmParams> params,
                  col_array<sp<lib_data::MlDataFrame<T>>>& dfs_train,
                  col_array<sp<lib_data::MlDataFrame<T>>>& dfs_test) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;

  col_array<sp<lib_models::MlModel>> models;
  col_array<T> fit_times;

  std::cout << msg << std::endl << " fit		";
  for (auto data : dfs_train) {
    start = std::chrono::system_clock::now();
    models.emplace_back(algo->Fit(data, params));
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    fit_times.push_back(T(elapsed_seconds.count()));
    auto str = std::to_string(fit_times.back());
    str = str.substr(0, str.find_last_of('.') + 4);
    std::cout << str << "		";
  }

  auto model_id = 0;
  std::cout << std::endl << " pre		";
  col_array<sp<lib_data::MlResultData<T>>> results;
  for (auto data : dfs_test) {
    start = std::chrono::system_clock::now();
    results.emplace_back(algo->Predict(data, models[model_id++], params));
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    auto str = std::to_string(elapsed_seconds.count());
    str = str.substr(0, str.find_last_of('.') + 4);
    std::cout << str << "		";
  }
  std::cout << std::endl << " acc		";

  auto result_id = 0;
  for (auto data : dfs_test) {
    auto str =
        std::to_string(results[result_id++]->GetAccuracy(data->GetTargets()));
    str = str.substr(0, str.find_last_of('.') + 4);
    std::cout << str << "		";
  }
  std::cout << std::endl << " n/s		";

  for (int i = 0; i < models.size(); ++i) {
    T nodes_per_sec = T(models[i]
                            ->Get<col_array<lib_algorithms::DteAlgorithmShared::
                                                Dte_NodeHeader_Classify<T>>>(
                                ModelsLib::kNodeArray)
                            .size());
    nodes_per_sec /= fit_times[i];
    auto str = std::to_string(nodes_per_sec);
    str = str.substr(0, str.find_last_of('.'));
    std::cout << str << "		";
  }
  std::cout << std::endl;
}

template <typename T>
col_array<string> LoadDatasets(col_array<sp<lib_data::MlDataFrame<T>>>& dfs,
                               string data_dir) {
  auto& parsing_lib = ParsingLib::GetInstance();
  col_array<string> names;
  for (auto& p :
       std::experimental::filesystem::recursive_directory_iterator(data_dir)) {
    std::cout << p << std::endl;
    std::stringstream path_stream;
    path_stream << p;
    names.emplace_back(path_stream.str().substr(
        path_stream.str().find_last_of('\\') + 1, path_stream.str().size()));
    dfs.emplace_back(
        parsing_lib.ParseFile<T>(ParsingLib::kCsv, path_stream.str()));
  }
  return names;
}

int main(int argc, char** argv) {
  if (argc < 3) return 0;
  string train_dir = argv[1];
  string test_dir = argv[2];

  auto& ens_lib = EnsemblesLib::GetInstance();
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;
  col_array<sp<lib_data::MlDataFrame<float>>> flt_df_train, flt_df_test;
  col_array<sp<lib_data::MlDataFrame<double>>> dbl_df_train, dbl_df_test;
  start = std::chrono::system_clock::now();
  std::cout << "Loading Single Precision Datasets:" << std::endl;
  std::cout << "Loading fit data..." << std::endl;
  auto dataset_names = LoadDatasets(flt_df_train, train_dir);
  std::cout << "Loading predict data..." << std::endl;
  LoadDatasets(flt_df_test, test_dir);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "Loading finished in " << elapsed_seconds.count() << std::endl;

  auto ert_params = ens_lib.CreateErtParamPack();
  ert_params->Set(AlgorithmsLib::kNrTrees, 100);
  ert_params->Set(AlgorithmsLib::kTreeBatchSize, 20);
  ert_params->Set(AlgorithmsLib::kMaxDepth, 100);
  ert_params->Set(AlgorithmsLib::kMaxSamplesPerTree, 1000);
  ert_params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);

  auto rf_params = ens_lib.CreateRfParamPack();
  rf_params->Set(AlgorithmsLib::kNrTrees, 100);
  rf_params->Set(AlgorithmsLib::kTreeBatchSize, 10);
  rf_params->Set(AlgorithmsLib::kMaxDepth, 100);
  rf_params->Set(AlgorithmsLib::kMaxSamplesPerTree, 1000);
  rf_params->Set(AlgorithmsLib::kAlgoType, AlgorithmsLib::kClassification);

  start = std::chrono::system_clock::now();
  std::cout << std::endl
            << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl
            << "Single Precision Benchmarks" << std::endl
            << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl
            << std::endl
            << "		";

  for (auto& name : dataset_names) std::cout << name << "	";
  std::cout << std::endl;

  RunBenchmark("CpuErt", ens_lib.CreateCpuErt<float>(), ert_params,
               flt_df_train, flt_df_test);
#ifdef Cuda_Found
  RunBenchmark("GpuErt", ens_lib.CreateGpuErt<float>(), ert_params,
               flt_df_train, flt_df_test);
  RunBenchmark("HybErt", ens_lib.CreateHybridErt<float>(), ert_params,
               flt_df_train, flt_df_test);
#endif
  RunBenchmark("CpuRf", ens_lib.CreateCpuRf<float>(), rf_params, flt_df_train,
               flt_df_test);
  RunBenchmark("GpuRf", ens_lib.CreateGpuRf<float>(), rf_params, flt_df_train,
               flt_df_test);
  RunBenchmark("HybRf", ens_lib.CreateHybridRf<float>(), rf_params,
               flt_df_train, flt_df_test);

  std::cout << std::endl
            << "---------------------------------------------------------------"
               "---------------------------";

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << std::endl
            << "Single precision total time: " << elapsed_seconds.count()
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl
            << std::endl;

  start = std::chrono::system_clock::now();
  std::cout << "Loading Double Precision Datasets:" << std::endl;
  std::cout << "Loading fit data..." << std::endl;
  LoadDatasets(dbl_df_train, train_dir);
  std::cout << "Loading predict data..." << std::endl;
  LoadDatasets(dbl_df_test, test_dir);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << "Loading finished in " << elapsed_seconds.count() << std::endl;

  start = std::chrono::system_clock::now();
  std::cout << std::endl
            << std::endl
            << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl
            << "Double Precision Benchmarks" << std::endl
            << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl
            << std::endl
            << "		";

  for (auto& name : dataset_names) std::cout << name << "	";
  std::cout << std::endl;

  RunBenchmark("CpuErt", ens_lib.CreateCpuErt<double>(), ert_params,
               dbl_df_train, dbl_df_test);
  RunBenchmark("GpuErt", ens_lib.CreateGpuErt<double>(), ert_params,
               dbl_df_train, dbl_df_test);
  RunBenchmark("HybErt", ens_lib.CreateHybridErt<double>(), ert_params,
               dbl_df_train, dbl_df_test);

  RunBenchmark("CpuRf", ens_lib.CreateCpuRf<double>(), rf_params, dbl_df_train,
               dbl_df_test);
  RunBenchmark("GpuRf", ens_lib.CreateGpuRf<double>(), rf_params, dbl_df_train,
               dbl_df_test);
  RunBenchmark("HybRf", ens_lib.CreateHybridRf<double>(), rf_params,
               dbl_df_train, dbl_df_test);

  std::cout << std::endl
            << "---------------------------------------------------------------"
               "---------------------------";

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  std::cout << std::endl
            << "Double precision total time: " << elapsed_seconds.count()
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl;
}
