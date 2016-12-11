#define DLLExport
#define TestExport

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "lib_ensembles.h"
#include "lib_parsing.h"

class LogStreamer {
 public:
  LogStreamer(sp<std::ofstream> dest) : dest_(dest) {}
  template <typename T>
  LogStreamer& operator<<(T const& value) {
    *dest_ << value;
    std::cout << value;
    return *this;
  }

 private:
  sp<std::ofstream> dest_;
};

LogStreamer out_stream(
    std::make_shared<std::ofstream>("./benchmark_output.txt"));

template <typename T>
using dataframe_pair = std::pair<string, sp<lib_data::MlDataFrame<T>>>;
template <typename T>
using dataframe_array = col_array<dataframe_pair<T>>;

template <typename T>
using algorithm_pair = std::pair<sp<lib_algorithms::MlAlgorithm<T>>,
                                 sp<lib_algorithms::MlAlgorithmParams>>;
template <typename T>
using algorithm_map = col_map<std::string, algorithm_pair<T>>;

template <typename T>
void RunBenchmark(algorithm_map<T>& algos, dataframe_array<T>& dfs_train,
                  dataframe_array<T>& dfs_test, int reps) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;

  col_array<T> fit_times;
  col_array<T> pred_times;
  col_array<T> acc_values;
  col_array<T> auc_values;
  col_array<T> node_counts;

  out_stream << "\n"
             << "Fit table:"
             << "\n"
             << "		";
  for (auto& pair : algos) out_stream << pair.first << "&	";
  out_stream << "\n";

  for (auto& data_pair : dfs_train) {
    out_stream << data_pair.first << "&		";
    auto algo_id = 1;
    for (auto& pair : algos) {
      T fit_time = 0;
      for (int i = 0; i < reps + 1; ++i) {
        // Warmup round
        if (i == 0) {
          auto model =
              pair.second.first->Fit(data_pair.second, pair.second.second);
          auto result = pair.second.first->Predict(data_pair.second, model,
                                                   pair.second.second);
          continue;
        }

        start = std::chrono::system_clock::now();
        auto model =
            pair.second.first->Fit(data_pair.second, pair.second.second);
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        fit_times.push_back(T(elapsed_seconds.count()));
        fit_time += fit_times.back();

        start = std::chrono::system_clock::now();
        auto result = pair.second.first->Predict(data_pair.second, model,
                                                 pair.second.second);
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        pred_times.push_back(T(elapsed_seconds.count()));

        acc_values.push_back(result->GetAccuracy());
        auc_values.push_back(result->GetAuc());
        node_counts.push_back(
            T(model
                  ->Get<col_array<lib_algorithms::DteAlgorithmShared::
                                      Dte_NodeHeader_Classify<T>>>(
                      ModelsLib::kNodeArray)
                  .size()));
      }
      auto str = std::to_string(fit_time / T(reps));
      str = str.substr(0, str.find_last_of('.') + 4);
      if (algo_id < algos.size())
        out_stream << str << "&	";
      else
        out_stream << str << "\\\\";
      ++algo_id;
    }
    out_stream << "\n";
  }

  out_stream << "\n"
             << "Predict table:"
             << "\n"
             << "		";
  for (auto& pair : algos) out_stream << pair.first << "&	";
  out_stream << "\n";

  auto model_id = 0;
  for (auto& data_pair : dfs_test) {
    out_stream << data_pair.first << "&		";
    auto algo_id = 1;
    for (auto& pair : algos) {
      T pred_time = 0;
      for (int i = 0; i < reps; ++i) pred_time += pred_times[model_id++];
      auto str = std::to_string(pred_time / T(reps));
      str = str.substr(0, str.find_last_of('.') + 4);
      if (algo_id < algos.size())
        out_stream << str << "&	";
      else
        out_stream << str << "\\\\";
      ++algo_id;
    }
    out_stream << "\n";
  }

  out_stream << "\n"
             << "Accuracy table:"
             << "\n"
             << "		";
  for (auto& pair : algos) out_stream << pair.first << "&	";
  out_stream << "\n";

  auto result_id = 0;
  for (auto& data_pair : dfs_test) {
    out_stream << data_pair.first << "&		";
    auto algo_id = 1;
    for (auto& pair : algos) {
      T acc = 0;
      for (int i = 0; i < reps; ++i) acc += acc_values[result_id++];

      auto str = std::to_string(acc / T(reps));
      str = str.substr(0, str.find_last_of('.') + 4);
      if (algo_id < algos.size())
        out_stream << str << "&	";
      else
        out_stream << str << "\\\\";
      ++algo_id;
    }
    out_stream << "\n";
  }

  out_stream << "\n"
             << "Auc table:"
             << "\n"
             << "		";
  for (auto& pair : algos) out_stream << pair.first << "&	";
  out_stream << "\n";

  result_id = 0;
  for (auto& data_pair : dfs_test) {
    out_stream << data_pair.first << "&		";
    auto algo_id = 1;
    for (auto& pair : algos) {
      T auc = 0;
      for (int i = 0; i < reps; ++i) auc += auc_values[result_id++];

      auto str = std::to_string(auc / T(reps));
      str = str.substr(0, str.find_last_of('.') + 4);
      if (algo_id < algos.size())
        out_stream << str << "&	";
      else
        out_stream << str << "\\\\";
      ++algo_id;
    }
    out_stream << "\n";
  }

  out_stream << "\n"
             << "Nodes per second table:"
             << "\n"
             << "		";
  for (auto& pair : algos) out_stream << pair.first << "&	";
  out_stream << "\n";

  model_id = 0;
  for (auto& data_pair : dfs_test) {
    out_stream << data_pair.first << "&		";
    auto algo_id = 1;
    for (auto& pair : algos) {
      T nodes_per_sec_reps = 0;
      T fit_time = 0;
      for (int i = 0; i < reps; ++i) {
        fit_time += fit_times[model_id];
        nodes_per_sec_reps += node_counts[model_id];
        ++model_id;
      }
      auto str = std::to_string((nodes_per_sec_reps / fit_time) / T(reps));
      str = str.substr(0, str.find_last_of('.'));
      if (algo_id < algos.size())
        out_stream << str << "&	";
      else
        out_stream << str << "\\\\";
      ++algo_id;
    }
    out_stream << "\n";
  }

  out_stream << "\n"
             << "Max/Min value table:"
             << "\n"
             << "		";
  out_stream << "\n";

  model_id = 0;
  for (auto& data_pair : dfs_test) {
    out_stream << data_pair.first << ":\n";
    for (auto& pair : algos) {
      out_stream << pair.first << "	";
      T max[] = {fit_times[model_id], pred_times[model_id],
                 acc_values[model_id], node_counts[model_id]};
      T min[] = {max[0], max[1], max[2], max[3]};

      ++model_id;
      for (int i = 1; i < reps; ++i) {
        max[0] = fit_times[model_id] > max[0] ? fit_times[model_id] : max[0];
        max[1] = pred_times[model_id] > max[1] ? pred_times[model_id] : max[1];
        max[2] = acc_values[model_id] > max[2] ? acc_values[model_id] : max[2];
        max[3] =
            node_counts[model_id] > max[3] ? node_counts[model_id] : max[3];

        min[0] = fit_times[model_id] < min[0] ? fit_times[model_id] : min[0];
        min[1] = pred_times[model_id] < min[1] ? pred_times[model_id] : min[1];
        min[2] = acc_values[model_id] < min[2] ? acc_values[model_id] : min[2];
        min[3] =
            node_counts[model_id] < min[3] ? node_counts[model_id] : min[3];

        ++model_id;
      }

      for (int i = 0; i < 4; ++i) {
        auto str_max = std::to_string(max[i]);
        auto str_min = std::to_string(min[i]);
        str_max = str_max.substr(0, str_max.find_last_of('.') + 4);
        str_min = str_min.substr(0, str_min.find_last_of('.') + 4);
        out_stream << "(" << str_max << "," << str_min << ")";
      }
      out_stream << "\n";
    }
    out_stream << "\n";
  }
}

template <typename T>
void LoadDatasets(dataframe_array<T>& dfs, string data_dir) {
  auto& parsing_lib = ParsingLib::GetInstance();
  for (auto& p :
       std::experimental::filesystem::recursive_directory_iterator(data_dir)) {
    std::stringstream path_stream;
    path_stream << p;
    string name = path_stream.str();
    name = name.substr(name.find_last_of('\\') + 1,
                       name.find_last_of('.') - (name.find_last_of('\\') + 1));
    dfs.emplace_back(dataframe_pair<T>(
        name, parsing_lib.ParseFile<T>(ParsingLib::kCsv, path_stream.str())));
    out_stream << dfs.back().first << "&		"
               << std::to_string(dfs.back().second->GetNrSamples())
               << "&	" << std::to_string(dfs.back().second->GetNrFeatures())
               << "&	" << std::to_string(dfs.back().second->GetNrTargets())
               << "&	" << std::to_string(int(
                                 log(dfs.back().second->GetNrFeatures()) + 1))
               << "\\\\"
               << "\n";
  }
}

template <typename T>
algorithm_map<T> CreateAlgorithms(col_map<string, sutil::any_type>& settings) {
  auto& ens_lib = EnsemblesLib::GetInstance();
  algorithm_map<T> result;

  auto ert_par = ens_lib.CreateErtParamPack();
  auto rf_par = ens_lib.CreateRfParamPack();

  col_map<string, AlgorithmsLib::DteParams> type_map;
  type_map["kNrTrees"] = AlgorithmsLib::kNrTrees;
  type_map["kMaxDepth"] = AlgorithmsLib::kMaxDepth;
  type_map["kTreeBatchSize"] = AlgorithmsLib::kTreeBatchSize;
  type_map["kMaxSamplesPerTree"] = AlgorithmsLib::kMaxSamplesPerTree;
  type_map["kNrFeatures"] = AlgorithmsLib::kNrFeatures;
  type_map["kMinNodeSize"] = AlgorithmsLib::kMinNodeSize;
  type_map["kMaxGpuBlocks"] = AlgorithmsLib::kMaxGpuBlocks;

  for (auto& pair : type_map) {
    auto itr = settings.find(pair.first);
    if (itr != settings.end()) {
      ert_par->Set(pair.second, itr->second.get_value<int>());
      rf_par->Set(pair.second, itr->second.get_value<int>());
    }
  }

  result["CpuErt"] = algorithm_pair<T>(ens_lib.CreateCpuErt<T>(), ert_par);
  result["CpuRf"] = algorithm_pair<T>(ens_lib.CreateCpuRf<T>(), rf_par);
#ifdef Cuda_Found
  result["GpuErt"] = algorithm_pair<T>(ens_lib.CreateGpuErt<T>(), ert_par);
  result["GpuRf"] = algorithm_pair<T>(ens_lib.CreateGpuRf<T>(), rf_par);
  result["HybErt"] = algorithm_pair<T>(ens_lib.CreateHybridErt<T>(), ert_par);
  result["HybRf"] = algorithm_pair<T>(ens_lib.CreateHybridRf<T>(), rf_par);
#endif

  return result;
}

col_map<string, sutil::any_type> ReadSettings(string settings_file_dir) {
  col_map<string, sutil::any_type> settings_map;
  std::ifstream open(settings_file_dir);

  int val_test;
  string line, key, value;
  char junk_chars[] = {'"', ' ', '	', ','};
  while (!open.eof()) {
    std::getline(open, line);
    auto delim_ind = line.find_first_of(':');
    if (delim_ind != string::npos) {
      key = line.substr(0, delim_ind);
      for (int i = 0; i < 4; ++i)
        key.erase(std::remove(key.begin(), key.end(), junk_chars[i]),
                  key.end());

      value = line.substr(delim_ind + 1, line.size());
      for (int i = 0; i < 4; ++i)
        value.erase(std::remove(value.begin(), value.end(), junk_chars[i]),
                    value.end());

      try {
        val_test = std::stoi(value);
        settings_map[key] = sutil::any_type(val_test);
      } catch (...) {
        settings_map[key] = sutil::any_type(value);
      }
    }
  }
  open.close();
  return settings_map;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    out_stream << "Settings file path missing from call."
               << "\n";
    return 0;
  }
  auto settings_map = ReadSettings(argv[1]);
  auto itr = settings_map.find("kRepetitions");
  int reps = itr != settings_map.end() ? itr->second.get_value<int>() : 1;
  itr = settings_map.find("run_single_precision");
  int single_precision =
      itr != settings_map.end() ? itr->second.get_value<int>() : 0;
  itr = settings_map.find("run_double_precision");
  int double_precision =
      itr != settings_map.end() ? itr->second.get_value<int>() : 0;

  auto& ens_lib = EnsemblesLib::GetInstance();
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;
  dataframe_array<float> flt_df_train, flt_df_test;
  dataframe_array<double> dbl_df_train, dbl_df_test;

  if (single_precision == 1) {
    start = std::chrono::system_clock::now();
    out_stream << "Loading Single Precision Datasets:\n";
    out_stream << "Loading fit data...\n";
    LoadDatasets(flt_df_train,
                 settings_map["fit_data_dir"].get_value<string>());
    out_stream << "Loading predict data..."
               << "\n";
    LoadDatasets(flt_df_test,
                 settings_map["predict_data_dir"].get_value<string>());
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    out_stream << "Loading finished in " << elapsed_seconds.count() << "\n";

    auto flt_agorithms = CreateAlgorithms<float>(settings_map);
    start = std::chrono::system_clock::now();
    out_stream
        << "\n"
        << "---------------------------------------------------------------"
           "---------------------------"
        << "\n"
        << "Single Precision Benchmarks"
        << "\n"
        << "---------------------------------------------------------------"
           "---------------------------"
        << "\n";

    RunBenchmark(flt_agorithms, flt_df_train, flt_df_test, reps);

    out_stream
        << "\n"
        << "---------------------------------------------------------------"
           "---------------------------";

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    out_stream << "\n"
               << "Single precision total time: " << elapsed_seconds.count()
               << "\n";
    out_stream
        << "---------------------------------------------------------------"
           "---------------------------"
        << "\n"
        << "\n";
  }

  if (double_precision == 1) {
    start = std::chrono::system_clock::now();
    out_stream << "Loading Double Precision Datasets:"
               << "\n";
    out_stream << "Loading fit data..."
               << "\n";
    LoadDatasets(dbl_df_train,
                 settings_map["fit_data_dir"].get_value<string>());
    out_stream << "Loading predict data..."
               << "\n";
    LoadDatasets(dbl_df_test,
                 settings_map["predict_data_dir"].get_value<string>());
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    out_stream << "Loading finished in " << elapsed_seconds.count() << "\n";

    auto dbl_agorithms = CreateAlgorithms<double>(settings_map);
    start = std::chrono::system_clock::now();
    out_stream
        << "\n"
        << "\n"
        << "---------------------------------------------------------------"
           "---------------------------"
        << "\n"
        << "Double Precision Benchmarks"
        << "\n"
        << "---------------------------------------------------------------"
           "---------------------------"
        << "\n"
        << "\n";

    RunBenchmark(dbl_agorithms, dbl_df_train, dbl_df_test, reps);

    out_stream
        << "\n"
        << "---------------------------------------------------------------"
           "---------------------------";

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    out_stream << "\n"
               << "Double precision total time: " << elapsed_seconds.count()
               << "\n";
    out_stream
        << "---------------------------------------------------------------"
           "---------------------------"
        << "\n";
  }
}
