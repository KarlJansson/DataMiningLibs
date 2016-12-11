#pragma once
#include "ml_resultdata.h"

namespace lib_data {
template <typename T>
class MlResultDataImpl : public MlResultData<T> {
 public:
  MlResultDataImpl();

  void AddSingleValue(string name, T value) override;
  void AddMultipleValue(string name, col_array<T> vals) override;

  void AddPredictions(col_array<col_array<T>> predictions) override;
  void AddTargets(const col_array<T> &targets) override;

  T GetTarget(int sample) override;
  T GetPrediction(int sample, int target) override;

  col_array<col_array<int>>& GetConfusionMatrix() override;
  T GetAccuracy() override;
  T GetAuc() override;
  T GetMse() override;

  int GetNrTargets() override;
  int GetNrSamples() override;

  T GetSingleValue(string name) override;
  col_array<T>& GetMultipleValue(string name) override;

  string ToString() override;
  MlResultData<T>& operator+=(const MlResultData<T>& rhs) override;

 private:
  col_array<col_array<int>> conf_matrix_;
  col_array<col_array<T>> predictions_;
  col_array<T> targets_;

  col_map<string, T> single_val_;
  col_map<string, col_array<T>> multiple_val_;
};
}