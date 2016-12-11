#include "precomp.h"

#include "ml_resultdata_impl.h"

namespace lib_data {
template <typename T>
MlResultDataImpl<T>::MlResultDataImpl() {}
template <typename T>
inline void MlResultDataImpl<T>::AddSingleValue(string name, T value) {
  single_val_[name] = value;
}
template <typename T>
inline void MlResultDataImpl<T>::AddMultipleValue(string name,
                                                  col_array<T> vals) {
  multiple_val_[name] = std::move(vals);
}
template <typename T>
void MlResultDataImpl<T>::AddPredictions(col_array<col_array<T>> predictions) {
  if (predictions_.empty()) predictions_ = std::move(predictions);
}

template <typename T>
void MlResultDataImpl<T>::AddTargets(const col_array<T> &targets) {
  targets_ = targets;
}

template <typename T>
T MlResultDataImpl<T>::GetTarget(int sample) {
  return targets_[sample];
}

template <typename T>
T MlResultDataImpl<T>::GetPrediction(int sample, int target) {
  return predictions_[sample][target];
}

template <typename T>
col_array<col_array<int>> &MlResultDataImpl<T>::GetConfusionMatrix() {
  if (!conf_matrix_.empty()) return conf_matrix_;

  conf_matrix_ = col_array<col_array<int>>(
      predictions_[0].size(), col_array<int>(predictions_[0].size(), 0));

  for (int i = 0; i < targets_.size(); ++i) {
    T high = 0;
    int target = 0;
    for (int ii = 0; ii < predictions_[i].size(); ++ii) {
      if (predictions_[i][ii] > high) {
        high = predictions_[i][ii];
        target = ii;
      }
    }
    if (std::abs(T(target) - targets_[i]) < 0.001)
      ++conf_matrix_[target][target];
    else
      ++conf_matrix_[target][int(targets_[i])];
  }

  return conf_matrix_;
}

template <typename T>
T MlResultDataImpl<T>::GetAccuracy() {
  T acc = 0;
  for (int i = 0; i < targets_.size(); ++i) {
    T high = 0;
    T target = 0;
    for (int ii = 0; ii < predictions_[i].size(); ++ii) {
      if (predictions_[i][ii] > high) {
        high = predictions_[i][ii];
        target = T(ii);
      }
    }
    if (std::abs(target - targets_[i]) < 0.001) acc += 1;
  }
  return acc / targets_.size();
}

template <typename T>
T MlResultDataImpl<T>::GetAuc() {
  col_array<col_map<T, col_array<long long>, std::greater<T>>> rankings;
  col_array<int> class_marks;
  for (int i = 0; i < predictions_.size(); ++i) {
    for (int ii = 0; ii < predictions_[i].size(); ++ii) {
      rankings[i][predictions_[i][ii]].emplace_back(ii == int(targets_[i]) ? 1
                                                                           : 0);
      class_marks[int(targets_[i])] = 1;
    }
  }

  col_array<long long> truePositive, falsePositive;
  col_array<T> aucs(rankings.size(), -1);
  int tpAccum, fpAccum;
  for (int i = 0; i < rankings.size(); ++i) {
    if (!class_marks[i]) continue;
    auto auc_iter = rankings[i].begin();
    truePositive.clear();
    falsePositive.clear();
    tpAccum = fpAccum = 0;
    while (auc_iter != rankings[i].end()) {
      for (int ii = 0; ii < auc_iter->second.size(); ++ii) {
        if (auc_iter->second[ii] == 1)
          ++tpAccum;
        else
          ++fpAccum;

        truePositive.emplace_back(tpAccum);
        falsePositive.emplace_back(fpAccum);
      }
      ++auc_iter;
    }

    aucs[i] = 0;
    if (truePositive[truePositive.size() - 1] == 0)
      aucs[i] = 0;
    else if (falsePositive[falsePositive.size() - 1] == 0)
      aucs[i] = 1;
    else {
      T cumNeg = 0.0, cip, cin;
      for (int x = int(truePositive.size()) - 1; x >= 0; --x) {
        if (x > 0) {
          cip = T(truePositive[x] - truePositive[x - 1]);
          cin = T(falsePositive[x] - falsePositive[x - 1]);
        } else {
          cip = T(truePositive[0]);
          cin = T(falsePositive[0]);
        }
        aucs[i] += cip * (cumNeg + (0.5f * cin));
        cumNeg += cin;
      }
      aucs[i] /= T(truePositive[truePositive.size() - 1] *
                   falsePositive[falsePositive.size() - 1]);
    }
  }

  T auc = 0;
  int divisor = 0;
  for (auto c : aucs) {
    if (c != -1) {
      auc += c;
      ++divisor;
    }
  }
  auc /= T(divisor);

  return auc;
}

template <typename T>
T MlResultDataImpl<T>::GetMse() {
  T mse = 0;
  for (int i = 0; i < predictions_.size(); ++i)
    mse += pow(targets_[i] - predictions_[i][0], 2);
  return mse / predictions_.size();
}

template <typename T>
int MlResultDataImpl<T>::GetNrTargets() {
  return int(predictions_[0].size());
}

template <typename T>
int MlResultDataImpl<T>::GetNrSamples() {
  return int(predictions_.size());
}

template <typename T>
T MlResultDataImpl<T>::GetSingleValue(string name) {
  return single_val_[name];
}

template <typename T>
col_array<T> &MlResultDataImpl<T>::GetMultipleValue(string name) {
  return multiple_val_[name];
}

template <typename T>
string MlResultDataImpl<T>::ToString() {
  string result = "";
  for (auto &pair : single_val_)
    result += pair.first + std::to_string(pair.second) + "\r\n";
  for (auto &pair : multiple_val_) {
    result += pair.first + "[";
    for (auto &val : pair.second) result += std::to_string(val) + ",";
    result += "]\r\n";
  }
  return result;
}

template <typename T>
MlResultData<T> &MlResultDataImpl<T>::operator+=(const MlResultData<T> &rhs) {
  auto rhs_ref = static_cast<const MlResultDataImpl<T> *>(&rhs);
  for (int i = 0; i < predictions_.size(); ++i) {
    for (int ii = 0; ii < predictions_[i].size(); ++ii) {
      predictions_[i][ii] += rhs_ref->predictions_[i][ii];
    }
  }
  return *this;
}

template MlResultDataImpl<float>::MlResultDataImpl();
template MlResultDataImpl<double>::MlResultDataImpl();
}