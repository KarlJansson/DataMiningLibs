#include "precomp.h"

#include "lib_core.h"
#include "lib_ensembles.h"
#include "lib_models.h"

#include "cpudte.h"

namespace lib_ensembles {
template <typename T>
CpuDte<T>::CpuDte() {}

template <typename T>
sp<lib_models::MlModel> CpuDte<T>::Fit(
    sp<lib_data::MlDataFrame<T>> data,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  const auto nr_total_trees = params->Get<int>(AlgorithmsLib::kNrTrees);
  auto algo_type =
      params->Get<AlgorithmsLib::AlgorithmType>(AlgorithmsLib::kAlgoType);
  const auto min_node_size = params->Get<int>(AlgorithmsLib::kMinNodeSize);
  const auto max_depth = params->Get<int>(AlgorithmsLib::kMaxDepth);
  const auto nr_samples = data->GetNrSamples();
  const auto max_samples_per_tree =
      nr_samples < params->Get<int>(AlgorithmsLib::kMaxSamplesPerTree)
          ? nr_samples
          : params->Get<int>(AlgorithmsLib::kMaxSamplesPerTree);
  const auto nr_fit_samples =
	  max_samples_per_tree <= 0 ? nr_samples : max_samples_per_tree;
  const auto batch_size = 1;  // params->Get<int>(EnsemblesLib::kTreeBatchSize);
  const auto nr_features = data->GetNrFeatures();
  const auto nr_targets = data->GetNrTargets();
  bool bagging = params->Get<bool>(AlgorithmsLib::kBagging);
  int trees_built = 0;
  auto k = params->Get<int>(AlgorithmsLib::kNrFeatures);
  k = k > 0 ? k : int(std::round(log(nr_features))) + 1;

  const auto& data_samples = data->GetSamples();
  const auto& target_data = data->GetTargets();

  int threads = std::thread::hardware_concurrency();

  auto init_train_node_func =
      [](lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>& node) {
        node.attribute = -1;
        node.node_index_count = -1;
        node.node_index_start = -1;
        node.parent_id = -1;
        node.split_point = -1;
        node.tracking_id = -1;
      };
  auto init_predict_node_func =
      [](lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>& node) {
        node.attribute = -1;
        node.child_count = -1;
        node.child_start = -1;
        node.probability_start = -1;
        node.split_point = -1;
      };

  col_array<sp<lib_models::MlModel>> models;
  auto decorator = ModelsLib::GetInstance().CreateDteModelDecorator<T>();
  for (int i = 0; i < threads; ++i)
    models.emplace_back(ModelsLib::GetInstance().CreateModel(decorator));

  auto tree_id_counter = params->Get<sp<int>>(AlgorithmsLib::kTreeCounter);
  auto tree_id_counter_mutex =
      params->Get<sp<mutex>>(AlgorithmsLib::kTreeCounterMutex);
  auto build_tree_func = [&](int id) {
    int trees_left = 0, nr_trees_built = 0, batch = 0;
    models[id]->Add(ModelsLib::kNrTargets, nr_targets);
    models[id]->Add(ModelsLib::kNrFeatures, nr_features);
    models[id]->Add(ModelsLib::kModelType, algo_type);
    models[id]->Add(ModelsLib::kNrTrees, 0);

    col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>
        tree_nodes, root_nodes;
    col_array<T> tree_probabilities;
    tree_nodes.reserve(nr_fit_samples);
    root_nodes.reserve(nr_fit_samples);
    tree_probabilities.reserve(nr_fit_samples);

    std::uniform_int_distribution<> att_rand(0, nr_features - 1);
    std::mt19937 att_rng;
    col_array<
        col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>>>
        node_queue(
            2,
            col_array<
                lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>>());
    node_queue[0].reserve(nr_fit_samples);
    node_queue[1].reserve(nr_fit_samples);

    while (true) {
      {
        mutex_lock(tree_id_counter_mutex);
        trees_left = *tree_id_counter;
        if (trees_left <= 0) break;
        if (trees_left > batch_size) {
          batch = batch_size;
          *tree_id_counter -= batch_size;
        } else {
          batch = trees_left;
          *tree_id_counter = 0;
        }
      }

      for (int i = 0; i < batch; ++i) {
        att_rng.seed(trees_left - i);
        col_array<col_array<int>> tree_indices(
            2, col_array<int>(nr_fit_samples, 0));
        Seed(trees_left - i, bagging, nr_fit_samples, tree_indices[0],
             nr_samples);

        col_array<col_array<T>> dist, best_dist;
        col_array<int> counts, best_counts;

        bool sensible_split = false, prior_done;
        T prior;
        T best_val = -1000;
        int attribute;
        T split_point;
        int iii;

        int buffer_id = 0;
        int depth = 0;
        int num_leafs = 0, num_nodes = 0;

        lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T> root_train;
        init_train_node_func(root_train);
        root_train.node_index_count = nr_fit_samples;
        root_train.node_index_start = 0;
        root_train.tracking_id = int(root_nodes.size());
        node_queue[buffer_id].emplace_back(root_train);
        lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T> root;
        init_predict_node_func(root);
        root_nodes.emplace_back(root);

        while (!node_queue[buffer_id].empty()) {
          // Iterate through unprocessed nodes
          while (!node_queue[buffer_id].empty()) {
            auto current_node = node_queue[buffer_id].back();
            node_queue[buffer_id].pop_back();

            ++num_nodes;
            auto save_leaf_func = [&]() {
              lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>*
                  current;
              if (current_node.parent_id == -1)
                current = &root_nodes[current_node.tracking_id];
              else
                current = &tree_nodes[current_node.tracking_id];

              current->probability_start = int(tree_probabilities.size());
              current->attribute = -1;
              current->child_count = 0;

              auto dist = col_array<T>(nr_targets, 0);
              for (int i = 0; i < current_node.node_index_count; ++i) {
                ++dist[int(
                    target_data[tree_indices[buffer_id]
                                            [current_node.node_index_start +
                                             i]])];
              }

              for (int i = 0; i < dist.size(); ++i) {
                tree_probabilities.emplace_back(dist[i]);
              }
            };

            auto process_node_func = [&]() -> int {
              // Check if node is a leaf
              if ((current_node.node_index_count > 0 &&
                   current_node.node_index_count <
                       (2 > min_node_size ? 2 : min_node_size))  // small
                  || ((max_depth > 0) && (depth >= max_depth))   // deep
                  ) {
                save_leaf_func();
              } else {
                prior_done = false;
                iii = k;
                sensible_split = false;
                best_val = -1000;
                int num_iter = 0;
                while (num_iter < nr_features && (iii > 0 || !sensible_split)) {
                  ++num_iter;
                  attribute = att_rand(att_rng);

                  split_point = GetDistribution(
                      dist, counts, attribute, current_node, att_rng,
                      data_samples, target_data, tree_indices[buffer_id],
                      algo_type, nr_samples, nr_targets);

                  T response = 0;
                  switch (algo_type) {
                    case AlgorithmsLib::kClassification:
                      response =
                          ClassificationResponse(dist, prior_done, prior);
                      break;
                    case AlgorithmsLib::kRegression:
                      response = RegressionResponse(dist, counts);
                      break;
                  }

                  if (best_val < response || iii == k) {
                    best_val = response;
                    current_node.attribute = attribute;
                    current_node.split_point = split_point;
                    best_dist = dist;
                    best_counts = counts;
                  }

                  if (response > 1e-2) {
                    if (algo_type == AlgorithmsLib::kClassification)
                      if (best_dist[0][0] + best_dist[0][1] == 0 ||
                          best_dist[1][0] + best_dist[1][1] == 0)
                        continue;
                      else if (algo_type == AlgorithmsLib::kRegression)
                        if (best_counts[0] == 0 || best_counts[1] == 0)
                          continue;
                    sensible_split = true;
                  }
                  --iii;
                }

                if (sensible_split) {
                  // Split node
                  int cpy_buffer = (buffer_id == 0 ? 1 : 0);
                  int inst_id;
                  int nr_instances = 0;
                  col_array<T> probs;
                  if (algo_type == AlgorithmsLib::kClassification) {
                    for (int ii = 0; ii < best_dist[0].size(); ++ii) {
                      nr_instances += int(best_dist[0][ii]);
                    }
                  } else
                    nr_instances = best_counts[0];

                  lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<
                      T>* current;
                  if (current_node.parent_id == -1)
                    current = &root_nodes[current_node.tracking_id];
                  else
                    current = &tree_nodes[current_node.tracking_id];

                  current->attribute = current_node.attribute;
                  current->split_point = current_node.split_point;
                  current->child_start = int(tree_nodes.size());
                  current->child_count = 2;

                  lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>
                      child1, child2;
                  init_train_node_func(child1);
                  init_train_node_func(child2);
                  child1.parent_id = current_node.tracking_id;
                  child1.node_index_start = current_node.node_index_start;
                  child1.node_index_count = nr_instances;

                  child2.parent_id = current_node.tracking_id;
                  child2.node_index_start =
                      current_node.node_index_start + nr_instances;

                  nr_instances = 0;
                  if (algo_type == AlgorithmsLib::kClassification) {
                    for (int ii = 0; ii < best_dist[0].size(); ++ii) {
                      nr_instances += int(best_dist[1][ii]);
                    }
                  } else
                    nr_instances = best_counts[1];

                  child2.node_index_count = nr_instances;

                  int left = current_node.node_index_start,
                      right = current_node.node_index_start +
                              current_node.node_index_count - 1;
                  for (int i = 0; i < current_node.node_index_count; ++i) {
                    inst_id = tree_indices[buffer_id]
                                          [current_node.node_index_start + i];
                    if (data_samples[current_node.attribute * nr_samples +
                                     inst_id] < current_node.split_point) {
                      tree_indices[cpy_buffer][left] = inst_id;
                      ++left;
                    } else {
                      tree_indices[cpy_buffer][right] = inst_id;
                      --right;
                    }
                  }

                  lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>
                      child1_clf, child2_clf;
                  init_predict_node_func(child1_clf);
                  init_predict_node_func(child2_clf);

                  child1.tracking_id = int(tree_nodes.size());
                  tree_nodes.emplace_back(child1_clf);
                  child2.tracking_id = int(tree_nodes.size());
                  tree_nodes.emplace_back(child2_clf);

                  node_queue[cpy_buffer].emplace_back(child1);
                  node_queue[cpy_buffer].emplace_back(child2);
                  return 2;
                } else {
                  save_leaf_func();
                }
              }
              return 0;
            };

            int created_nodes = process_node_func();
            if (created_nodes == 0) ++num_leafs;
          }

          buffer_id = (buffer_id == 0 ? 1 : 0);

          depth++;
        }

        ++nr_trees_built;
      }
    }

    auto root_offset = int(root_nodes.size());
    root_nodes.reserve(tree_nodes.size() + root_nodes.size());
    for (auto& node : root_nodes) node.child_start += root_offset;
    for (auto& node : tree_nodes) {
      root_nodes.emplace_back(node);
      root_nodes.back().child_start += root_offset;
    }

    models[id]->Add(ModelsLib::kNrTrees, nr_trees_built);
    models[id]->Add(ModelsLib::kNodeArray, root_nodes);
    models[id]->Add(ModelsLib::kProbArray, tree_probabilities);
  };

  CoreLib::GetInstance().TBBParallelFor(0, threads, build_tree_func);
  auto model = models.empty() ? nullptr : models[0];
  col_array<sp<lib_models::MlModel>> merge_models;
  for (int i = 1; i < models.size(); ++i) {
    if (!model)
      model = models[i];
    else if (models[i])
      merge_models.emplace_back(models[i]);
  }
  if (!merge_models.empty()) model->Merge(merge_models);
  return model;
}

template <typename T>
sp<lib_data::MlResultData<T>> CpuDte<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  const auto nr_total_trees = params->Get<int>(AlgorithmsLib::kNrTrees);
  const auto algo_type =
      params->Get<AlgorithmsLib::AlgorithmType>(AlgorithmsLib::kAlgoType);
  const auto min_node_size = params->Get<int>(AlgorithmsLib::kMinNodeSize);
  const auto max_depth = params->Get<int>(AlgorithmsLib::kMaxDepth);
  const auto nr_samples = data->GetNrSamples();
  const auto nr_features = data->GetNrFeatures();
  const auto nr_targets = data->GetNrTargets();

  const auto& data_samples = data->GetSamples();
  const auto& nodes = model->Get<col_array<
      lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>>(
      ModelsLib::kNodeArray);
  const auto& probabilities = model->Get<col_array<T>>(ModelsLib::kProbArray);

  col_array<T> votes(nr_targets * nr_samples, 0);
  mutex vote_mutex;

  std::atomic<int> tree_id_counter;
  tree_id_counter = nr_total_trees;
  auto predict_tree_func = [&](int id) {
    int tree_id;
    tree_id = --tree_id_counter;

    while (tree_id >= 0) {
      col_array<T> predictions(nr_samples, 0);
      for (int i = 0; i < nr_samples; ++i) {
        auto* node = &nodes[tree_id];
        while (node->child_count != 0 && node->child_start != -1 &&
               node->attribute != -4) {
          if (data_samples[node->attribute * nr_samples + i] <
              node->split_point)
            node = &nodes[node->child_start];
          else
            node = &nodes[node->child_start + 1];
        }

        int classPred = 0;
        switch (algo_type) {
          case AlgorithmsLib::kClassification: {
            T maxCount = 0;
            for (int ii = 0; ii < nr_targets; ++ii) {
              if (maxCount < probabilities[node->probability_start + ii]) {
                maxCount = probabilities[node->probability_start + ii];
                classPred = ii;
              }
            }
            predictions[i] = T(classPred);
            break;
          }
          case AlgorithmsLib::kRegression:
            predictions[i] = probabilities[node->probability_start];
            break;
        }
      }

      mutex_lock lock(vote_mutex);
      for (int i = 0; i < predictions.size(); ++i) {
        switch (algo_type) {
          case AlgorithmsLib::kClassification:
            votes[nr_targets * i + int(predictions[i])] += 1;
            break;
          case AlgorithmsLib::kRegression:
            votes[i] += predictions[i];
            break;
        }
      }
      tree_id = --tree_id_counter;
    }
  };

  CoreLib::GetInstance().TBBParallelFor(0, nr_total_trees, predict_tree_func);

  col_array<col_array<T>> votes_result(nr_samples, col_array<T>());
  int voteCount;
  int classPred;
  for (int i = 0; i < nr_samples; i++) {
    voteCount = 0;
    for (int ii = 0; ii < nr_targets; ++ii) {
      if (algo_type == AlgorithmsLib::kRegression) {
        votes_result[i].emplace_back(votes[nr_targets * i + ii] /
                                     T(nr_total_trees));
      } else {
        votes_result[i].emplace_back(votes[nr_targets * i + ii]);
        if (votes_result[i].back() > voteCount) {
          voteCount = int(votes_result[i].back());
          classPred = ii;
        }
      }
    }
  }
  auto result_data = DataLib::GetInstance().CreateResultData<T>();
  result_data->AddPredictions(votes_result);
  return result_data;
}

template <typename T>
T CpuDte<T>::RegressionResponse(col_array<col_array<T>>& means,
                                col_array<int>& count) {
  T result;
  if (count[0] == 0 || count[1] == 0)
    result = -flt_max;
  else {
    col_array<int> counts(2, 0);
    counts[0] = count[0] == 0 ? 1 : count[0];
    counts[1] = count[1] == 0 ? 1 : count[1];

    means[0][0] /= counts[0];
    means[1][0] /= counts[1];

    T diff = ((means[0][0]) - (means[1][0]));
    result = (counts[0] * counts[1] * diff * diff / (counts[0] + counts[1]));
  }
  return result;
}

template <typename T>
T CpuDte<T>::ClassificationResponse(col_array<col_array<T>>& dist,
                                    bool& priorDone, T& prior) {
  if (!priorDone) {  // needs to be computed only once per branch
    // Entropy over collumns
    prior = 0;
    T sumForColumn, total = 0;
    for (int ii = 0; ii < dist[0].size(); ii++) {
      sumForColumn = 0;
      for (int i = 0; i < dist.size(); i++) {
        sumForColumn += dist[i][ii];
      }
      prior -= lnFunc(sumForColumn);
      total += sumForColumn;
    }
    prior = (prior + lnFunc(total));

    priorDone = true;
  }

  // Entropy over rows
  T posterior = 0;
  T sumForBranch;
  for (int branchNum = 0; branchNum < dist.size(); branchNum++) {
    sumForBranch = 0;
    for (int classNum = 0; classNum < dist[0].size(); classNum++) {
      posterior = posterior + lnFunc(dist[branchNum][classNum]);
      sumForBranch += dist[branchNum][classNum];
    }
    posterior = posterior - lnFunc(sumForBranch);
  }
  posterior = -posterior;
  return prior - posterior;
}

template <typename T>
void CpuDte<T>::Seed(int seed, bool bagging, int nr_samples,
                     col_array<int>& indices, int total_samples) {
  if (bagging) {
    std::mt19937 rng;
    rng.seed(seed);
    int rand_ind;
    std::uniform_int_distribution<> ind_rand(0, total_samples - 1);
    for (int i = 0; i < nr_samples; ++i) {
      rand_ind = ind_rand(rng);
      indices[i] = rand_ind;
    }
  } else {
    if (nr_samples == total_samples)
      for (int i = 0; i < nr_samples; ++i) indices[i] = i;
    else {
      col_array<int> all_indices;
      all_indices.reserve(total_samples);
      for (int i = 0; i < total_samples; ++i) all_indices.push_back(i);

      std::mt19937 rng;
      rng.seed(seed);
      int rand_ind;
      for (int i = 0; i < nr_samples; ++i) {
        std::uniform_int_distribution<> ind_rand(0, all_indices.size() - 1);
        rand_ind = ind_rand(rng);
        indices[i] = all_indices[rand_ind];
        all_indices[rand_ind] = all_indices.back();
        all_indices.pop_back();
      }
    }
  }
}

template <typename T>
T CpuDte<T>::lnFunc(T num) {
  if (num <= 1e-6) {
    return 0;
  } else {
    return num * log(num);
  }
}

template CpuDte<float>::CpuDte();
template CpuDte<double>::CpuDte();
}
