#include "precomp.h"

#include "lib_algorithms.h"
#include "lib_models.h"

#include "dte_model_decorator.h"

namespace lib_models {
template <typename T>
DteModelDecorator<T>::DteModelDecorator() {}

template <typename T>
void DteModelDecorator<T>::AggregateModels(
    col_array<sp<lib_models::MlModel>> models) {
  if (models.size() < 2) return;

  auto node_size = 0, prob_size = 0;
  for (int i = 0; i < models.size(); ++i) {
	  node_size += models[i]
		  ->Get<col_array<lib_algorithms::DteAlgorithmShared::
		  Dte_NodeHeader_Classify<T>>>(
			  ModelsLib::kNodeArray).size();
	  prob_size += models[i]->Get<col_array<T>>(ModelsLib::kProbArray).size();
  }

  col_array<T> aggregate_prob;
  aggregate_prob.reserve(prob_size);
  std::function<void(int)> rec_add_nodes;
  col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>
      aggregate_node;
  aggregate_node.reserve(node_size);

  for (int i = 0; i < models.size(); ++i) {
    auto& node_headers =
        models[i]
            ->Get<col_array<lib_algorithms::DteAlgorithmShared::
                                Dte_NodeHeader_Classify<T>>>(
                ModelsLib::kNodeArray);
    auto& prob_data = models[i]->Get<col_array<T>>(ModelsLib::kProbArray);
    auto trees = models[i]->Get<int>(ModelsLib::kNrTrees);
    auto targets = models[i]->Get<int>(ModelsLib::kNrTargets);
    for (int ii = 0; ii < trees; ++ii) {
      aggregate_node.emplace_back(node_headers[ii]);
      if (aggregate_node.back().child_count <= 0) {
        for (int iii = 0; iii < targets; ++iii)
          aggregate_prob.emplace_back(
              prob_data[node_headers[ii].probability_start + iii]);
      }
    }
  }

  auto trees_agg = 0;
  for (int i = 0; i < models.size(); ++i) {
    auto& node_headers =
        models[i]
            ->Get<col_array<lib_algorithms::DteAlgorithmShared::
                                Dte_NodeHeader_Classify<T>>>(
                ModelsLib::kNodeArray);
    auto& prob_data = models[i]->Get<col_array<T>>(ModelsLib::kProbArray);
    auto trees = models[i]->Get<int>(ModelsLib::kNrTrees);
    auto targets = models[i]->Get<int>(ModelsLib::kNrTargets);
    rec_add_nodes = [&](int node_id) {
      if (aggregate_node[node_id].child_count > 0) {
        int child_start = int(aggregate_node.size());
        for (int i = 0; i < aggregate_node[node_id].child_count; ++i) {
          aggregate_node.emplace_back(
              node_headers[aggregate_node[node_id].child_start + i]);
        }
        aggregate_node[node_id].child_start = child_start;
        for (int i = 0; i < aggregate_node[node_id].child_count; ++i)
          rec_add_nodes(child_start + i);
      } else {
        int prob_start = int(aggregate_prob.size());
        for (int ii = 0; ii < targets; ++ii)
          aggregate_prob.emplace_back(
              prob_data[aggregate_node[node_id].probability_start + ii]);
        aggregate_node[node_id].probability_start = prob_start;
      }
    };

    for (int ii = 0; ii < trees; ++ii) rec_add_nodes(trees_agg + ii);
    trees_agg += trees;
  }

  models[0]->Add(ModelsLib::kNrTrees, trees_agg);
  models[0]->Add(ModelsLib::kNodeArray, aggregate_node);
  models[0]->Add(ModelsLib::kProbArray, aggregate_prob);
}

template <typename T>
col_array<sp<lib_models::MlModel>> DteModelDecorator<T>::SplitModel(
    sp<lib_models::MlModel> model, const int parts) {
  auto trees = model->Get<int>(ModelsLib::kNrTrees);
  auto targets = model->Get<int>(ModelsLib::kNrTargets);
  auto decorator = ModelsLib::GetInstance().CreateDteModelDecorator<T>();
  col_array<sp<lib_models::MlModel>> models(
      parts, ModelsLib::GetInstance().CreateModel(decorator));
  auto& node_headers = model->Get<col_array<
      lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>>(
      ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);

  std::function<void(int)> rec_add_nodes;
  auto tree_split = trees / parts;
  for (int i = 0; i < parts; ++i) {
    col_array<T> prob_array;
    col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>
        node_array;

    rec_add_nodes = [&](int node_id) {
      if (node_array[node_id].child_count > 0) {
        int child_start = int(node_array.size());
        for (int i = 0; i < node_array[node_id].child_count; ++i) {
          node_array.emplace_back(
              node_headers[node_array[node_id].child_start + i]);
        }
        node_array[node_id].child_start = child_start;
        for (int i = 0; i < node_array[node_id].child_count; ++i)
          rec_add_nodes(child_start + i);
      } else {
        int prob_start = int(prob_array.size());
        for (int ii = 0; ii < targets; ++ii)
          prob_array.emplace_back(
              prob_data[node_array[node_id].probability_start + ii]);
        node_array[node_id].probability_start = prob_start;
      }
    };

    models[i]->Add(ModelsLib::kNrTrees, tree_split);
    models[i]->Add(ModelsLib::kNrTargets, targets);
    models[i]->Add(ModelsLib::kNrFeatures,
                   model->Get<int>(ModelsLib::kNrFeatures));
    models[i]->Add(
        ModelsLib::kModelType,
        model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType));

    auto tree_offset = tree_split * i;
    for (int ii = 0; ii < tree_split; ++ii) {
      node_array.emplace_back(node_headers[tree_offset + ii]);
      if (node_array.back().child_count <= 0) {
        int prob_start = int(prob_array.size());
        for (int iii = 0; iii < targets; ++iii)
          prob_array.emplace_back(
              prob_data[node_headers[tree_offset + ii].probability_start +
                        iii]);
        node_array.back().probability_start = prob_start;
      }
    }

    for (int ii = 0; ii < tree_split; ++ii) rec_add_nodes(ii);

    models[i]->Add(ModelsLib::kNodeArray, node_array);
    models[i]->Add(ModelsLib::kProbArray, prob_array);
  }
  return models;
}

template DteModelDecorator<float>::DteModelDecorator();
template DteModelDecorator<double>::DteModelDecorator();
}