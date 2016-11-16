#include "precomp.h"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cassert>
#include "gpudte_algorithm.h"
#include "lib_core.h"
#include "lib_gpu.h"

#include "gpudte.h"

namespace lib_cuda_algorithms {
template <typename T>
sp<lib_models::MlModel> GpuDteAlgorithm<T>::Fit(
    sp<lib_data::MlDataFrame<T>> data,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  GpuDteAlgorithmShared::gpuDTE_StaticInfo static_info;
  GpuDteAlgorithmShared::gpuDTE_DatasetInfo dataset_info;
  GpuDteAlgorithmShared::gpuDTE_IterationInfo iteration_info;
  auto device = GpuLib::GetInstance().CreateGpuDevice(
      params->Get<int>(AlgorithmsLib::kDevId));
  auto gpu_params = GpuDteAlgorithmShared::GpuParams<T>();
  auto algo_type =
      params->Get<AlgorithmsLib::AlgorithmType>(AlgorithmsLib::kAlgoType);
  auto batch_size = params->Get<int>(AlgorithmsLib::kTreeBatchSize);
  auto max_gpu_blocks = params->Get<int>(AlgorithmsLib::kMaxGpuBlocks);
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  int trees_built = 0;
  auto k = params->Get<int>(AlgorithmsLib::kNrFeatures);
  k = k > 0 ? k : int(std::round(log(nr_features))) + 1;

  auto decorator = ModelsLib::GetInstance().CreateDteModelDecorator<T>();
  col_array<sp<lib_models::MlModel>> models;
  lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T> tmp_node;
  col_array<
      col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>>
      tree_nodes;
  col_array<col_array<T>> tree_probabilities;
  HostAllocFit host_alloc(device, nr_targets, max_gpu_blocks);

  // auto barrier = CoreLib::GetInstance().CreateBarrier(2);
  // bool run_rec_func = true;
  sutil::LockFreeList<std::pair<int, int>> job_list;
  col_map<int, int> track_map;
  int node_size =
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>);
  int type_size = sizeof(T);
  int prob_id = 0;
  auto data_rec_func = [&]() {
    // device.SetDeviceForThread(params->Get<int>(EnsemblesLib::kDevId));
    // barrier->Wait();
    // do {
    auto pair = job_list.pop_front();
    device->CopyToHost(host_alloc.probability_cpy,
                       gpu_params.probability_buffers[prob_id],
                       type_size * pair->first * nr_targets, 1);
    device->CopyToHost(host_alloc.node_cpy,
                       gpu_params.node_buffers[pair->second],
                       node_size * pair->first, 1);
    device->SynchronizeDevice();
    for (int i = 0; i < pair->first; ++i) {
      auto& gpu_node = host_alloc.node_cpy[i];
      if (gpu_node.attribute != -1) {
        if (gpu_node.attribute >= nr_features || gpu_node.attribute < 0) {
          CoreLib::GetInstance().ThrowException("Faulty gpu node encountered.");
        }
      }

      auto itr = track_map.find(gpu_node.parent_id);
      if (gpu_node.parent_id >= 0 && itr != track_map.end()) {
        tree_nodes.back()[itr->second].child_start =
            int(tree_nodes.back().size());
        track_map.erase(itr->first);
      }

      tmp_node.child_count = gpu_node.attribute == -1 ? 0 : 2;
      tmp_node.attribute = gpu_node.attribute;
      tmp_node.split_point = gpu_node.split_point;

      if (tmp_node.child_count == 0) {
        tmp_node.probability_start = int(tree_probabilities.back().size());
        for (int ii = 0; ii < nr_targets; ++ii) {
          tree_probabilities.back().emplace_back(
              host_alloc.probability_cpy[i * nr_targets + ii]);
        }
      }

      if (tmp_node.child_count > 0)
        track_map[gpu_node.tracking_id] = int(tree_nodes.back().size());
      tree_nodes.back().emplace_back(tmp_node);
    }
    prob_id = prob_id == 0 ? 1 : 0;
    // barrier->Wait();
    //} while (run_rec_func);
  };
  // sp<std::thread> data_rec_thread =
  //    std::make_shared<std::thread>(data_rec_func);

  AllocateFit(device, params, &gpu_params, data, static_info, dataset_info,
              iteration_info);

  col_array<
      col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>>>
      node_cache(
          2,
          col_array<
              lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>>());
  int nodes_pulled = 0;
  int stream_buffer = 2;
  int trees_left = 0;
  int batch = 0;

  auto tree_id_counter = params->Get<sp<int>>(AlgorithmsLib::kTreeCounter);
  auto tree_id_counter_mutex =
      params->Get<sp<mutex>>(AlgorithmsLib::kTreeCounterMutex);

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

    while (batch > 0) {
      tree_probabilities.emplace_back(col_array<T>());
      tree_nodes.emplace_back(
          col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<
              T>>());
      stream_buffer = 2;
      iteration_info.depth = 0;
      iteration_info.read_buffer_id = 0;
      iteration_info.write_buffer_id = 1;
      iteration_info.prob_buffer_id = 0;
      iteration_info.tick_tock = true;

      int trees_launched = batch > batch_size ? batch_size : batch;
      batch -= trees_launched;
      models.emplace_back(ModelsLib::GetInstance().CreateModel(decorator));
      models.back()->Add(ModelsLib::kNrTrees, trees_launched);
      models.back()->Add(ModelsLib::kNrFeatures, nr_features);
      models.back()->Add(ModelsLib::kNrTargets, nr_targets);
      models.back()->Add(ModelsLib::kModelType, algo_type);

      int nodes_left = trees_launched;
      int layer_id = 0;
      col_array<int> buffer_counts(3, 0);
      buffer_counts[iteration_info.read_buffer_id] = nodes_left;

      iteration_info.threads_launched = max_gpu_blocks * block_size_;
      gpu_functions_->CopyIterationInfo(iteration_info);
      device->SynchronizeDevice();
      gpu_functions_->CallCudaKernel(max_gpu_blocks / block_size_, block_size_,
                                     gpu_params,
                                     GpuDteAlgorithmShared::kSetupKernel);
      device->SynchronizeDevice();

      iteration_info.threads_launched = trees_launched;
      gpu_functions_->CopyIterationInfo(iteration_info);
      gpu_functions_->CallCudaKernel(trees_launched, block_size_, gpu_params,
                                     GpuDteAlgorithmShared::kInitTreeBatch);
      device->SynchronizeDevice();

      // Build trees
      do {
        bool swap_next = false;
        // Build node layer
        do {
          int nodes_launched = nodes_left > max_gpu_blocks / max_nominal_
                                   ? max_gpu_blocks / max_nominal_
                                   : nodes_left;

          nodes_left -= nodes_launched;
          iteration_info.first_part = true;
          iteration_info.threads_launched = nodes_launched;
          gpu_functions_->CopyIterationInfo(iteration_info);
          for (int i = 0; i < k; ++i) {
            gpu_functions_->CallCudaKernel(nodes_launched, block_size_,
                                           gpu_params,
                                           GpuDteAlgorithmShared::kFindSplit);
            device->SynchronizeDevice();
            if (i == 0) {
              iteration_info.first_part = false;
              gpu_functions_->CopyIterationInfo(iteration_info);
              device->SynchronizeDevice();
            }
          }

          gpu_functions_->CallCudaKernel(nodes_launched, block_size_,
                                         gpu_params,
                                         GpuDteAlgorithmShared::kPerformSplit);
          device->SynchronizeDevice();
          device->CopyToHost(host_alloc.node_cursor_cpy,
                             gpu_params.node_cursors,
                             sizeof(int) * cpy_buffer_size_);
          device->SynchronizeDevice();

          iteration_info.node_offset += nodes_launched;
          buffer_counts[iteration_info.write_buffer_id] =
              host_alloc.node_cursor_cpy[new_nodes_];

          // Swap write buffer
          if (swap_next) {
            iteration_info.node_offset = 0;
            SwapBuffers(&iteration_info.read_buffer_id, &stream_buffer);
            swap_next = false;

            // Stream partial layer results
            iteration_info.prob_buffer_id =
                iteration_info.prob_buffer_id == 0 ? 1 : 0;

            job_list.push_front(std::pair<int, int>(
                buffer_counts[stream_buffer], stream_buffer));
            buffer_counts[stream_buffer] = 0;
            data_rec_func();
            // barrier->Wait();
            nodes_left = nodes_pulled;
          } else if (!node_cache[layer_id].empty() &&
                     nodes_left - int(max_gpu_blocks / max_nominal_) <= 0) {
            nodes_pulled = max_gpu_blocks > node_cache[layer_id].size()
                               ? int(node_cache[layer_id].size())
                               : max_gpu_blocks;

            // Pre-stream next layer chunk for next iteration
            buffer_counts[stream_buffer] = nodes_pulled;
            StreamFromCache(device, host_alloc, stream_buffer, layer_id,
                            node_cache, buffer_counts,
                            gpu_params.node_buffers[stream_buffer]);

            if (buffer_counts[iteration_info.write_buffer_id] > 0)
              StreamToCache(
                  device, host_alloc, iteration_info.write_buffer_id, layer_id,
                  node_cache, buffer_counts,
                  gpu_params.node_buffers[iteration_info.write_buffer_id]);

            swap_next = true;
          }

          if (!swap_next) {
            // Stream nodes to the cache
            SwapBuffers(&iteration_info.write_buffer_id, &stream_buffer);

            if (buffer_counts[stream_buffer] > 0)
              StreamToCache(device, host_alloc, stream_buffer, layer_id,
                            node_cache, buffer_counts,
                            gpu_params.node_buffers[stream_buffer]);
          }

          // Update node counts on GPU
          host_alloc.node_cursor_cpy[work_cursor_] =
              host_alloc.node_cursor_cpy[new_nodes_] = 0;
          device->CopyToDevice(host_alloc.node_cursor_cpy,
                               gpu_params.node_cursors,
                               sizeof(int) * cpy_buffer_size_);
          device->SynchronizeDevice();
        } while (nodes_left > 0);

        // Stream the last layer results
        iteration_info.prob_buffer_id =
            iteration_info.prob_buffer_id == 0 ? 1 : 0;

        job_list.push_front(
            std::pair<int, int>(buffer_counts[iteration_info.read_buffer_id],
                                iteration_info.read_buffer_id));
        buffer_counts[iteration_info.read_buffer_id] = 0;
        data_rec_func();
        // barrier->Wait();

        // Prepare next layer
        layer_id = layer_id == 0 ? 1 : 0;
        if (!node_cache[layer_id].empty()) {
          nodes_left = max_gpu_blocks < node_cache[layer_id].size()
                           ? max_gpu_blocks
                           : int(node_cache[layer_id].size());
          buffer_counts[iteration_info.read_buffer_id] = nodes_left;
          StreamFromCache(
              device, host_alloc, iteration_info.read_buffer_id, layer_id,
              node_cache, buffer_counts,
              gpu_params.node_buffers[iteration_info.read_buffer_id]);
        }

        ++iteration_info.depth;
        iteration_info.node_offset = 0;
        iteration_info.tick_tock = !iteration_info.tick_tock;
      } while (nodes_left > 0);

      trees_built += trees_launched;

      models.back()->Add(ModelsLib::kNodeArray, tree_nodes.back());
      models.back()->Add(ModelsLib::kProbArray, tree_probabilities.back());
    }
  }

  // run_rec_func = false;
  // barrier->Wait();
  // if (data_rec_thread->joinable()) data_rec_thread->join();

  auto model = models.empty() ? nullptr : models[0];
  col_array<sp<lib_models::MlModel>> merge_models;
  for (int i = 1; i < models.size(); ++i) {
    if (!model)
      model = models[i];
    else if (models[i])
      merge_models.emplace_back(models[i]);
  }
  if (!merge_models.empty()) model->Merge(merge_models);

  gpu_params.finalize(device);
  return model;
}

template <typename T>
sp<lib_data::MlResultData<T>> GpuDteAlgorithm<T>::Predict(
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    sp<lib_algorithms::MlAlgorithmParams> params) {
  GpuDteAlgorithmShared::gpuDTE_StaticInfo static_info;
  GpuDteAlgorithmShared::gpuDTE_DatasetInfo dataset_info;
  GpuDteAlgorithmShared::gpuDTE_IterationInfo iteration_info;
  auto device = GpuLib::GetInstance().CreateGpuDevice(
      params->Get<int>(AlgorithmsLib::kDevId));
  auto result_data = DataLib::GetInstance().CreateResultData<T>();
  auto gpu_params = GpuDteAlgorithmShared::GpuParams<T>();
  int nr_samples = data->GetNrSamples();
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int target_values = model->Get<int>(ModelsLib::kNrTargets);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) target_values = 1;
  auto max_gpu_blocks = params->Get<int>(AlgorithmsLib::kMaxGpuBlocks);

  AllocatePredict(device, params, &gpu_params, data, model, static_info,
                  dataset_info, iteration_info);
  HostAllocPredict host_alloc(device, nr_samples, target_values);

  // Run prediction process
  {
    int launch_threads;
    int total_threads = nr_trees * nr_samples;

    while (total_threads > 0) {
      launch_threads = ceil(T(total_threads) / T(block_size_)) > max_gpu_blocks
                           ? max_gpu_blocks * block_size_
                           : total_threads;
      iteration_info.threads_launched = launch_threads;
      gpu_functions_->CopyIterationInfo(iteration_info);
      device->SynchronizeDevice();
      gpu_functions_->CallCudaKernel(launch_threads / block_size_, block_size_,
                                     gpu_params,
                                     GpuDteAlgorithmShared::kPredict);
      device->SynchronizeDevice();

      iteration_info.tree_offset += launch_threads;
      total_threads -= launch_threads;
    }
  }

  // Fill out result buffers
  device->CopyToHost(host_alloc.predictions_cpy, gpu_params.predictions,
                     sizeof(T) * nr_samples * target_values);
  device->SynchronizeDevice();
  col_array<col_array<T>> predictions(nr_samples, col_array<T>());
  auto lambda_func = [&](int i) {
    if (model_type == AlgorithmsLib::kRegression)
      predictions[i].emplace_back(host_alloc.predictions_cpy[i]);
    else
      for (int ii = 0; ii < target_values; ++ii)
        predictions[i].emplace_back(
            host_alloc.predictions_cpy[i * target_values + ii]);
  };
  CoreLib::GetInstance().TBBParallelFor(0, nr_samples, lambda_func);

  gpu_params.finalize(device);
  result_data->AddPredictions(predictions);
  return result_data;
}

template <typename T>
void GpuDteAlgorithm<T>::AllocateFit(
    sp<lib_gpu::GpuDevice> dev, sp<lib_algorithms::MlAlgorithmParams> params,
    GpuDteAlgorithmShared::GpuParams<T>* gpu_params,
    sp<lib_data::MlDataFrame<T>> data,
    GpuDteAlgorithmShared::gpuDTE_StaticInfo& static_info,
    GpuDteAlgorithmShared::gpuDTE_DatasetInfo& dataset_info,
    GpuDteAlgorithmShared::gpuDTE_IterationInfo& iteration_info) {
  auto nr_samples = data->GetNrSamples();
  auto nr_features = data->GetNrFeatures();
  auto nr_targets = data->GetNrTargets();
  auto nr_total_trees = params->Get<int>(AlgorithmsLib::kNrTrees);
  const auto max_samples_per_tree =
      nr_samples < params->Get<int>(AlgorithmsLib::kMaxSamplesPerTree)
          ? nr_samples
          : params->Get<int>(AlgorithmsLib::kMaxSamplesPerTree);
  auto batch_size = params->Get<int>(AlgorithmsLib::kTreeBatchSize);
  const auto nr_fit_samples =
      max_samples_per_tree <= 0 ? nr_samples : max_samples_per_tree;
  auto max_gpu_blocks = params->Get<int>(AlgorithmsLib::kMaxGpuBlocks);

  // Allocate training buffers
  auto& data_samples = data->GetSamples();
  auto& data_targets = data->GetTargets();

  col_array<std::pair<void**, size_t>> mem_offsets;
  for (int i = 0; i < 3; ++i) {
    mem_offsets.emplace_back(std::pair<void**, size_t>(
        (void**)&gpu_params->node_buffers[i],
        sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>) *
            max_gpu_blocks));
  }
  for (int i = 0; i < 2; ++i) {
    mem_offsets.emplace_back(
        std::pair<void**, size_t>((void**)&gpu_params->indices_buffer[i],
                                  sizeof(int) * nr_fit_samples * batch_size));
    mem_offsets.emplace_back(std::pair<void**, size_t>(
        (void**)&gpu_params->probability_buffers[i],
        sizeof(T) * max_gpu_blocks * nr_targets * max_nominal_));
  }
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->node_cursors, sizeof(int) * cpy_buffer_size_));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->node_tmp_buffer,
      sizeof(GpuDteAlgorithmShared::gpuDTE_TmpNodeValues<T>) * max_gpu_blocks));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->probability_tmp_buffer,
      sizeof(T) * max_gpu_blocks * nr_targets * max_nominal_));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->target_starts, sizeof(int) * nr_targets));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->indices_inbag,
      sizeof(bool) * nr_fit_samples * max_gpu_blocks));
  mem_offsets.emplace_back(
      std::pair<void**, size_t>((void**)&gpu_params->random_states,
                                sizeof(curandStateMRG32k3a) * max_gpu_blocks));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->attribute_type, sizeof(int) * nr_features));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->target_data, sizeof(T) * data_targets.size()));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->dataset, sizeof(T) * data_samples.size()));

  for (int i = 0; i < mem_offsets.size(); ++i)
    dev->AllocateMemory(mem_offsets[i].first, mem_offsets[i].second);

  T* dataset_cpy;
  T* target_cpy;
  int* attribute_cpy;
  int* node_cursor_cpy;
  dev->AllocateHostMemory((void**)&node_cursor_cpy,
                          sizeof(int) * cpy_buffer_size_);
  dev->AllocateHostMemory((void**)&dataset_cpy,
                          sizeof(T) * data_samples.size());
  dev->AllocateHostMemory((void**)&target_cpy, sizeof(T) * data_targets.size());
  dev->AllocateHostMemory((void**)&attribute_cpy,
                          sizeof(T) * data_samples.size());
  memset(attribute_cpy, 2, sizeof(int) * nr_features);
  memset(node_cursor_cpy, 0, sizeof(int) * cpy_buffer_size_);
  memcpy(dataset_cpy, data_samples.data(), sizeof(T) * data_samples.size());
  memcpy(target_cpy, data_targets.data(), sizeof(T) * data_targets.size());
  dev->CopyToDevice(dataset_cpy, gpu_params->dataset,
                    sizeof(T) * data_samples.size());
  dev->CopyToDevice(target_cpy, gpu_params->target_data,
                    sizeof(T) * data_targets.size());
  dev->CopyToDevice(attribute_cpy, gpu_params->attribute_type,
                    sizeof(int) * nr_features);
  dev->CopyToDevice(node_cursor_cpy, gpu_params->node_cursors,
                    sizeof(int) * cpy_buffer_size_);

  memset(&dataset_info, 0, sizeof(dataset_info));
  dataset_info.nr_attributes = nr_features;
  dataset_info.nr_instances = nr_samples;
  dataset_info.nr_target_values = nr_targets;
  dataset_info.data_type = params->Get<int>(AlgorithmsLib::kAlgoType);

  memset(&static_info, 0, sizeof(static_info));
  static_info.loaded_trees = batch_size;
  static_info.total_trees = nr_total_trees;
  static_info.max_node_size = params->Get<int>(AlgorithmsLib::kMinNodeSize);
  static_info.max_node_depth = params->Get<int>(AlgorithmsLib::kMaxDepth);
  static_info.max_inst_tree = nr_fit_samples;
  static_info.node_buffer_size = max_gpu_blocks;

  auto k = params->Get<int>(AlgorithmsLib::kNrFeatures);
  static_info.nr_features = k > 0 ? k : int(std::round(log(nr_features))) + 1;
  static_info.max_class_count = nr_targets;
  static_info.balanced_sampling =
      params->Get<bool>(AlgorithmsLib::kEasyEnsemble);

  memset(&iteration_info, 0, sizeof(iteration_info));
  iteration_info.read_buffer_id = iteration_info.write_buffer_id = 0;
  iteration_info.tick_tock = true;

  gpu_functions_->CopyDataStaticInfo(dataset_info, static_info);
  gpu_functions_->CopyIterationInfo(iteration_info);
  dev->SynchronizeDevice();
  dev->DeallocateHostMemory(dataset_cpy);
  dev->DeallocateHostMemory(target_cpy);
  dev->DeallocateHostMemory(attribute_cpy);
  dev->DeallocateHostMemory(node_cursor_cpy);
}

template <typename T>
void GpuDteAlgorithm<T>::AllocatePredict(
    sp<lib_gpu::GpuDevice> dev, sp<lib_algorithms::MlAlgorithmParams> params,
    GpuDteAlgorithmShared::GpuParams<T>* gpu_params,
    sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
    GpuDteAlgorithmShared::gpuDTE_StaticInfo& static_info,
    GpuDteAlgorithmShared::gpuDTE_DatasetInfo& dataset_info,
    GpuDteAlgorithmShared::gpuDTE_IterationInfo& iteration_info) {
  int nr_targets = model->Get<int>(ModelsLib::kNrTargets);
  int nr_samples = data->GetNrSamples();
  int nr_trees = model->Get<int>(ModelsLib::kNrTrees);
  int nr_features = model->Get<int>(ModelsLib::kNrFeatures);
  auto model_type =
      model->Get<AlgorithmsLib::AlgorithmType>(ModelsLib::kModelType);
  if (model_type == AlgorithmsLib::kRegression) nr_targets = 1;

  // Allocate prediction buffers
  auto& node_headers = model->Get<col_array<
      lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>>>(
      ModelsLib::kNodeArray);
  auto& prob_data = model->Get<col_array<T>>(ModelsLib::kProbArray);
  auto& data_samples = data->GetSamples();

  col_array<std::pair<void**, size_t>> mem_offsets;
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->predictions, sizeof(T) * nr_samples * nr_targets));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->node_buffers_classify,
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>) *
          int(node_headers.size())));
  mem_offsets.emplace_back(
      std::pair<void**, size_t>((void**)&gpu_params->probability_tmp_buffer,
                                sizeof(T) * prob_data.size()));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->dataset, sizeof(T) * data_samples.size()));
  mem_offsets.emplace_back(std::pair<void**, size_t>(
      (void**)&gpu_params->attribute_type, sizeof(int) * nr_features));

  for (int i = 0; i < mem_offsets.size(); ++i)
    dev->AllocateMemory(mem_offsets[i].first, mem_offsets[i].second);

  lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>* node_head_cpy;
  T* prob_data_cpy;
  T* data_samples_cpy;
  T* pred_init_cpy;
  int* attribute_cpy;
  dev->AllocateHostMemory((void**)&pred_init_cpy,
                          sizeof(T) * nr_samples * nr_targets);
  dev->AllocateHostMemory(
      (void**)&node_head_cpy,
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>) *
          (node_headers.size()));
  dev->AllocateHostMemory((void**)&prob_data_cpy, sizeof(T) * prob_data.size());
  dev->AllocateHostMemory((void**)&data_samples_cpy,
                          sizeof(T) * data_samples.size());
  dev->AllocateHostMemory((void**)&attribute_cpy, sizeof(int) * nr_features);
  memset(attribute_cpy, 2, sizeof(int) * nr_features);
  memset(pred_init_cpy, 0, sizeof(T) * nr_samples * nr_targets);
  memcpy(
      node_head_cpy, node_headers.data(),
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>) *
          node_headers.size());

  memcpy(prob_data_cpy, prob_data.data(), sizeof(T) * prob_data.size());
  memcpy(data_samples_cpy, data_samples.data(),
         sizeof(T) * data_samples.size());
  dev->CopyToDevice(
      node_head_cpy, gpu_params->node_buffers_classify,
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Classify<T>) *
          node_headers.size());
  dev->CopyToDevice(prob_data_cpy, gpu_params->probability_tmp_buffer,
                    sizeof(T) * prob_data.size());
  dev->CopyToDevice(data_samples_cpy, gpu_params->dataset,
                    sizeof(T) * data_samples.size());
  dev->CopyToDevice(attribute_cpy, gpu_params->attribute_type,
                    sizeof(int) * nr_features);
  dev->CopyToDevice(pred_init_cpy, gpu_params->predictions,
                    sizeof(T) * nr_samples * nr_targets);

  memset(&dataset_info, 0, sizeof(dataset_info));
  dataset_info.nr_attributes = nr_features;
  dataset_info.nr_instances = nr_samples;
  dataset_info.nr_target_values = nr_targets;
  dataset_info.data_type = model_type;

  memset(&static_info, 0, sizeof(static_info));
  static_info.loaded_trees = nr_trees;
  static_info.total_trees = nr_trees;
  static_info.max_node_size = params->Get<int>(AlgorithmsLib::kMinNodeSize);
  static_info.max_node_depth = params->Get<int>(AlgorithmsLib::kMaxDepth);

  auto k = params->Get<int>(AlgorithmsLib::kNrFeatures);
  static_info.nr_features = k > 0 ? k : int(std::round(log(nr_features))) + 1;
  static_info.max_class_count = nr_targets;
  static_info.balanced_sampling =
      params->Get<bool>(AlgorithmsLib::kEasyEnsemble);

  memset(&iteration_info, 0, sizeof(iteration_info));
  iteration_info.read_buffer_id = 0;
  iteration_info.tree_offset = 0;

  gpu_functions_->CopyDataStaticInfo(dataset_info, static_info);
  gpu_functions_->CopyIterationInfo(iteration_info);
  dev->SynchronizeDevice();
  dev->DeallocateHostMemory(node_head_cpy);
  dev->DeallocateHostMemory(prob_data_cpy);
  dev->DeallocateHostMemory(data_samples_cpy);
  dev->DeallocateHostMemory(attribute_cpy);
  dev->DeallocateHostMemory(pred_init_cpy);
}

template <typename T>
void GpuDteAlgorithm<T>::SwapBuffers(int* lhs, int* rhs) {
  int tmp = *rhs;
  *rhs = *lhs;
  *lhs = tmp;
}

template <typename T>
void GpuDteAlgorithm<T>::StreamToCache(
    sp<lib_gpu::GpuDevice> dev, HostAllocFit& host_alloc, int src_id,
    int layer_id,
    col_array<
        col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>>>&
        node_cache,
    col_array<int>& buffer_counts,
    lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[src_id];
  buffer_counts[src_id] = 0;
  if (nr_nodes <= 0) return;

  dev->CopyToHost(
      host_alloc.node_cpy, node_headers,
      nr_nodes *
          sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>));
  dev->SynchronizeDevice();

  // Add to cache
  int cache_id = layer_id == 0 ? 1 : 0;
  for (int i = 0; i < nr_nodes; ++i)
    node_cache[cache_id].emplace_back(host_alloc.node_cpy[i]);
}

template <typename T>
void GpuDteAlgorithm<T>::StreamFromCache(
    sp<lib_gpu::GpuDevice> dev, HostAllocFit& host_alloc, int dst_id,
    int layer_id,
    col_array<
        col_array<lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>>>&
        node_cache,
    col_array<int>& buffer_counts,
    lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>* node_headers) {
  int nr_nodes = buffer_counts[dst_id];

  // Pre-stream next layer chunk for next iteration
  memcpy(host_alloc.node_cpy, node_cache[layer_id].data(),
         sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>) *
             nr_nodes);
  dev->CopyToDevice(
      host_alloc.node_cpy, node_headers,
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>) *
          nr_nodes);

  if (node_cache[layer_id].size() - nr_nodes > 0)
    for (int i = 0; i < node_cache[layer_id].size() - nr_nodes; ++i)
      node_cache[layer_id][i] = node_cache[layer_id][nr_nodes + i];

  for (int i = 0; i < nr_nodes; ++i) node_cache[layer_id].pop_back();
  dev->SynchronizeDevice();
}

template <typename T>
GpuDteAlgorithm<T>::HostAllocFit::HostAllocFit(sp<lib_gpu::GpuDevice> dev,
                                               size_t targets,
                                               int max_gpu_blocks)
    : dev_(dev) {
  dev_->AllocateHostMemory((void**)&probability_cpy,
                           sizeof(T) * max_gpu_blocks * targets * max_nominal_);
  dev_->AllocateHostMemory(
      (void**)&node_cpy,
      sizeof(lib_algorithms::DteAlgorithmShared::Dte_NodeHeader_Train<T>) *
          max_gpu_blocks);
  dev_->AllocateHostMemory((void**)&node_cursor_cpy,
                           sizeof(int) * cpy_buffer_size_);
}
template <typename T>
GpuDteAlgorithm<T>::HostAllocFit::~HostAllocFit() {
  dev_->DeallocateHostMemory(probability_cpy);
  dev_->DeallocateHostMemory(node_cpy);
  dev_->DeallocateHostMemory(node_cursor_cpy);
}

template <typename T>
GpuDteAlgorithm<T>::HostAllocPredict::HostAllocPredict(
    sp<lib_gpu::GpuDevice> dev, size_t samples, size_t targets)
    : dev_(dev) {
  dev_->AllocateHostMemory((void**)&predictions_cpy,
                           samples * targets * sizeof(T));
}
template <typename T>
GpuDteAlgorithm<T>::HostAllocPredict::~HostAllocPredict() {
  dev_->DeallocateHostMemory(predictions_cpy);
}

template GpuDteAlgorithm<float>::GpuDteAlgorithm(sp<GpuDte<float>> func);
template GpuDteAlgorithm<double>::GpuDteAlgorithm(sp<GpuDte<double>> func);
}