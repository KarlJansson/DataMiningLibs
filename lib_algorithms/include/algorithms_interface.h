#pragma once

namespace lib_algorithms {
template <typename T>
class MlAlgorithm;
class MlAlgorithmParams;
class DLLExport AlgorithmsInterface {
 public:
  static AlgorithmsInterface& GetInstance();

  enum AlgorithmType { kClassification = 0, kRegression };

  sp<MlAlgorithmParams> CreateAlgorithmParams(int size);
  col_array<sp<lib_algorithms::MlAlgorithmParams>> SplitDteParamPack(
      sp<lib_algorithms::MlAlgorithmParams> params, const int parts);

  template <typename T>
  sp<MlAlgorithm<T>> CreateCpuAlgorithm(sp<MlAlgorithm<T>> algo);
  template <typename T>
  sp<MlAlgorithm<T>> CreateGpuAlgorithm(sp<MlAlgorithm<T>> algo);
  template <typename T>
  sp<MlAlgorithm<T>> CreateHybridAlgorithm(sp<MlAlgorithm<T>> gpu_algo,
                                           sp<MlAlgorithm<T>> cpu_algo);

  enum CommonParams {
	  kDevId = 0,

	  kCommonEndMarker
  };
  enum DteParams {
	  kNrTrees = kCommonEndMarker,
	  kTreeCounter,
	  kTreeCounterMutex,
	  kTreeBatchSize,
	  kAlgoType,
	  kNrFeatures,
	  kMaxDepth,
	  kMaxSamplesPerTree,
	  kMinNodeSize,
	  kEasyEnsemble,
	  kBagging,

	  kDteEndMarker
  };

 private:
  AlgorithmsInterface() {}
  ~AlgorithmsInterface() {}
};
}