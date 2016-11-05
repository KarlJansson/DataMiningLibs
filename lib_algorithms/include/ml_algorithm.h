#pragma once

namespace lib_models {
class MlModel;
}
namespace lib_data {
template <typename T>
class MlDataFrame;
template <typename T>
class MlResultData;
}

namespace lib_algorithms {
class MlAlgorithmParams;
template <typename T>
class DLLExport MlAlgorithm {
 public:
  MlAlgorithm() {}
  virtual ~MlAlgorithm() {}

  /**
  *	\brief
  *	Fit a model to a data set.
  *	\param
  *	data - Data to be fited into a model.
  *	\returns
  *	A shared pointer to the fited model.
  */
  virtual sp<lib_models::MlModel> Fit(sp<lib_data::MlDataFrame<T>> data,
                                      sp<MlAlgorithmParams> params) = 0;
  /**
  *	\brief
  *	Predict a data frame using a model.
  * \param
  * data - Data to be predicted by a model
  * model - The model used to perform the prediction on the data.
  * \returns
  * A shared pointer to the resulting predictions
  */
  virtual sp<lib_data::MlResultData<T>> Predict(
      sp<lib_data::MlDataFrame<T>> data, sp<lib_models::MlModel> model,
      sp<MlAlgorithmParams> params) = 0;
};
}