#pragma once

namespace lib_preprocess {
class DLLExport Document {
 public:
  Document();
  virtual ~Document();

  virtual void LoadDocument(string path, int target_col = -1) = 0;
  virtual void LoadDocument(char* str_buff, int target_col = -1) = 0;
  virtual void SaveDocument(string out_path, int split_percent = 0) = 0;

  virtual void NumberfyAttribute(int attribute_id,
                                 bool force_replace = false) = 0;
  virtual void NominolizeAttribute(int attribute_id) = 0;
  virtual void TargetifyAttribute(int attribute_id) = 0;
  virtual std::stringstream GetModifiedString() = 0;

  int NrFeatures();
  int NrSamples();
  int TargetCol();

 protected:
  int target_col_;
  int nr_samples_;
  int nr_features_;
};
}