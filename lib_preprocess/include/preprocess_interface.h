#pragma once
#include "document.h"

namespace lib_preprocess {
class DLLExport PreprocessInterface {
 public:
  static PreprocessInterface &GetInstance();

  sp<Document> CreatePreprocessDocument();

 private:
  PreprocessInterface();
  ~PreprocessInterface();
};
}