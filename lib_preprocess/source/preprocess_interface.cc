#include "precomp.h"

#include "preprocess_interface.h"
#include "document_impl.h"

namespace lib_preprocess {
PreprocessInterface &PreprocessInterface::GetInstance() {
  static PreprocessInterface instance;
  return instance;
}

sp<Document> PreprocessInterface::CreatePreprocessDocument() {
  return std::make_shared<DocumentImpl>();
}

PreprocessInterface::PreprocessInterface() {}
PreprocessInterface::~PreprocessInterface() {}
}