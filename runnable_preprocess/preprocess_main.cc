#define DLLExport
#define TestExport

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include "lib_preprocess.h"

void ProcessDirectory(string data_dir, int percent) {
  auto document_loader =
      PreprocessLib::GetInstance().CreatePreprocessDocument();
  for (auto& p :
       std::experimental::filesystem::recursive_directory_iterator(data_dir)) {
    std::cout << p << std::endl;
    std::stringstream path_stream;
    path_stream << p;
    document_loader->LoadDocument(path_stream.str());
    for (int i = 0; i < document_loader->NrFeatures(); ++i)
      if (i == document_loader->TargetCol())
        document_loader->TargetifyAttribute(i);
      else
        document_loader->NumberfyAttribute(i);
    document_loader->SaveDocument(path_stream.str() + "_mod", percent);
  }
}

int main(int argc, char** argv) {
  int p = std::stoi(argv[1]);
  for (int i = 2; i < argc; ++i) ProcessDirectory(argv[i], p);
}