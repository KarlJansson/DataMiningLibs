#pragma once
#include "document.h"

namespace lib_preprocess {
class DocumentImpl : public Document {
 public:
  DocumentImpl() = default;

  void LoadDocument(string path, int target_col = -1) override;
  void SaveDocument(string out_path, int split_percent = 100) override;

  void NumberfyAttribute(int attribute_id, bool force_replace = false) override;
  void NominolizeAttribute(int attribute_id) override;
  void TargetifyAttribute(int attribute_id) override;

 private:
  class document_ctype : public std::ctype<char> {
    mask my_table[table_size];

   public:
    document_ctype(size_t refs = 0)
        : std::ctype<char>(&my_table[0], false, refs) {
      std::copy_n(classic_table(), table_size, my_table);
      my_table[','] = (mask)space;
      my_table['	'] = (mask)space;
      my_table[' '] = (mask)digit;
    }
  };

  col_array<col_map<string, int>> attribute_unique_values_;
  col_array<string> attribute_names_;
  col_map<string, sp<string>> attribute_values_;
  col_array<col_array<sp<string>>> attribute_data_;
  col_array<col_array<string>> output_values_;
};
}