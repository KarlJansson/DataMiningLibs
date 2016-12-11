#include "precomp.h"

#include "document_impl.h"
#include "lib_core.h"

namespace lib_preprocess {
void DocumentImpl::LoadDocument(string path, int target_col) {
  std::ifstream open(path);
  if (open.fail()) {
    open.close();
    CoreLib::GetInstance().ThrowException("File not found: " + path);
  }
  if (open.eof()) CoreLib::GetInstance().ThrowException("Empty file: " + path);

  Parse(open, target_col);
  open.close();
}

void DocumentImpl::LoadDocument(char *str_buff, int target_col) {
  std::stringstream open(str_buff);
  Parse(open, target_col);
}

template <typename T>
void DocumentImpl::Parse(T &stream, int target_col) {
  attribute_unique_values_.clear();
  attribute_names_.clear();
  attribute_values_.clear();
  attribute_data_.clear();
  output_values_.clear();

  std::locale read_settings(std::locale::classic(), new document_ctype);
  string line;
  std::getline(stream, line);

  std::stringstream linestream(line);
  linestream.imbue(read_settings);

  std::copy(std::istream_iterator<string>(linestream),
            std::istream_iterator<string>(),
            std::back_inserter(attribute_names_));
  for (auto &str : attribute_names_)
    str.erase(std::remove_if(str.begin(), str.end(), isspace), str.end());

  attribute_data_.assign(attribute_names_.size(), col_array<sp<string>>());
  output_values_.assign(attribute_names_.size(), col_array<string>());
  attribute_unique_values_.assign(attribute_names_.size(),
                                  col_map<string, int>());
  std::locale loc;
  for (auto &str : attribute_names_)
    for (auto &e : str) e = std::tolower(e, loc);
  nr_features_ = int(attribute_names_.size());

  target_col_ = target_col;
  if (target_col_ == -1)
    for (int i = 0; i < attribute_names_.size(); ++i)
      if (attribute_names_[i].compare("class") == 0 ||
          attribute_names_[i].compare("target") == 0 ||
          attribute_names_[i].compare("output") == 0 ||
		  attribute_names_[i].compare("regression") == 0)
        target_col_ = i;
  if (target_col_ < 0 || target_col_ >= attribute_names_.size())
    target_col_ = int(attribute_names_.size() - 1);

  string part;
  nr_samples_ = 0;
  do {
    std::getline(stream, line);
	if (line.empty()) break;
    std::stringstream lstream(line);
    lstream.imbue(read_settings);

    int att_id = 0;
    while (!lstream.eof()) {
      lstream >> part;
	  if (lstream.fail()) break;
      part.erase(std::remove_if(part.begin(), part.end(), isspace), part.end());

      auto itr = attribute_values_.find(part);
      if (itr == attribute_values_.end()) {
        attribute_values_[part] = std::make_shared<string>(part);
        itr = attribute_values_.find(part);
      }

      ++attribute_unique_values_[att_id][part];
      attribute_data_[att_id].push_back(itr->second);

      ++att_id;
    }
    if (att_id != attribute_names_.size())
      CoreLib::GetInstance().ThrowException("Missing attribute in instance");
    ++nr_samples_;
  } while (!stream.eof());
}

void DocumentImpl::SaveDocument(string out_path, int split_percent) {
  if (nr_samples_ < 1) return;
  if (split_percent < 0 || split_percent > 100) return;

  auto &target_map = attribute_unique_values_[target_col_];
  auto target_cpy = target_map;
  for (auto &pair : target_cpy) pair.second = 0;
  auto offset_map = target_cpy;

  auto percent = 100 - split_percent;
  for (int i = 0; i < 2; ++i) {
    if (percent == 0) {
      percent = 100 - percent;
      continue;
    }
    double multi = double(percent) / 100.0;

    std::stringstream content;
    for (int i = 0; i < attribute_names_.size() - 1; ++i)
      content << attribute_names_[i] << ",";
    content << attribute_names_[attribute_names_.size() - 1];
    content << std::endl;

    for (int i = 0; i < nr_samples_; ++i) {
      if (offset_map[*attribute_data_[target_col_][i]] > 0) {
        --offset_map[*attribute_data_[target_col_][i]];
        continue;
      }

      if (target_map[*attribute_data_[target_col_][i]] * multi <
          target_cpy[*attribute_data_[target_col_][i]])
        continue;

      ++target_cpy[*attribute_data_[target_col_][i]];
      for (int ii = 0; ii < attribute_data_.size(); ++ii) {
        if (output_values_[ii].empty())
          content << *attribute_data_[ii][i];
        else
          content << output_values_[ii][i];
        if (ii < attribute_data_.size() - 1) content << ",";
      }

      content << std::endl;
    }

    string output = content.str();
    output.pop_back();
    std::ofstream open(out_path + "_" + std::to_string(percent));
    open << output;
    open.close();

    offset_map = target_cpy;
    for (auto &pair : target_cpy) pair.second = 0;
    percent = 100 - percent;
  }
}

void DocumentImpl::NumberfyAttribute(int attribute_id, bool force_replace) {
  if (attribute_id >= attribute_names_.size()) return;

  bool replace_values = force_replace;
  double value;
  int replacement_value = 0;
  col_map<string, string> replacement_map;
  auto &att_map = attribute_unique_values_[attribute_id];
  for (auto &pair : att_map) {
    std::stringstream lstream(pair.first);
    replacement_map[pair.first] = std::to_string(replacement_value++);
    lstream >> value;
    if (lstream.fail()) replace_values = true;
  }

  output_values_[attribute_id].clear();
  if (replace_values)
    for (int i = 0; i < attribute_data_[attribute_id].size(); ++i)
      output_values_[attribute_id].push_back(
          replacement_map[*attribute_data_[attribute_id][i]]);
}
void DocumentImpl::NominolizeAttribute(int attribute_id) {}
void DocumentImpl::TargetifyAttribute(int attribute_id) {
  if (attribute_id >= attribute_names_.size()) return;

  if (attribute_unique_values_[attribute_id].size() < 20)
    NumberfyAttribute(attribute_id, true);
}

std::stringstream DocumentImpl::GetModifiedString() {
  auto &target_map = attribute_unique_values_[target_col_];
  auto target_cpy = target_map;
  for (auto &pair : target_cpy) pair.second = 0;
  auto offset_map = target_cpy;

  std::stringstream content;
  for (int i = 0; i < attribute_names_.size() - 1; ++i)
    content << attribute_names_[i] << ",";
  content << attribute_names_[attribute_names_.size() - 1];
  content << std::endl;

  for (int i = 0; i < nr_samples_; ++i) {
    for (int ii = 0; ii < attribute_data_.size(); ++ii) {
      if (output_values_[ii].empty())
        content << *attribute_data_[ii][i];
      else
        content << output_values_[ii][i];
      if (ii < attribute_data_.size() - 1) content << ",";
    }

    content << std::endl;
  }

  return content;
}
}