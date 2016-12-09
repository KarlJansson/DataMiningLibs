#include "precomp.h"

#include <iterator>
#include "csv_parser.h"

namespace lib_parsing {
template <typename T>
CsvParser<T>::CsvParser() {}

template <typename T>
CsvParser<T>::~CsvParser() {}

template <typename T>
inline bool CsvParser<T>::Parse(sp<lib_data::MlDataFrame<T>> data,
                                string filepath) {
  std::ifstream open(filepath);
  if (open.fail()) {
    open.close();
    return false;
  }

  auto ret = Parse(data, open);
  open.close();

  return ret;
}

template <typename T>
inline bool CsvParser<T>::Parse(sp<lib_data::MlDataFrame<T>> data,
                                char* raw_data) {
  // if (stream.empty()) return false;
  std::stringstream stream(raw_data);
  return Parse(data, stream);
}

template <typename T>
bool CsvParser<T>::Parse(sp<lib_data::MlDataFrame<T>> data,
                         col_array<char>& raw_data) {
  if (raw_data.empty()) return false;
  std::stringstream stream(raw_data.data());
  return Parse(data, stream);
}

template <typename T>
template <typename Ts>
inline bool CsvParser<T>::Parse(sp<lib_data::MlDataFrame<T>> data, Ts& stream) {
  std::locale x(std::locale::classic(), new csv_ctype);

  col_array<T> targets;
  col_array<col_array<T>> samples;
  col_array<T> att_ids;
  col_array<string> attributes;
  col_array<col_map<string, T>> attribute_col_map;

  string line;
  std::getline(stream, line);
  if (stream.eof()) return false;

  std::stringstream linestream(line);
  linestream.imbue(x);

  std::copy(std::istream_iterator<string>(linestream),
            std::istream_iterator<string>(), std::back_inserter(attributes));
  attribute_col_map.assign(attributes.size(), col_map<string, T>());
  att_ids.assign(attributes.size(), 0);
  samples.assign(attributes.size() - 1, col_array<T>());

  auto target_id = attributes.size() - 1;
  for (int i = 0; i < attributes.size(); ++i)
    if (attributes[i].compare("class") || attributes[i].compare("target") ||
        attributes[i].compare("output"))
      target_id = i;

  T value;
  string part;
  do {
    std::getline(stream, line);
    std::stringstream lstream(line);
    lstream.imbue(x);

    int att_id = 0;
    while (!lstream.eof()) {
      auto& att_col_map = attribute_col_map[att_id];
      lstream >> part;
      std::stringstream pstream(part);
      pstream >> value;

      if (pstream.fail()) {
        if (att_col_map.find(part) == att_col_map.end()) {
          att_col_map[part] = att_ids[att_id];
          ++att_ids[att_id];
        }

        value = att_col_map[part];
      } else
        att_col_map[part] = value;

      if (att_id == target_id)
        targets.emplace_back(value);
      else
        samples[att_id].emplace_back(value);

      ++att_id;
    }
    if (att_id != attributes.size()) return false;
  } while (!stream.eof());

  col_array<T> sample_packed;
  sample_packed.assign(samples.size() * samples[0].size(), 0);
  for (int i = 0; i < samples.size(); ++i)
    std::copy(samples[i].begin(), samples[i].end(),
              sample_packed.begin() + i * samples[0].size());

  data->AddTargetData(targets);
  data->AddSampleData(sample_packed);
  data->AddTargetMap(attribute_col_map.back());
  attribute_col_map.pop_back();
  data->AddFeatureMap(attribute_col_map);
  return true;
}

template CsvParser<float>::CsvParser();
template CsvParser<double>::CsvParser();
}
