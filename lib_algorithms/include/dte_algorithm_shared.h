#pragma once

namespace lib_algorithms {
#define flt_max 3.402823466e+38F

class DteAlgorithmShared {
 public:
  template <typename T>
  struct Dte_NodeHeader_Train {
    int tracking_id;
    int parent_id;

    int node_index_start;
    int node_index_count;

    int attribute;
    T split_point;
  };

  template <typename T>
  struct Dte_NodeHeader_Classify {
    int child_start;
    int child_count;
    int probability_start;

    int attribute;
    T split_point;
  };
};
}