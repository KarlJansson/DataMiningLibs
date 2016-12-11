#pragma once
#include <atomic>

#include "lib_julia.h"

namespace lib_julia {
char csv_data[] = {
    "a1,a2,a3,a4,a5,a6,a7,a8,a9,a10\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,"
    "1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,"
    "4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,"
    "0"};

auto &julia_face = JuliaResources::get();

TEST(lib_julia, julia_resource_test) {
  auto data_id = julia_face.SaveDataset<float>(csv_data);
  auto dataset = julia_face.GetDataset<float>(data_id);
  julia_face.RemoveDataset(data_id);
}
}