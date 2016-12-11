#pragma once
#include <atomic>

#include "lib_julia.h"

namespace lib_julia {
char *csv_data =
    "a1,a2,a3,a4,a5,a6,a7,a8,a9,a10\na,1,2,3,4,5,6,7,8,0\nb,1,2,3,4,5,6,7,8,"
    "1\nc,1,2,3,4,5,6,7,8,0\ne,1,2,3,4,5,6,7,8,1\nf,1,2,3,4,5,6,7,8,0\nd,1,2,3,"
    "4,5,6,7,8,1\ng,1,2,3,4,5,6,7,8,0\nh,1,2,3,4,5,6,7,8,1\ni,1,2,3,4,5,6,7,8,"
    "0";

char *test =
    "word_freq_make,word_freq_address,word_freq_all,word_freq_3d,word_freq_our,"
    "word_freq_over,word_freq_remove,word_freq_internet,word_freq_order,word_"
    "freq_mail,word_freq_receive,word_freq_will,word_freq_people,word_freq_"
    "report,word_freq_addresses,word_freq_free,word_freq_business,word_freq_"
    "email,word_freq_you,word_freq_credit,word_freq_your,word_freq_font,word_"
    "freq_000,word_freq_money,word_freq_hp,word_freq_hpl,word_freq_george,word_"
    "freq_650,word_freq_lab,word_freq_labs,word_freq_telnet,word_freq_857,word_"
    "freq_data,word_freq_415,word_freq_85,word_freq_technology,word_freq_1999,"
    "word_freq_parts,word_freq_pm,word_freq_direct,word_freq_cs,word_freq_"
    "meeting,word_freq_original,word_freq_project,word_freq_re,word_freq_edu,"
    "word_freq_table,word_freq_conference,char_freq_;,char_freq_(,char_freq_[,"
    "char_freq_!,char_freq_$,char_freq_#,capital_run_length_average,capital_"
    "run_length_longest,capital_run_length_total,CLASS\r\n"
    "0, 0.64, 0.64, 0, 0.32, 0, 0, 0, 0, 0, 0, 0.64, 0, 0, 0, 0.32, 0, 1.29, "
    "1.93, 0, 0.96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.778, 0, 0, 3.756, 61, 278, 1\r\n"
    "0.21, 0.28, 0.5, 0, 0.14, 0.28, 0.21, 0.07, 0, 0.94, 0.21, 0.79, 0.65, "
    "0.21, 0.14, 0.14, 0.07, 0.28, 3.47, 0, 1.59, 0, 0.43, 0.43, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.132, "
    "0, 0.372, 0.18, 0.048, 5.114, 101, 1028, 1\r\n"
    "0.06, 0, 0.71, 0, 1.23, 0.19, 0.19, 0.12, 0.64, 0.25, 0.38, 0.45, 0.12, "
    "0, 1.75, 0.06, 0.06, 1.03, 1.36, 0.32, 0.51, 0, 1.16, 0.06, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0, 0, 0.12, 0, 0.06, 0.06, 0, 0, "
    "0.01, 0.143, 0, 0.276, 0.184, 0.01, 9.821, 485, 2259, 1\r\n";

auto &julia_face = JuliaResources::get();

TEST(lib_julia, julia_resource_test) {
  auto data_id = julia_face.SaveDataset<float>(test);
  auto dataset = julia_face.GetDataset<float>(data_id);
  julia_face.RemoveDataset(data_id);
}
}