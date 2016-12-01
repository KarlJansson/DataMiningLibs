#pragma once
#include "lib_algorithms.h"
#include "lib_data.h"
#include "lib_ensembles.h"
#include "lib_models.h"
#include "lib_parsing.h"

extern "C" float** experiment_1(char* data, int kNrTrees, int kMaxDepth,
                                int kAlgoType, bool kBagging);

extern "C" float** experiment_2(char* data1, char* data2, int kNrTrees,
                                int kMaxDepth, int kAlgoType, bool kBagging);

extern "C" float** free_memory(float** data, int rows);