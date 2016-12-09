#pragma once
#include "lib_algorithms.h"
#include "lib_data.h"
#include "lib_ensembles.h"
#include "lib_models.h"
#include "lib_parsing.h"

extern "C" {
DLLExport float** experiment_1(char* data, int kNrTrees, int kMaxDepth,
                               int kAlgoType, bool kBagging);

DLLExport float** experiment_2(char* data1, char* data2, int kNrTrees,
                               int kMaxDepth, int kAlgoType, bool kBagging);

DLLExport void free_memory(float** data, int rows);
}