#include "precomp.h"

#include "document.h"

namespace lib_preprocess {
Document::Document() : nr_samples_(-1), target_col_(-1), nr_features_(-1) {}
Document::~Document() {}

int Document::NrFeatures() { return nr_features_; }
int Document::NrSamples() { return nr_samples_; }
int Document::TargetCol() { return target_col_; }
}