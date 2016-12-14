#pragma once

extern "C" {
/*
        Result getter functions
*/
DLLExport float get_prediction(int result_id, int sample, int target);
DLLExport float get_target(int result_id, int sample);
DLLExport int get_confusion_matrix(int result_id, int x, int y);
DLLExport int get_nr_targets(int result_id);
DLLExport int get_nr_samples(int result_id);
DLLExport float get_accuracy(int result_id);
DLLExport float get_auc(int result_id);
DLLExport float get_mse(int result_id);

/*
        Load and unload functions
*/
DLLExport int load_dataset(char* data);
DLLExport int load_model(char* model_path);
DLLExport void save_model(int model_id, char* save_path);

DLLExport void remove_dataset(int id);
DLLExport void remove_model(int id);
DLLExport void remove_result(int id);

/*
        Algorithm functions
*/
DLLExport int gpurf_fit(int dataset, int nr_trees, int max_depth,
                        bool classification = true);
DLLExport int gpurf_predict(int dataset, int model, bool classification = true);

DLLExport int gpuert_fit(int dataset, int nr_trees, int max_depth,
                         bool classification = true);
DLLExport int gpuert_predict(int dataset, int model,
                             bool classification = true);

DLLExport int cpurf_fit(int dataset, int nr_trees, int max_depth,
                        bool classification = true);
DLLExport int cpurf_predict(int dataset, int model, bool classification = true);

DLLExport int cpuert_fit(int dataset, int nr_trees, int max_depth,
                         bool classification = true);
DLLExport int cpuert_predict(int dataset, int model,
                             bool classification = true);

DLLExport int hybridrf_fit(int dataset, int nr_trees, int max_depth,
                           bool classification = true);
DLLExport int hybridrf_predict(int dataset, int model,
                               bool classification = true);

DLLExport int hybridert_fit(int dataset, int nr_trees, int max_depth,
                            bool classification = true);
DLLExport int hybridert_predict(int dataset, int model,
                                bool classification = true);
}