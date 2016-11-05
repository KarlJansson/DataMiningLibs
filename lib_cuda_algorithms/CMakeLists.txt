# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
	../lib_models/include
	../lib_data/include
	../lib_gpu/include
	../lib_core/include
	../lib_parsing/include
	../lib_algorithms/include
)

set(cpp_files
	./test/test_gpurf.h
	./test/test_gpuert.h
	./test/test_hybrid_rf.h
	./test/test_hybrid_ert.h
	./test/test_resources.h

	../source_shared/precomp.h
	./include/lib_cuda_algorithms.h
	./include/cuda_algorithms_interface.h	
	./source/gpurf.h
	./source/gpudte.h
	./source/gpuert.h
	./source/gpudte_algorithm.h
	./source/gpudte_algorithm_shared.h

	./source/gpurf.cu
	./source/gpudte.cu
	./source/gpuert.cu
	./source/gpudte_algorithm.cc
	./source/gpudte_algorithm_shared.cc
	../source_shared/precomp.cc
	./source/cuda_algorithms_interface.cc
)

source_group(source FILES
	./source/gpurf.h
	./source/gpudte.h
	./source/gpuert.h
	./source/gpudte_algorithm.h
	./source/gpudte_algorithm_shared.h

	./source/gpurf.cu
	./source/gpudte.cu
	./source/gpuert.cu
	./source/gpudte_algorithm.cc
	./source/gpudte_algorithm_shared.cc
)

source_group(interface FILES
	./include/lib_cuda_algorithms.h
	./include/cuda_algorithms_interface.h

	./source/cuda_algorithms_interface.cc
)

source_group(test FILES
	./test/test_gpurf.h
	./test/test_gpuert.h
	./test/test_hybrid_rf.h
	./test/test_hybrid_ert.h
	./test/test_resources.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

include_directories( 
	${TBB_INCLUDE_DIR}
)

CUDA_ADD_LIBRARY(lib_cuda_algorithms SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_cuda_algorithms PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

include_directories( 
	${dir_list}
	${CUDA_INCLUDE_DIRS}
)

target_link_libraries(lib_cuda_algorithms
	${CUDA_cusparse_LIBRARY}
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}

	lib_models
	lib_data
	lib_parsing
	lib_gpu
	lib_core
	lib_algorithms
)

if (MSVC)
   set_target_properties(lib_cuda_algorithms PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)
