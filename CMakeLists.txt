# required cmake version
cmake_minimum_required(VERSION 2.8)
project(datamininglibs C CXX)

find_package(GTest)
find_package(CUDA)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_64_BIT_DEVICE_CODE ON)

if(CUDA_FOUND)
	add_definitions(-DCuda_Found)
else(CUDA_FOUND)
	message("Could not find the Cuda toolkit, the Cuda algorithms will not be built.")
endif(CUDA_FOUND)

add_definitions(-DUNICODE -D_UNICODE)
if(WIN32)
	add_definitions(-DWindowsBuild)
	set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -MP" )
	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
	-gencode arch=compute_30,code=sm_30;
	-g;-G;-lineinfo)
	set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};
	-gencode arch=compute_20,code=sm_20;
	-gencode arch=compute_30,code=sm_30;
	-gencode arch=compute_35,code=sm_35;
	#-gencode arch=compute_50,code=sm_50;
	#-gencode arch=compute_60,code=sm_60
	)
	set(CUDA_NVCC_FLAGS_MINSIZEREL ${CUDA_NVCC_FLAGS_MINSIZEREL};
	-gencode arch=compute_20,code=sm_20;
	-gencode arch=compute_30,code=sm_30;
	-gencode arch=compute_35,code=sm_35;
	#-gencode arch=compute_50,code=sm_50;
	#-gencode arch=compute_60,code=sm_60
	)

	set(TBB_LIBRARIES
		optimized ${TBB_LIBRARY_DIR}/tbb${TBB_LIBRARY_NAME}
		debug ${TBB_LIBRARY_DIR}/tbb_debug${TBB_LIBRARY_NAME}
	)
else(WIN32)
	set(TBB_INCLUDE_DIR "/usr/include")
	set(TBB_LIBRARY_DIR "/usr/lib")
	add_definitions(-DUnixBuild)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )
	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
	-gencode arch=compute_20,code=sm_20;
	-gencode arch=compute_30,code=sm_30;
	-gencode arch=compute_35,code=sm_35;
	-gencode arch=compute_50,code=sm_50;
	-gencode arch=compute_60,code=sm_60;
	-gencode arch=compute_61,code=sm_61;
	-std=c++11)
	
	set(TBB_LIBRARIES
		${TBB_LIBRARY_DIR}/libtbb.so
	)
endif(WIN32)

set (LIBRARY_OUTPUT_PATH Build_Output/Libs CACHE PATH "Lib path")
set (EXECUTABLE_OUTPUT_PATH Build_Output/bin CACHE PATH "Exe path")
mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

add_subdirectory(./lib_core)
add_subdirectory(./lib_algorithms)
add_subdirectory(./lib_julia)
add_subdirectory(./lib_parsing)
add_subdirectory(./lib_models)
add_subdirectory(./lib_data)
add_subdirectory(./lib_gpu)
add_subdirectory(./lib_ensembles)
add_subdirectory(./lib_preprocess)

if(CUDA_FOUND)
	add_subdirectory(./lib_cuda_algorithms)
endif(CUDA_FOUND)

if(WIN32)
	add_subdirectory(./runnable_preprocess)
	add_subdirectory(./runnable_benchmark)
endif(WIN32)

if(GTEST_FOUND)
	add_subdirectory(./runnable_test)
endif(GTEST_FOUND)
