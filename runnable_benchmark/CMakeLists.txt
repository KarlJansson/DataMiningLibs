# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./
	../source_shared
	../source_shared/include
	../lib_algorithms/source
	../lib_julia/source
	../lib_parsing/source
	../lib_models/source
	../lib_data/source
	../lib_gpu/source
	../lib_core/source
	../lib_ensembles/source
	../lib_algorithms/test
	../lib_julia/test
	../lib_parsing/test 
	../lib_models/test
	../lib_data/test
	../lib_gpu/test
	../lib_core/test
	../lib_ensembles/test
	../lib_algorithms/include
	../lib_julia/include
	../lib_parsing/include
	../lib_models/include
	../lib_data/include
	../lib_gpu/include
	../lib_core/include
	../lib_ensembles/include
	${TBB_INCLUDE_DIR}
)

set(cpp_files
	benchmark_main.cc
)

source_group(source FILES
	benchmark_main.cc
)

add_executable(runnable_benchmark ${cpp_files})
if(UNIX)
	target_compile_features(runnable_benchmark PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(runnable_benchmark
	lib_algorithms
	lib_julia
	lib_parsing
	lib_models
	lib_data
	lib_gpu
	lib_core
	lib_ensembles
)

include_directories(runnable_benchmark
	${dir_list}
)

if(WIN32)
	add_custom_command(TARGET runnable_benchmark POST_BUILD
		COMMAND ../../runnable_test/copy_shared_libs.bat
	)
endif(WIN32)
