# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
	${TBB_INCLUDE_DIR}
	../lib_models/include
	../lib_data/include
	../lib_gpu/include
	../lib_core/include
	../lib_parsing/include
	../lib_algorithms/include
	../lib_ensembles/include
)

set(cpp_files
	./test/test_main_interface.h

	../source_shared/precomp.h
	./include/lib_julia.h
	./include/julia_interface.h

	../source_shared/precomp.cc
	./source/julia_interface.cc
)

source_group(source FILES
)

source_group(interface FILES
	./include/lib_julia.h
	./include/julia_interface.h

	./source/julia_interface.cc
)

source_group(test FILES
	./test/test_main_interface.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

add_library(lib_julia SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_julia PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_julia
	${CUDA_cusparse_LIBRARY}
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}
	
	lib_models
	lib_data
	lib_parsing
	lib_gpu
	lib_core
	lib_algorithms
	lib_ensembles
)

include_directories(lib_julia
	${dir_list}
	${CUDA_INCLUDE_DIRS}
)

if (MSVC)
   set_target_properties(lib_julia PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)
