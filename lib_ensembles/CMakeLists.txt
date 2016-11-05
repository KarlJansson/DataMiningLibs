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
	../lib_cuda_algorithms/include
	${TBB_INCLUDE_DIR}
)

set(cpp_files
	./test/test_cpuert.h
	./test/test_cpurf.h
	./test/test_resources.h

	../source_shared/precomp.h
	./include/lib_ensembles.h
	./include/ensembles_interface.h	
	./source/cpudte.h
	./source/cpuert.h
	./source/cpurf.h

	../source_shared/precomp.cc
	./source/ensembles_interface.cc
	./source/cpudte.cc
	./source/cpuert.cc
	./source/cpurf.cc
)

source_group(source FILES
	./source/cpudte.h
	./source/cpuert.h
	./source/cpurf.h

	./source/cpudte.cc
	./source/cpuert.cc
	./source/cpurf.cc
)

source_group(interface FILES
	./include/lib_ensembles.h
	./include/ensembles_interface.h

	./source/ensembles_interface.cc
)

source_group(test FILES
	./test/test_cpuert.h
	./test/test_cpurf.h
	./test/test_resources.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

include_directories( 
	${dir_list}
)

ADD_LIBRARY(lib_ensembles SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_ensembles PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_ensembles 
	lib_models
	lib_data
	lib_parsing
	lib_gpu
	lib_core
	lib_algorithms
	lib_cuda_algorithms
)

if (MSVC)
   set_target_properties(lib_ensembles PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)
