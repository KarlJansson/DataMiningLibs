# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
	../lib_data/include
	../lib_core/include
	${TBB_INCLUDE_DIR}
)

set(cpp_files
	./test/test_preprocess.h

	../source_shared/precomp.h
	./include/lib_preprocess.h
	./include/preprocess_interface.h
	./include/document.h
	./source/document_impl.h

	./source/document_impl.cc
	../source_shared/precomp.cc
	./source/preprocess_interface.cc
	./source/document.cc
)

source_group(source FILES
	./source/document_impl.h

	./source/document_impl.cc
)

source_group(interface FILES
	./include/document.h
	./source/document.cc
	./include/preprocess_interface.h
	./source/preprocess_interface.cc
	./include/lib_preprocess.h
)

source_group(test FILES
	./test/test_preprocess.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

add_library(lib_preprocess SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_preprocess PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_preprocess
	lib_data
	lib_core
)

include_directories(lib_preprocess
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_preprocess PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)
