Datamining libs
=====================
Datamining libraries for decision tree ensembles running on GPU's and multi-core CPU's.

Requirements
---------------------
    Compiler with c++11 support (GCC 5.4, Clang 3.8, MSVC 2013)
    Cmake: https://cmake.org/download/
    TBB: https://www.threadingbuildingblocks.org

    Optional requirement for compilation of the tests:
    Google test: https://github.com/google/googletest
	The latest Cuda toolkit: https://developer.nvidia.com/cuda-toolkit

Compiling on Windows
---------------------
Build the visual studio projects using the following script:
    
    ./build_cmake_vs_projects.bat

Browse into the newly created __project_files_cmake_win64__ directory where you will find the solution file that can be opened and compiled in visual studio.

Compiling on Unix
---------------------
Tested on Ubuntu 16.04 LTS (GCC 5.4 and Clang 3.8) with Cuda toolkit 8.0

Build the make files using one of the following script:
    
    ./build_cmake_unix_make_gcc.sh
	./build_cmake_unix_make_clang.sh

Browse into the newly created __project_files_cmake_unix64__ directory and execute the make command. 