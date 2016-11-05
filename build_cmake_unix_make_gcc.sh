rm -rf __project_files_cmake_unix64__
mkdir __project_files_cmake_unix64__
cd __project_files_cmake_unix64__
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cmake -G "Unix Makefiles" ../
cd ..
