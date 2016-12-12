rem __project_files_cmake_win64__
mkdir __project_files_cmake_win64__& pushd __project_files_cmake_win64__
cmake.exe -DTBB_INCLUDE_DIR="D:/API/tbb/include" -DTBB_LIBRARY_DIR="D:/API/tbb/lib/intel64/vc14" -DTBB_LIBRARY_NAME=".lib" -G "Visual Studio 14 2015 Win64" ../
popd