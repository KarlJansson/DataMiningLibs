rem __project_files_cmake_win64__
mkdir __project_files_cmake_win64__& pushd __project_files_cmake_win64__
cmake.exe -DTBB_INCLUDE_DIR="F:/APIs/TBB/include" -DTBB_LIBRARY_DIR="F:/APIs/TBB/lib/intel64/vc14" -G "Visual Studio 14 2015 Win64" ../
popd