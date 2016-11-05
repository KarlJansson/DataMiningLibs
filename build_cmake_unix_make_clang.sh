rm -rf __project_files_cmake_unix64__
mkdir __project_files_cmake_unix64__
cd __project_files_cmake_unix64__
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
cmake -D_CMAKE_TOOLCHAIN_PREFIX=llvm- -DCMAKE_USER_MAKE_RULES_OVERRIDE=./ClangOverrides.txt -G "Unix Makefiles" ../
cd ..
