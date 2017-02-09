rem datamininglibs_x64
mkdir datamininglibs_x64& pushd datamininglibs_x64
mkdir include
mkdir lib
mkdir bin
popd

set build_dir=..\BuildFiles\public_datamininglibs

set libs[0]=lib_algorithms
set libs[1]=lib_data
set libs[2]=lib_julia
set libs[3]=lib_models
set libs[4]=lib_parsing
set libs[5]=lib_gpu
set libs[6]=lib_core
set libs[7]=lib_ensembles
set libs[8]=lib_cuda_algorithms
set libs[9]=lib_preprocess

xcopy /s ".\source_shared\include\*" ".\datamininglibs_x64\include\" /Y
for /F "tokens=2 delims==" %%s in ('set libs[') do xcopy /s "%build_dir%\%%s\include\*" ".\datamininglibs_x64\include\%%s\include\" /Y
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist %build_dir%\%%s\Build_Output\Libs\Release\%%s.lib (
	xcopy "%build_dir%\%%s\Build_Output\Libs\Release\%%s.lib" ".\datamininglibs_x64\lib\" /Y
)
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist %build_dir%\%%s\Build_Output\Libs\Release\%%s.dll (
	xcopy "%build_dir%\%%s\Build_Output\Libs\Release\%%s.dll" ".\datamininglibs_x64\bin\" /Y
)