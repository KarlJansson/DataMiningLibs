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

for /F "tokens=2 delims==" %%s in ('set libs[') do if exist ..\%%s\Build_Output\Libs\Debug\%%s.dll (
    xcopy ..\%%s\Build_Output\Libs\Debug\%%s.dll .\Build_Output\bin\Debug\ /Y
)
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist ..\%%s\Build_Output\Libs\Release\%%s.dll (
    xcopy ..\%%s\Build_Output\Libs\Release\%%s.dll .\Build_Output\bin\Release\ /Y
)
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist ..\%%s\Build_Output\Libs\MinSizeRel\%%s.dll (
    xcopy ..\%%s\Build_Output\Libs\MinSizeRel\%%s.dll .\Build_Output\bin\MinSizeRel\ /Y
)

xcopy F:\APIs\TBB\bin\intel64\vc14\tbb_debug.dll .\Build_Output\bin\Debug\ /Y
xcopy F:\APIs\TBB\bin\intel64\vc14\tbb.dll .\Build_Output\bin\Release\ /Y
xcopy F:\APIs\TBB\bin\intel64\vc14\tbb.dll .\Build_Output\bin\MinSizeRel\ /Y