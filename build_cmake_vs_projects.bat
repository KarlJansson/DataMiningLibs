if not exist ..\BuildFiles\ (
  mkdir ..\BuildFiles\
)

if not exist ..\BuildFiles\public_datamininglibs\ (
  mkdir ..\BuildFiles\public_datamininglibs\
)

cd ..\BuildFiles\public_datamininglibs\
cmake -G "Visual Studio 14 2015 Win64" ../../public_datamininglibs/
cd ..\..\public_datamininglibs\