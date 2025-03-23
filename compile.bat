@echo off

:: Change directory to the new build folder
cd "C:\Users\gmyst\Desktop\Work\Delft\LiteRT-Project\testing\build"

:: Run CMake with Visual Studio generator
cmake -G "Visual Studio 17 2022" ..

:: Restore cd
cd "C:\Users\gmyst\Desktop\Work\Delft\LiteRT-Project"

SET SOLUTION_PATH=C:\Users\gmyst\Desktop\Work\Delft\LiteRT-Project\testing\build\TFLiteCheck.sln
SET MSBUILD_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe
SET BUILD_CONFIGURATION=Release
SET PLATFORM=x64

REM Run MSBuild with specified parameters
"%MSBUILD_PATH%" "%SOLUTION_PATH%" /p:Configuration=%BUILD_CONFIGURATION% /p:Platform=%PLATFORM%
