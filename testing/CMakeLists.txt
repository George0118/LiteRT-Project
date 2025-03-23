cmake_minimum_required(VERSION 3.17)
project(TFLiteCheck)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_0/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_1/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_2/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_3/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_4/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_5/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_6/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_7/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_8/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../models/model_result/model_9/)

# lib has 3 file: tensorflowlite.dll.if.lib flatbuffers.lib tensorflowlite.dll
find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)
find_library(FLATBUFFERS flatbuffers  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)

set(OpenCV_DIR "C:/Users/gmyst/Desktop/Tools/opencv/build")
set(CMAKE_PREFIX_PATH "C:/Users/gmyst/Desktop/Tools/opencv/build")
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

set(SOURCES
    main.cpp  # Your main program
)

add_executable(TFLiteCheck ${SOURCES})
target_link_libraries(TFLiteCheck PRIVATE ${TFLITE} ${FLATBUFFERS} ${OpenCV_LIBS})

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/tensorflowlite.dll ${CMAKE_BINARY_DIR}/Release/
)