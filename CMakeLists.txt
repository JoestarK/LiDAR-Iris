cmake_minimum_required (VERSION 3.8)

SET(CMAKE_BUILD_TYPE Release)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
set(OpenCV_DIR "/usr/local/share/OpenCV")

project (LidarIris)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories("./LidarIris/")

add_executable(demo "demo.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_link_libraries(demo ${PCL_LIBRARIES} ${OpenCV_LIBS})
