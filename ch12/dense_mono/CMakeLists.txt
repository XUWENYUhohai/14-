cmake_minimum_required(VERSION 2.8)
project(dense_monocular)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${OpenCV_LIBS})

add_executable(semi_dense_mapping semi_dense_mapping.cpp)
target_link_libraries(semi_dense_mapping ${OpenCV_LIBS})