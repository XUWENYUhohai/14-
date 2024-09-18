# cmake_minimum_required(VERSION 2.8)
# project(coordinateTransform)

include_directories("/usr/include/eigen3")

add_executable(coordinateTransform coordinateTransform.cpp)

#添加Pangolin依赖
#find_package(Pangolin REQUIRED)
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})